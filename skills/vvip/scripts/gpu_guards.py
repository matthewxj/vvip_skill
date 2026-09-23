#!/usr/bin/env python3
"""Negative controls on an isolated enforce server with max-num-seqs=1.

Uses an operator-configured disable-file path; never overwrites an existing switch.
These checks prove request behavior, not a long-duration memory-leak bound.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import uuid

from gpu_smoke import read_events, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', required=True)
    p.add_argument('--server-log', required=True, type=Path)
    p.add_argument('--disable-file', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.out.exists() or args.disable_file.exists() or not args.disable_file.is_absolute():
        p.error('new output and absent absolute disable-file required')
    ready = [e for e in read_events(args.server_log.read_text()) if e['event'] == 'ready']
    if (len(ready) != 1 or ready[0]['settings']['mode'] != 'enforce'
            or ready[0]['settings']['disable_file'] != str(args.disable_file)):
        p.error('server must enforce with this exact configured switch and a fresh log')
    run_id = 'vvip-guards-' + uuid.uuid4().hex
    offset = args.server_log.stat().st_size
    records = []
    error = None
    try:
        for case, incoming_priority in [('equal', 0), ('reverse', 10), ('disabled', -10)]:
            switch = None
            try:
                if case == 'disabled':
                    switch = args.disable_file.open('x')
                    switch.close()
                first = threading.Event()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    running = pool.submit(stream, args.base_url, args.model, run_id+'-'+case+'-running',
                                          0, 128, 180, first)
                    if not first.wait(180):
                        raise TimeoutError('running request never emitted a token')
                    incoming = stream(args.base_url,args.model,run_id+'-'+case+'-incoming',
                                      incoming_priority,16,180)
                    low = running.result(timeout=180)
                if low['finish_reason'] != 'length' or incoming['finish_reason'] != 'length':
                    raise AssertionError('negative control did not complete')
                records.append(dict(case=case, running=low, incoming=incoming))
            finally:
                if switch is not None:
                    args.disable_file.unlink()
        # Close a real output stream after its first token, then reuse the service.
        headers = {'Content-Type':'application/json'}
        if os.environ.get('VVIP_API_KEY'):
            headers['Authorization'] = 'Bearer '+os.environ['VVIP_API_KEY']
        body = dict(model=args.model,prompt='List consecutive integers:',max_tokens=4096,
                    stream=True,ignore_eos=True,return_token_ids=True,
                    request_id=run_id+'-disconnect',priority=0)
        req = urllib.request.Request(args.base_url.rstrip('/')+'/v1/completions',
                                     data=json.dumps(body).encode(),headers=headers)
        seen = False
        with urllib.request.urlopen(req,timeout=180) as response:
            for line in response:
                if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                    chunk=json.loads(line[6:])
                    if any(c.get('token_ids') for c in chunk.get('choices',[])):
                        seen=True
                        break
        if not seen: raise AssertionError('no token before disconnect')
        followup = stream(args.base_url,args.model,run_id+'-followup',0,16,180)
        if followup['finish_reason'] != 'length': raise AssertionError('follow-up failed')
        records.append(dict(case='disconnect',followup=followup))
        with args.server_log.open('rb') as log:
            log.seek(offset)
            events=[e for e in read_events(log.read().decode(errors='replace')) if run_id in json.dumps(e)]
        if any(e['event']=='preempted' for e in events):
            raise AssertionError('unexpected VVIP preemption in negative controls')
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        events=[]
    for record in records:
        for value in record.values():
            if isinstance(value,dict) and 'token_ids' in value:
                ids=value.pop('token_ids');value['token_count']=len(ids)
                value['token_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    report=dict(schema='vvip.guards/v1',passed=error is None,error=error,run_id=run_id,
                runtime_sha256=ready[0]['runtime_sha256'],boot_id=ready[0]['boot_id'],
                records=records,events=events)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as output:json.dump(report,output,indent=2)
    print(json.dumps({'passed':error is None,'error':error}))
    raise SystemExit(0 if error is None else 1)


if __name__=='__main__':main()
