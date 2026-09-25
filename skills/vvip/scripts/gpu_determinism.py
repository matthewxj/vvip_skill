#!/usr/bin/env python3
"""Same-process, same-card repeat controls; output equality is a measured result.

Requires an isolated recompute/enforce server, max-num-seqs=4, prefix cache off,
configured disable file, and current gpu_smoke/benchmark siblings. No backend flags
are changed. This diagnostic is separate from performance measurements.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import uuid

from benchmark import workload
from gpu_smoke import read_events, read_run_events, stream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--model',required=True)
    p.add_argument('--server-log',type=Path,required=True)
    p.add_argument('--disable-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.disable_file.exists() or not a.disable_file.is_absolute():
        p.error('new output and absent absolute disable-file required')
    ready=[e for e in read_events(a.server_log.read_text()) if e['event']=='ready']
    if (len(ready)!=1 or ready[0]['settings']['mode']!='enforce'
            or ready[0]['settings']['action']!='recompute'
            or ready[0]['settings']['disable_file']!=str(a.disable_file)):
        p.error('dedicated recompute/enforce server with this exact switch required')
    run='vvip-repeat-'+uuid.uuid4().hex
    jobs=workload('saturated',0)
    snapshot=a.server_log.stat()
    trials=[];comparisons=[];events=[];error=None;owns=False
    def wave(label):
        start=time.monotonic()+.1
        def one(job):
            target=start+job['offset'];time.sleep(max(0,target-time.monotonic()))
            result=stream(a.base_url,a.model,run+'-'+label+'-'+str(job['index']),
                          job['priority'],job['tokens'],300,prompt=job['prompt'])
            result.update(index=job['index'],tier=job['tier'],dispatch_lag_s=max(0,result['started']-target))
            if result['finish_reason']!='length':raise AssertionError('unexpected terminal')
            if result['dispatch_lag_s']>.1:raise AssertionError('dispatch lag exceeded 100 ms')
            return result
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:return list(pool.map(one,jobs))
    try:
        a.disable_file.touch(exist_ok=False);owns=True
        wave('warm')
        for index,mode in enumerate(['off','off','recompute','off','recompute','recompute']):
            if mode=='off' and not owns:
                a.disable_file.touch(exist_ok=False);owns=True
            if mode=='recompute' and owns:
                a.disable_file.unlink();owns=False
            trials.append(dict(index=index,mode=mode,records=wave(f't{index}')))
        events=read_run_events(a.server_log, snapshot, run, ready[0]['boot_id'])
        for trial in trials:
            related=[e for e in events if f'{run}-t{trial["index"]}-' in json.dumps(e)]
            count=sum(e['event']=='preempted' for e in related)
            if (trial['mode']=='off' and count) or (trial['mode']=='recompute' and not count):
                raise AssertionError('switch/preemption control failed')
            trial['preemptions']=count
        baseline=trials[0]['records']
        for trial in trials[1:]:
            diffs=[]
            for original,row in zip(baseline,trial['records']):
                assert original['index']==row['index']
                before,after=original['token_ids'],row['token_ids']
                first=next((i for i,(x,y) in enumerate(zip(before,after)) if x!=y),None)
                diffs.append(dict(index=row['index'],tier=row['tier'],identical=before==after,
                                  first_difference_zero_based=first))
            comparisons.append(dict(reference=0,trial=trial['index'],mode=trial['mode'],
                                    identical=sum(d['identical'] for d in diffs),compared=len(diffs),records=diffs))
    except Exception as exc:
        error=type(exc).__name__
    finally:
        if owns:a.disable_file.unlink()
    evidence_error=None
    try:
        events=read_run_events(a.server_log, snapshot, run, ready[0]['boot_id'])
    except (RuntimeError, ValueError, OSError) as exc:
        evidence_error=type(exc).__name__
    for trial in trials:
        for row in trial['records']:
            ids=row.pop('token_ids');row['token_count']=len(ids)
            row['token_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    passed=error is None and evidence_error is None
    report=dict(schema='vvip.determinism-diagnostic/v1',request_contract_passed=passed,error=error,
                evidence_error=evidence_error,
                run_id=run,runtime_sha256=ready[0]['runtime_sha256'],boot_id=ready[0]['boot_id'],
                workload_sha256=hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest(),
                trials=trials,comparisons=comparisons,events=events,
                scope='Same process/GPU; bitwise equality is measured, not required for request-contract success')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as output:json.dump(report,output,indent=2)
    print(json.dumps({'request_contract_passed':passed,'error':error,'comparisons':comparisons,
                      'evidence_error':evidence_error}))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
