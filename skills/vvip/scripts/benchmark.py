#!/usr/bin/env python3
"""Bounded, repeated identical arrivals for native/VVIP comparisons (stdlib only).

Run only on an isolated server. Each trial drains before the next; this measures
finite bursts, not an infinite open-loop service capacity. Reports omit text and URLs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request
import uuid

from gpu_smoke import read_events

SCENARIOS = {
    # (ordinary count, ordinary output length, VIP count, context repetitions)
    'spare': (1, 256, 1, 0),
    'saturated': (4, 512, 2, 0),
    'burst': (8, 512, 4, 0),
    'long-prefill': (4, 512, 2, 96),
}


def workload(scenario, trial):
    ordinary, length, vips, context = SCENARIOS[scenario]
    jobs = []
    for index in range(ordinary + vips):
        vip = index >= ordinary
        nonce = hashlib.sha256(f'{scenario}/{trial}/{index}'.encode()).hexdigest()
        prompt = f'{nonce}\n' + ('The warehouse contains red green blue and yellow boxes. ' * context)
        prompt += '\nList consecutive integers starting at 1, separated by commas:'
        jobs.append(dict(index=index, tier='vip' if vip else 'ordinary',
                         priority=-10 if vip else 0, tokens=64 if vip else length,
                         offset=0.5 + 0.25 * (index - ordinary) if vip else 0,
                         prompt=prompt))
    return jobs


def request(base, model, job, request_id, start, timeout):
    target = start + job['offset']
    time.sleep(max(0, target - time.monotonic()))
    result = dict(request_id=request_id, index=job['index'], tier=job['tier'],
                  priority=job['priority'], requested_tokens=job['tokens'],
                  scheduled=target, started=time.monotonic(), token_count=0,
                  prompt_sha256=hashlib.sha256(job['prompt'].encode()).hexdigest(),
                  finish_reason=None, stop_reason=None, error=None)
    result['dispatch_lag_s'] = max(0, result['started'] - target)
    payload = dict(model=model, prompt=job['prompt'], request_id=request_id,
                   priority=job['priority'], max_tokens=job['tokens'], stream=True,
                   stream_options={'include_usage': True}, temperature=0, seed=42,
                   ignore_eos=True, return_token_ids=True)
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('VVIP_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VVIP_API_KEY']
    req = urllib.request.Request(base.rstrip('/') + '/v1/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    tokens, gaps, previous, done, terminals = [], [], None, False, 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                now = time.monotonic()
                if now - result['started'] > timeout:
                    raise TimeoutError('deadline')
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise RuntimeError('SSE error')
                if chunk.get('usage'):
                    result['usage'] = {k: v for k, v in chunk['usage'].items()
                                       if k in {'prompt_tokens', 'completion_tokens', 'total_tokens'}}
                for choice in chunk.get('choices', []):
                    ids = choice.get('token_ids') or []
                    if ids or choice.get('text'):
                        result.setdefault('first_token', now)
                        if previous is not None:
                            gaps.append(now - previous)
                        previous = now
                    tokens.extend(ids)
                    if choice.get('finish_reason'):
                        terminals += 1
                        result['finish_reason'] = choice['finish_reason']
                        result['stop_reason'] = choice.get('stop_reason')
        if not done or terminals != 1:
            raise RuntimeError('missing or duplicate terminal')
        if result['finish_reason'] == 'length' and len(tokens) != job['tokens']:
            raise RuntimeError('incomplete token accounting')
        if result['finish_reason'] not in {'length', 'abort'}:
            raise RuntimeError('unexpected terminal')
        if result['finish_reason'] == 'abort' and result['stop_reason'] != 'vvip_preempted':
            raise RuntimeError('unrecognized abort')
    except Exception as exc:
        # Keep the request in the denominator; do not leak URLs/server text.
        result['error'] = type(exc).__name__
    result.update(ended=time.monotonic(), token_count=len(tokens),
                  token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  terminal_count=terminals, done=done, max_chunk_gap_s=max(gaps, default=0))
    if 'first_token' in result:
        result['ttft_s'] = result['first_token'] - result['started']
        result['arrival_ttft_s'] = result['first_token'] - target
        result['tpot_s'] = ((result['ended'] - result['first_token']) / (len(tokens) - 1)
                            if len(tokens) > 1 else None)
    result['e2e_s'] = result['ended'] - result['started']
    return result


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {'n': 0, 'mean': None, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    return dict(n=len(ordered), mean=statistics.mean(ordered),
                **{f'p{p}': ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]
                   for p in (50, 95, 99)}, max=ordered[-1])


def summarize(records, duration, slo):
    summary = {}
    for tier in ('vip', 'ordinary', 'all'):
        rows = [r for r in records if tier == 'all' or r['tier'] == tier]
        success = [r for r in rows if not r['error'] and r['finish_reason'] == 'length']
        aborts = [r for r in rows if not r['error'] and r['finish_reason'] == 'abort']
        summary[tier] = dict(
            submitted=len(rows), completed=len(success), aborted=len(aborts),
            errors=sum(r['error'] is not None for r in rows),
            completion_rate=len(success) / len(rows) if rows else None,
            ttft_s=quantiles([r['ttft_s'] for r in rows if 'ttft_s' in r]),
            e2e_completed_s=quantiles([r['e2e_s'] for r in success]),
            max_chunk_gap_s=quantiles([r['max_chunk_gap_s'] for r in success]),
            tpot_completed_s=quantiles([r['tpot_s'] for r in success if r.get('tpot_s') is not None]),
            ttft_slo_s=slo,
            slo_completed=sum(r.get('arrival_ttft_s', float('inf')) <= slo for r in success),
            slo_attainment=sum(r.get('arrival_ttft_s', float('inf')) <= slo for r in success) / len(rows) if rows else None,
            emitted_tokens=sum(r['token_count'] for r in rows),
            completed_request_tokens=sum(r['token_count'] for r in success),
            aborted_partial_tokens=sum(r['token_count'] for r in aborts),
            output_tokens_per_s=sum(r['token_count'] for r in rows) / duration,
            completed_request_tokens_per_s=sum(r['token_count'] for r in success) / duration,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True,
                        choices=['native', 'native-async', 'off', 'shadow', 'recompute', 'abort'])
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--slo', type=float, default=1.0)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 100 or not 0 <= args.block < 100:
        parser.error('trials must be 1..100; block must be 0..99')
    if not 0 < args.timeout <= 600 or not math.isfinite(args.slo) or args.slo <= 0:
        parser.error('timeout must be 0..600 and SLO positive finite')
    if args.out.exists():
        parser.error('output already exists')
    run_id = 'vvip-bench-' + uuid.uuid4().hex
    # Warm ordinary/vip prompt lengths and decode kernels before measurement.
    warm = workload(args.scenario, -1)
    with ThreadPoolExecutor(max_workers=len(warm)) as executor:
        start = time.monotonic() + 0.1
        futures = [executor.submit(request, args.base_url, args.model, job,
                   run_id + f'-warm-{job["index"]}', start, args.timeout) for job in warm]
        warm_results = [f.result() for f in futures]
    if any(r['error'] for r in warm_results):
        parser.exit(1, 'warmup request failed; inspect server logs\n')
    offset = args.server_log.stat().st_size
    trials = []
    workload_digests = []
    for index in range(args.trials):
        # Distinct prompts per trial/block; identical across every variant.
        jobs = workload(args.scenario, args.block * 100 + index)
        workload_digests.append(hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest())
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            start = time.monotonic() + 0.05
            futures = [executor.submit(request, args.base_url, args.model, job,
                       run_id + f'-t{index}-{job["index"]}', start, args.timeout) for job in jobs]
            rows = [f.result() for f in futures]
        duration = max(r['ended'] for r in rows) - min(r['started'] for r in rows)
        trials.append(dict(index=index, duration_s=duration, records=rows,
                           summary=summarize(rows, duration, args.slo)))
    with args.server_log.open('rb') as log:
        log.seek(offset)
        events = [e for e in read_events(log.read().decode(errors='replace'))
                  if run_id in json.dumps(e)]
    records = [r for t in trials for r in t['records']]
    errors = [r['request_id'] for r in records if r['error']]
    max_lag = max(r['dispatch_lag_s'] for r in records)
    duration = sum(t['duration_s'] for t in trials)
    event_counts = dict(Counter(e['event'] for e in events))
    invalid = []
    if errors: invalid.append('request errors')
    if max_lag > 0.1: invalid.append('dispatch lag exceeded 100 ms')
    if any(r['finish_reason'] == 'abort' for r in records) and args.variant != 'abort':
        invalid.append('unexpected abort')
    if args.variant in {'native', 'native-async', 'off', 'shadow'} and event_counts.get('preempted'):
        invalid.append('unexpected VVIP mutation')
    if args.scenario == 'spare' and event_counts.get('preempted'):
        invalid.append('unexpected VVIP mutation without pressure')
    if args.scenario != 'spare' and args.variant in {'recompute', 'abort'} and not event_counts.get('preempted'):
        invalid.append('no actual preemption')
    report = dict(schema='vvip.benchmark/v1', run_id=run_id, valid=not invalid,
                  invalid_reasons=invalid, variant=args.variant, scenario=args.scenario,
                  block=args.block, model=args.model, trials=trials,
                  workload_sha256=workload_digests, max_dispatch_lag_s=max_lag,
                  measured_duration_s=duration, summary=summarize(records, duration, args.slo),
                  events=events, event_counts=event_counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps({k: report[k] for k in ('valid', 'variant', 'scenario', 'block', 'event_counts', 'summary')}))
    raise SystemExit(0 if report['valid'] else 1)


if __name__ == '__main__':
    main()
