#!/usr/bin/env python3
"""Combine complete, matched benchmark blocks; reject missing/invalid comparisons."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics

from benchmark import quantiles, summarize


def paired_delta_ci(pairs, seed=42):
    """Bootstrap paired trial median differences, baseline minus treatment, seconds."""
    differences = [a - b for a, b in pairs]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return dict(n_pairs=len(pairs), mean_delta_s=statistics.mean(differences),
                ci95_low_s=draws[49], ci95_high_s=draws[1949], bootstrap_resamples=2000)


def aggregate(reports):
    by_scenario = defaultdict(dict)
    for report in reports:
        if report.get('schema') != 'vvip.benchmark/v1' or not report['valid']:
            raise ValueError('invalid or unsupported report')
        key = (report['block'], report['variant'])
        scenario = report['scenario']
        if key in by_scenario[scenario]:
            raise ValueError('duplicate block/variant/scenario')
        by_scenario[scenario][key] = report
    output = {}
    for scenario, group in sorted(by_scenario.items()):
        variants = sorted({key[1] for key in group})
        blocks = sorted({key[0] for key in group})
        if 'native' not in variants:
            raise ValueError('native synchronous baseline is required')
        values = {}
        for block in blocks:
            base = group.get((block, 'native'))
            if base is None:
                raise ValueError('missing baseline block')
            for variant in variants:
                candidate = group.get((block, variant))
                if candidate is None:
                    raise ValueError('missing variant block')
                if candidate['workload_sha256'] != base['workload_sha256']:
                    raise ValueError('arrival/prompt workload mismatch')
                if candidate['model'] != base['model']:
                    raise ValueError('model alias mismatch')
                if candidate.get('timeouts') != base.get('timeouts'):
                    raise ValueError('timeout policy mismatch')
                if candidate['summary']['vip']['ttft_slo_s'] != base['summary']['vip']['ttft_slo_s']:
                    raise ValueError('SLO mismatch')
        for variant in variants:
            selected = [group[(block, variant)] for block in blocks]
            rows = [row for r in selected for trial in r['trials'] for row in trial['records']]
            duration = sum(r['measured_duration_s'] for r in selected)
            summary = summarize(rows, duration, selected[0]['summary']['vip']['ttft_slo_s'])
            preemptions = [event for r in selected for event in r['events'] if event['event'] == 'preempted']
            summary['measured_duration_s'] = duration
            summary['trials'] = sum(len(r['trials']) for r in selected)
            summary['preemptions'] = len(preemptions)
            summary['interrupted_output_tokens'] = sum(e['generated_tokens'] for e in preemptions)
            summary['recomputed_tokens'] = sum(e['tokens'] for r in selected for e in r['events'] if e['event']=='recomputed')
            # This is generated progress at interruption, NOT measured GPU work.
            if variant != 'native':
                pairs = []
                token_comparisons = []
                for block in blocks:
                    for a, b in zip(group[(block, 'native')]['trials'], group[(block, variant)]['trials']):
                        pairs.append((a['summary']['vip']['ttft_s']['p50'], b['summary']['vip']['ttft_s']['p50']))
                        baseline_rows = {row['index']: row for row in a['records']}
                        for row in b['records']:
                            original = baseline_rows[row['index']]
                            if (row['finish_reason'] == original['finish_reason'] == 'length'
                                    and not row['error'] and not original['error']):
                                token_comparisons.append(row['token_sha256'] == original['token_sha256'])
                if any(a is None or b is None for a,b in pairs):
                    raise ValueError('missing VIP TTFT; cannot compute paired comparison')
                summary['vip_trial_median_ttft_delta'] = paired_delta_ci(pairs)
                summary['completed_token_identity_vs_native'] = dict(
                    compared=len(token_comparisons), identical=sum(token_comparisons),
                    different=len(token_comparisons)-sum(token_comparisons))
            values[variant] = summary
        output[scenario] = dict(blocks=blocks, variants=values)
    return output


def markdown(result):
    lines = ['# Repeated workload results', '',
             'TTFT is seconds; SLO requires successful completion and arrival-to-first-token within the preset threshold.',
             'Output throughput includes partial output; completed-request throughput excludes aborted/failed requests.', '',
             '| Scenario | Variant | VIP n | TTFT p50 / p95 / p99 (s) | VIP SLO | Ordinary complete | Output / completed tok/s |',
             '| --- | --- | ---: | --- | --- | --- | ---: |']
    for scenario, entry in result['scenarios'].items():
        for variant, s in entry['variants'].items():
            vip, ordinary, all_rows = s['vip'], s['ordinary'], s['all']
            q=vip['ttft_s']
            qstr=' / '.join(f'{q[k]:.4f}' if q[k] is not None else 'NA' for k in ('p50','p95','p99'))
            lines.append(f'| {scenario} | {variant} | {vip["submitted"]} | {qstr} | {vip["slo_attainment"]:.1%} | {ordinary["completion_rate"]:.1%} | {all_rows["output_tokens_per_s"]:.2f} / {all_rows["completed_request_tokens_per_s"]:.2f} |')
    lines += ['', 'Quantiles use nearest rank. Small-sample tails are descriptive. Paired bootstrap intervals and all request metrics are in the JSON.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='JSON reports or .jsonl.gz datasets')
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.md').exists():
        parser.error('output already exists')
    reports=[]
    for path in args.reports:
        if path.name.endswith('.jsonl.gz'):
            with gzip.open(path,'rt') as source:
                reports.extend(json.loads(line) for line in source if line.strip())
        else:
            reports.append(json.loads(path.read_text()))
    try:
        result=dict(schema='vvip.benchmark-summary/v1', scenarios=aggregate(reports),
                    evidence_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in args.reports])
    except (ValueError,KeyError) as exc:
        parser.exit(2,f'cannot compare: {exc}\n')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    args.out.with_suffix('.md').write_text(markdown(result))
    print(json.dumps({'scenarios':list(result['scenarios']),'reports':len(reports)}))


if __name__=='__main__': main()
