"""Small end-to-end client and accounting checks; no GPU evidence is implied."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/vvip/scripts'))
from benchmark import workload, request, summarize, quantiles


class BenchmarkChecks(unittest.TestCase):
    def test_reproducible_trace_and_denominators(self):
        self.assertEqual(workload('saturated', 3), workload('saturated', 3))
        self.assertNotEqual(workload('saturated', 3)[0]['prompt'], workload('saturated', 4)[0]['prompt'])
        jobs = workload('burst', 0)
        self.assertEqual(sum(j['tier'] == 'vip' for j in jobs), 4)
        self.assertEqual(quantiles(range(1, 101))['p95'], 95)
        common = dict(tier='ordinary', max_chunk_gap_s=0.1, e2e_s=1, ttft_s=.1,
                      arrival_ttft_s=.1, tpot_s=.02)
        records = [dict(common, error=None, finish_reason='length', token_count=8),
                   dict(common, error=None, finish_reason='abort', token_count=2),
                   dict(common, error='TimeoutError', finish_reason=None, token_count=1)]
        s = summarize(records, duration=2, slo=1)['ordinary']
        self.assertEqual((s['submitted'],s['completed'],s['aborted'],s['errors']), (3,1,1,1))
        self.assertEqual(s['completion_rate'], 1/3)
        self.assertEqual(s['slo_attainment'], 1/3)
        self.assertEqual(s['output_tokens_per_s'], 5.5)
        self.assertEqual(s['completed_request_tokens_per_s'], 4)
        self.assertEqual(s['aborted_partial_tokens'], 2)

    def test_matched_analysis_rejects_missing_and_mismatched_work(self):
        from analyze_benchmark import aggregate, paired_delta_ci
        from copy import deepcopy
        row = dict(tier='vip', error=None, finish_reason='length', token_count=8,
                   ttft_s=2., arrival_ttft_s=2., e2e_s=3., max_chunk_gap_s=.1, tpot_s=.1,
                   index=0,token_sha256='same')
        trial = dict(records=[row], summary=summarize([row],3.,1.))
        base = dict(schema='vvip.benchmark/v1', valid=True, block=0, variant='native',
                    scenario='spare', workload_sha256=['same'], model='test',
                    summary=trial['summary'], trials=[trial], measured_duration_s=3., events=[])
        treatment=deepcopy(base); treatment['variant']='recompute'
        result=aggregate([base,treatment])
        self.assertEqual(result['spare']['variants']['recompute']['vip_trial_median_ttft_delta']['mean_delta_s'],0)
        self.assertEqual(result['spare']['variants']['recompute']['completed_token_identity_vs_native']['identical'],1)
        treatment['trials'][0]['records'][0]['token_sha256']='changed'
        self.assertEqual(aggregate([base,treatment])['spare']['variants']['recompute']['completed_token_identity_vs_native']['different'],1)
        treatment['workload_sha256']=['different']
        with self.assertRaisesRegex(ValueError,'workload mismatch'): aggregate([base,treatment])
        with self.assertRaisesRegex(ValueError,'duplicate'): aggregate([base,base])
        ci=paired_delta_ci([(2.,1.)]*5)
        self.assertEqual((ci['ci95_low_s'],ci['ci95_high_s']),(1.,1.))
        # The public evidence bundle is compressed JSON Lines; verify that its
        # CLI path computes the same data, rather than only testing in-memory input.
        import gzip,subprocess,tempfile
        full=deepcopy(base)
        full['trials'][0]['records'].append(dict(row,tier='ordinary',index=1))
        other=deepcopy(full);other['variant']='recompute'
        with tempfile.TemporaryDirectory() as folder:
            dataset=Path(folder)/'reports.jsonl.gz';out=Path(folder)/'summary.json'
            with gzip.open(dataset,'wt') as f:
                for report in (full,other):f.write(json.dumps(report)+'\n')
            script=Path(__file__).resolve().parents[1]/'skills/vvip/scripts/analyze_benchmark.py'
            result=subprocess.run([sys.executable,str(script),str(dataset),'--out',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(out.read_text())['scenarios'],aggregate([full,other]))

    def test_http_stream_and_incomplete_response(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"token_ids":[1,2],"text":"ok"}]}\n\n')
                if payload['request_id'] != 'broken':
                    self.wfile.write(b'data: {"choices":[{"finish_reason":"length"}]}\n\n')
                    if payload['request_id'] == 'duplicate':
                        self.wfile.write(b'data: {"choices":[{"finish_reason":"length"}]}\n\n')
                    self.wfile.write(b'data: [DONE]\n\n')
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            job = dict(workload('spare',0)[0], tokens=2)
            base = f'http://127.0.0.1:{server.server_port}'
            good = request(base, 'test', job, 'good', time.monotonic(), 2)
            bad = request(base, 'test', job, 'broken', time.monotonic(), 2)
            self.assertIsNone(good['error'])
            self.assertEqual(good['token_count'],2)
            self.assertTrue(good['done'])
            self.assertEqual(bad['error'],'RuntimeError')
            self.assertEqual(summarize([good,bad],1,1)['ordinary']['completion_rate'],.5)
            from gpu_smoke import stream
            self.assertEqual(stream(base,'test','good',0,2,2)['terminal_count'],1)
            event=threading.Event()
            stream(base,'test','good',0,2,2,event,prompt='Synthetic context',first_after_tokens=3)
            self.assertFalse(event.is_set())
            stream(base,'test','good',0,2,2,event,first_after_tokens=2)
            self.assertTrue(event.is_set())
            with self.assertRaisesRegex(RuntimeError,'exactly one terminal'):
                stream(base,'test','duplicate',0,2,2)
            with self.assertRaisesRegex(RuntimeError,'number of token IDs'):
                stream(base,'test','good',0,3,2)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()
