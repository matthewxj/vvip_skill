"""Regression checks against real local HTTP failure modes; no GPU required."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/vvip/scripts"
sys.path.insert(0, str(SCRIPTS))
from gpu_smoke import read_run_events, stream
from benchmark import request, workload
from completion_client import completion_events


@contextmanager
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if body["model"] == "headers":
                    time.sleep(.7)
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
                def emit(choice):
                    self.wfile.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
                    self.wfile.flush()

                if body["model"] == "trickle":
                    # Each read succeeds before the socket timeout, but there is
                    # no complete SSE line until well after the total deadline.
                    for _ in range(15):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(.08)
                    self.wfile.write(b"\n")
                if body["model"] in {"prefill", "roles"}:
                    for _ in range(6):
                        emit({"delta": {"role": "assistant"}, "text": ""})
                        time.sleep(.06)
                if body["model"] in {"prefill", "idle", "progress", "short-ttft", "complete"}:
                    emit({"token_ids": [1]})
                    if body["model"] == "short-ttft":
                        time.sleep(.25)
                        emit({"token_ids": [2]})
                    if body["model"] in {"idle", "progress"}:
                        for _ in range(15):
                            time.sleep(.06)
                            emit({"token_ids": [2]} if body["model"] == "progress" else {"text": ""})
                    emit({"finish_reason": "length"})
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self.wfile.write(b'data: {"choices":[{"token_ids":[1]}]}\n\n')
                # Deliberately omit the terminal and [DONE].
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class ClientFailures(unittest.TestCase):
    def test_partial_line_cannot_extend_deadline(self):
        with endpoint() as base:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                stream(base, "trickle", "deadline", 0, 1, .25)
            self.assertLess(time.monotonic() - started, .7)
            job = dict(workload("spare", 0)[0], tokens=1)
            started = time.monotonic()
            row = request(base, "trickle", job, "bench-deadline", started, .25)
            self.assertEqual(row["error"], "TimeoutError")
            self.assertLess(row["e2e_s"], .7)

    def test_phase_deadlines_and_successful_terminal(self):
        with endpoint() as base:
            # Role/empty frames must not start the short output-idle budget.
            result = stream(base, "prefill", "long-prefill", 0, 1, 2,
                            ttft_timeout=1, idle_timeout=.15)
            self.assertEqual(result["finish_reason"], "length")
            # A short first-output budget must not become a socket-read budget.
            result = stream(base, "short-ttft", "decode", 0, 2, 2,
                            ttft_timeout=.2, idle_timeout=.6)
            self.assertEqual(result["token_ids"], [1, 2])
            for model, total, ttft, idle, phase in (
                ("headers", .25, 1, 1, "total"),
                ("roles", 2, .2, .1, "first_output"),
                ("idle", 2, 1, .2, "output_idle"),
                ("progress", .25, 1, 1, "total"),
            ):
                with self.subTest(model=model):
                    started = time.monotonic()
                    with self.assertRaisesRegex(TimeoutError, phase):
                        stream(base, model, model, 0, 1, total,
                               ttft_timeout=ttft, idle_timeout=idle)
                    self.assertLess(time.monotonic() - started, .7)
            # Normal terminal and [DONE] remain a successful client outcome.
            self.assertEqual(stream(base, "complete", "complete", 0, 1, 1)["terminal_count"], 1)

    def test_early_client_close_joins_deadline_guard(self):
        with endpoint() as base:
            before = set(threading.enumerate())
            with completion_events(base, dict(model="idle", request_id="close"), 30) as chunks:
                self.assertTrue(next(chunks)["choices"][0]["token_ids"])
                guards = [t for t in threading.enumerate() if t not in before and t.name.endswith("(guard)")]
                self.assertEqual(len(guards), 1)
            self.assertFalse(guards[0].is_alive())

    def test_log_rotation_restart_and_partial_run_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "engine.log"
            path.write_text("old lines\n")
            snapshot = path.stat()
            event = dict(event="preempted", boot_id="boot", victim_id="run-low")
            with path.open("a") as log:
                log.write("vvip_event=" + json.dumps(event) + "\n")
            self.assertEqual(read_run_events(path, snapshot, "run-", "boot"), [event])
            with path.open("a") as log:
                log.write('vvip_event={"event":"ready","boot_id":"new"}\n')
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                read_run_events(path, snapshot, "run-", "boot")
            path.rename(path.with_suffix(".old"))
            path.write_text("replacement" * 100)
            with self.assertRaisesRegex(RuntimeError, "rotated"):
                read_run_events(path, snapshot, "run-")

    def test_smoke_transport_failure_still_writes_failed_report(self):
        with endpoint() as base, tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "server.log"
            log.write_text('vvip_event={"event":"ready","boot_id":"test",'
                           '"runtime_sha256":"test","settings":{"mode":"enforce",'
                           '"action":"recompute"}}\n')
            out = Path(folder) / "report.json"
            result = subprocess.run([
                sys.executable, str(SCRIPTS / "gpu_smoke.py"), "--base-url", base,
                "--model", "incomplete", "--action", "recompute", "--timeout", "1",
                "--server-log", str(log), "--out", str(out),
            ], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(out.exists(), result.stderr)
            report = json.loads(out.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["records"]["baseline"]["error"], "RuntimeError")
            # A warmup failure is also evidence, never a missing report or a
            # valid performance result that silently excludes failed requests.
            bench_out = Path(folder) / "bench.json"
            bench = subprocess.run([
                sys.executable, str(SCRIPTS / "benchmark.py"), "--base-url", base,
                "--model", "incomplete", "--variant", "native", "--scenario", "spare",
                "--timeout", "1", "--server-log", str(log), "--out", str(bench_out),
            ], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(bench.returncode, 0)
            self.assertFalse(json.loads(bench_out.read_text())["valid"])

    def test_guards_preserve_events_on_client_failure(self):
        from unittest.mock import patch
        import gpu_guards
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            log, out, switch = path / "engine.log", path / "report.json", path / "disabled"
            ready = dict(event="ready", boot_id="boot", runtime_sha256="runtime",
                         settings=dict(mode="enforce", disable_file=str(switch)))
            log.write_text("vvip_event=" + json.dumps(ready) + "\n")
            def fake_stream(base, model, request_id, priority, tokens, timeout, first=None):
                event = dict(event="preempted", boot_id="boot", victim_id=request_id)
                with log.open("a") as target:
                    target.write("vvip_event=" + json.dumps(event) + "\n")
                if first:
                    first.set()
                raise RuntimeError("private server response must not enter reports")
            argv = ["gpu_guards.py", "--model", "test", "--server-log", str(log),
                    "--disable-file", str(switch), "--out", str(out)]
            with patch.object(sys, "argv", argv), patch.object(gpu_guards, "stream", fake_stream):
                with self.assertRaises(SystemExit):
                    gpu_guards.main()
            report = json.loads(out.read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["events"])
            self.assertNotIn("private server response", out.read_text())


if __name__ == "__main__":
    unittest.main()
