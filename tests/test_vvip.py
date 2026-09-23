"""CPU behavioral checks. Optionally exercise pinned upstream release methods.

VVIP_UPSTREAM_SOURCE=/path/to/vllm python3 -m unittest discover -s tests -v
The lightweight scheduler here is a test double, not GPU integration evidence.
"""
import ast
from collections import deque
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/vvip/scripts"))
from vvip_runtime.compat import check_config, inspect_source
from vvip_runtime.policy import Settings, choose_victim
from vvip_runtime.cli import launch_command


class Status:
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED_ABORTED = 4
    WAITING_FOR_REMOTE_KVS = 5
    WAITING_FOR_STREAMING_REQ = 6
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 7

    @staticmethod
    def is_finished(status):
        return status == Status.FINISHED_ABORTED


class Request(NS):
    __hash__ = object.__hash__

    def is_finished(self):
        return self.status == Status.FINISHED_ABORTED

    def take_events(self):
        return []


def req(name, priority=0):
    return Request(request_id=name, priority=priority, arrival_time=1,
                   num_output_tokens=8, num_computed_tokens=16, max_tokens=128,
                   num_tokens_with_spec=17, num_output_placeholders=0,
                   num_in_flight_tokens=0, num_stale_output_tokens=0,
                   num_preemptions=0, spec_token_ids=[], drop_stale_output=False,
                   resumable=False, has_encoder_inputs=False, client_index=0, trace_headers=None,
                   status=Status.WAITING)


class Queue(list):
    def peek_request(self):
        return min(self, key=lambda r: (r.priority, r.arrival_time, r.request_id))

    def prepend_request(self, request):
        self.append(request)

    def remove_requests(self, requests):
        self[:] = [r for r in self if r not in requests]


class Base:
    def _select_waiting_queue_for_scheduling(self):
        queues = [q for q in (self.waiting, self.skipped_waiting) if q]
        return min(queues, key=lambda q: q.peek_request().priority)

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def _preempt_request(self, request, timestamp):
        self.kv_cache_manager.free(request)
        request.status = Status.PREEMPTED
        request.num_computed_tokens = 0
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def finish_requests(self, request_id, status):
        request = self.requests.pop(request_id)
        self.running.remove(request)
        request.status = status
        self.kv_cache_manager.free(request)
        self.finished_req_ids.add(request_id)
        return [request]

    def schedule(self, throttle_prefills=False):
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            r = self.waiting.peek_request()
            self.waiting.remove(r)
            r.status = Status.RUNNING
            self.running.append(r)
            self.blocks[r.request_id] = [1]
        return NS(num_scheduled_tokens={r.request_id: 1 for r in self.running})

    def update_from_output(self, *args):
        return {}


@contextmanager
def scheduler_class():
    """Stub only the external imports, so the complete VVIP adapter runs."""
    modules = {}
    bindings = {
        "vllm.v1.core.sched.scheduler": {"Scheduler": Base},
        "vllm.v1.core.sched.interface": {"PauseState": NS(UNPAUSED=0)},
        "vllm.v1.request": {"RequestStatus": Status},
        "vllm.v1.engine": {
            "EngineCoreOutput": NS,
            "EngineCoreOutputs": lambda: NS(outputs=[], finished_requests=None),
            "FinishReason": NS(ABORT="abort"),
        },
    }
    for name, values in bindings.items():
        modules[name] = ModuleType(name)
        modules[name].__dict__.update(values)
    with patch.dict(sys.modules, modules):
        sys.modules.pop("vvip_runtime.scheduler", None)
        mod = importlib.import_module("vvip_runtime.scheduler")
        yield mod.VVIPScheduler
    sys.modules.pop("vvip_runtime.scheduler", None)


def harness(cls, action="recompute", mode="enforce"):
    s = cls.__new__(cls)
    s.vvip = Settings(mode=mode, action=action, trigger_wait_ms=0, min_interval_ms=0)
    s._queued, s._first_running, s._counts = {}, {}, {}
    s._events, s._aborted, s._resuming = deque(), [], set()
    s._replay_watermarks, s._replay_step = {}, {}
    s.requests, s.running, s.waiting = {}, [], Queue()
    s.skipped_waiting = Queue()
    s.max_num_running_reqs, s.max_num_scheduled_tokens = 1, 32
    s.scheduler_config = NS(long_prefill_token_threshold=0)
    s.pause_state = 0
    s.prev_step_scheduled_req_ids = {"low"}
    s.reset_preempted_req_ids, s.finished_req_ids = set(), set()
    s.finished_req_ids_dict = None
    s._inflight_prefills = set()
    s.log_stats, s.defer_block_free = False, False
    s.deferred_frees = deque()
    s.blocks = {}
    s.kv_cache_manager = NS(
        coordinator=NS(single_type_managers=[NS(req_to_blocks=s.blocks)]),
        free=lambda r: s.blocks.pop(r.request_id, None))
    s.encoder_cache_manager = NS(free=lambda r: None)
    s._connector_finished = lambda r: (False, None)
    s.ec_connector = None
    s.logged = []
    s._event = lambda event, **fields: s.logged.append({"event": event, **fields})
    low, high = req("low", 0), req("high", -10)
    s.add_request(low)
    s.waiting.remove(low)
    s.running.append(low)
    low.status = Status.RUNNING
    s.blocks["low"] = [1]
    s._first_running["low"] = 100
    s.add_request(high)
    s._queued["high"] = 99
    return s, low, high


class VVIPChecks(unittest.TestCase):
    def test_settings_and_launch(self):
        for values in ({"mode": "auto"}, {"action": "kill"},
                       {"trigger_wait_ms": float("nan")}, {"max_per_request": 0}):
            with self.assertRaises(ValueError):
                Settings(**values)
        self.assertIn("--no-async-scheduling", launch_command("model", ["--port", "8123"]))
        with self.assertRaises(ValueError):
            launch_command("model", ["--async-scheduling=true"])

    def test_policy_strict_priority_and_protection(self):
        high, low, same = req("high", -1), req("low", 0), req("same", -1)
        settings = Settings()
        self.assertIs(choose_victim(high, [same, low], {}, {"low": 0}, 1, settings), low)
        for counts, now in (({"low": 2}, 1), ({}, 30)):
            self.assertIsNone(choose_victim(high, [low], counts, {"low": 0}, now, settings))
        low.num_in_flight_tokens = 1
        self.assertIsNone(choose_victim(high, [low], {}, {"low": 0}, 1, settings))

    def test_both_actions(self):
        with scheduler_class() as cls:
            for action in ("recompute", "abort"):
                with self.subTest(action=action):
                    s, low, high = harness(cls, action)
                    s._maybe_preempt(100, False)
                    self.assertNotIn("low", s.blocks)
                    self.assertNotIn(low, s.running)
                    s._maybe_preempt(100, False)  # cannot target an already moved victim
                    self.assertEqual(len(s.logged), 1)
                    output = s.update_from_output(None, None)
                    if action == "abort":
                        self.assertNotIn("low", s.requests)
                        self.assertEqual(output[0].outputs[0].finish_reason, "abort")
                        self.assertEqual(output[0].outputs[0].stop_reason, "vvip_preempted")
                        self.assertEqual(output[0].finished_requests, {"low"})
                        self.assertEqual(s.update_from_output(None, None), {})
                    else:
                        self.assertIn(low, s.waiting)
                        self.assertEqual(low.num_output_tokens, 8)
                        self.assertEqual(low.num_computed_tokens, 0)
                        self.assertNotIn("low", s.prev_step_scheduled_req_ids)
                        self.assertEqual(output, {})
                    with patch("vvip_runtime.scheduler.time.monotonic", return_value=100):
                        scheduled = s.schedule()
                    self.assertIn("high", scheduled.num_scheduled_tokens)
                    if action == "recompute":
                        Base.finish_requests(s, "high", Status.FINISHED_ABORTED)
                        with patch("vvip_runtime.scheduler.time.monotonic", return_value=100):
                            s.schedule()
                        self.assertIn(low, s.running)
                        self.assertEqual(s.logged[-1]["event"], "resumed")

    def test_no_mutation_and_failure_guards(self):
        with scheduler_class() as cls:
            for mode in ("off", "shadow"):
                s, low, _ = harness(cls, mode=mode)
                s._maybe_preempt(100, False)
                self.assertIn(low, s.running)
                self.assertIn("low", s.blocks)
                self.assertEqual([x["event"] for x in s.logged],
                                 [] if mode == "off" else ["would_preempt"])
            for guard in ("pause", "wait", "rate", "spare", "same", "inflight", "throttle"):
                s, low, high = harness(cls)
                if guard == "pause": s.pause_state = 1
                if guard == "wait": s.vvip = Settings(mode="enforce", trigger_wait_ms=2000)
                if guard == "rate": s._events.extend([99] * 60)
                if guard == "spare": s.max_num_running_reqs = 2
                if guard == "same": high.priority = low.priority
                if guard == "inflight": low.num_in_flight_tokens = 1
                s._maybe_preempt(100, guard == "throttle")
                self.assertIn(low, s.running, guard)
            s, _, _ = harness(cls)
            s.kv_cache_manager.free = lambda r: None
            with self.assertRaisesRegex(RuntimeError, "release invariant"):
                s._maybe_preempt(100, False)
            with tempfile.TemporaryDirectory() as tmp:
                switch = Path(tmp) / "disabled"
                switch.touch()
                s, low, _ = harness(cls)
                s.vvip = Settings(mode="enforce", disable_file=str(switch))
                s._maybe_preempt(100, False)
                self.assertIn(low, s.running)
            for grammar in (None, ValueError("bad grammar"), object()):
                s, low, high = harness(cls)
                high.status = Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
                high.structured_output_request = NS(grammar=grammar)
                s.waiting.remove(high)
                s.skipped_waiting.append(high)
                s._maybe_preempt(100, False)
                self.assertEqual(low in s.running, grammar is None or isinstance(grammar, Exception))

    def test_replay_counts_only_recomputed_positions_after_output(self):
        with scheduler_class() as cls:
            s, low, high = harness(cls)
            s._replay_watermarks["low"] = 16
            low.num_computed_tokens = 12  # native schedule cursor after assigning 8
            with patch.object(Base, "schedule", return_value=NS(num_scheduled_tokens={"low":8})):
                s.schedule()
            self.assertEqual(s._replay_step, {"low":8})
            self.assertFalse(any(e["event"]=="recomputed" for e in s.logged))
            s.update_from_output(None,None)
            self.assertEqual(s.logged[-1], dict(event="recomputed",request_id="low",tokens=8))
            low.num_computed_tokens = 20
            with patch.object(Base, "schedule", return_value=NS(num_scheduled_tokens={"low":8})):
                s.schedule()
            self.assertEqual(s._replay_step, {"low":4})
            self.assertNotIn("low",s._replay_watermarks)

    def test_config_gates(self):
        config = NS(scheduler_config=NS(policy="priority", async_scheduling=False),
                    parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1,
                                       decode_context_parallel_size=1, prefill_context_parallel_size=1),
                    model_config=NS(runner_type="generate", is_encoder_decoder=False,
                                    is_diffusion=False, is_hybrid=False, is_attention_free=False),
                    max_concurrent_batches=1, kv_transfer_config=None, ec_transfer_config=None,
                    speculative_config=None, lora_config=None)
        check_config(config)
        config.kv_transfer_config = NS()
        with self.assertRaisesRegex(RuntimeError, "kv_transfer"):
            check_config(config)
        config.kv_transfer_config = None
        config.model_config.is_hybrid = True
        state = NS(mamba_type=NS(name="GDN_ATTN"), mamba_cache_mode="none", num_speculative_blocks=0)
        kv = NS(kv_cache_groups=[NS(kv_cache_spec=state), NS(kv_cache_spec=NS())])
        config.cache_config = NS(enable_prefix_caching=False, mamba_cache_mode="none")
        check_config(config, kv)
        config.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(RuntimeError, "hybrid requires"):
            check_config(config, kv)
        config.cache_config.enable_prefix_caching = False
        state.mamba_type.name = "MAMBA2"
        with self.assertRaisesRegex(RuntimeError, "hybrid requires"):
            check_config(config, kv)
        config.model_config.is_hybrid = False
        config.scheduler_config.async_scheduling = True
        with self.assertRaisesRegex(RuntimeError, "async"):
            check_config(config)

    def test_gpu_verdict_requires_correlated_evidence(self):
        from gpu_smoke import verify, read_events
        baseline = dict(request_id="baseline", finish_reason="length", token_ids=[1, 2])
        low = dict(request_id="low-id", finish_reason="length", token_ids=[1, 2], ended=3)
        high = dict(request_id="high-id", finish_reason="length", ended=2)
        with self.assertRaisesRegex(AssertionError, "unique exact-request"):
            verify(baseline, low, high, high, [], "enforce", "recompute")
        event = dict(event="preempted", action="recompute", victim_id="cmpl-low-id-0-1234abcd",
                     requester_id="cmpl-high-id-0-5678cdef", boot_id="boot", terminal=False,
                     private_blocks_released=True)
        resumed = dict(event="resumed", request_id="cmpl-low-id-0-1234abcd", boot_id="boot")
        events = read_events("prefix vvip_event=" + json.dumps(event) + "\nvvip_event=" + json.dumps(resumed))
        verify(baseline, low, high, high, events, "enforce", "recompute")
        with self.assertRaisesRegex(AssertionError, "unique exact-request"):
            verify(baseline, low, high, high, events + [event], "enforce", "recompute")
        low["token_ids"] = [1, 3]
        with self.assertRaisesRegex(AssertionError, "token IDs differ"):
            verify(baseline, low, high, high, events, "enforce", "recompute")

    @unittest.skipUnless(os.environ.get("VVIP_UPSTREAM_SOURCE"), "optional upstream source check")
    def test_actual_upstream_release_methods(self):
        root = Path(os.environ["VVIP_UPSTREAM_SOURCE"])
        self.assertTrue(inspect_source(root)["compatible"])
        tree = ast.parse((root / "v1/core/sched/scheduler.py").read_text())
        base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        names = {"_preempt_request", "finish_requests", "_free_request", "_free_blocks",
                 "_free_request_blocks", "_request_blocks_can_be_freed"}
        nodes = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"RequestStatus": Status, "remove_all": lambda items, removed:
                     [r for r in items if r not in removed]}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "upstream-release", "exec"), namespace)
        with scheduler_class() as cls:
            for action in ("recompute", "abort"):
                s, low, high = harness(cls, action)
                for name in names:
                    setattr(s, name, namespace[name].__get__(s))
                s._maybe_preempt(100, False)
                self.assertNotIn("low", s.blocks)
                self.assertEqual(low.status, Status.PREEMPTED if action == "recompute"
                                 else Status.FINISHED_ABORTED)
                if action == "abort":
                    self.assertEqual(s.update_from_output(None, None)[0].outputs[0].finish_reason, "abort")


if __name__ == "__main__":
    unittest.main()
