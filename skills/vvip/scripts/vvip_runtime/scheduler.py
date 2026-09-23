"""Priority preemption at a synchronous vLLM iteration boundary.

There is one authority: the engine scheduler. No external lease is granted or
released. Never import this module for CPU-only inspection.
"""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from .compat import check_config, require_compatible
from .policy import Settings, choose_victim

logger = logging.getLogger(__name__)


class VVIPScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        require_compatible()
        self.vvip = Settings.from_env()
        super().__init__(*args, **kwargs)
        check_config(self.vllm_config, self.kv_cache_manager.kv_cache_config)
        self._boot = uuid.uuid4().hex
        package = Path(__file__).parent
        self._runtime_sha256 = hashlib.sha256(b"".join(
            path.name.encode() + b"\0" + path.read_bytes()
            for path in sorted(package.iterdir()) if path.suffix in {".py", ".json"}
        )).hexdigest()
        self._queued = {}
        self._first_running = {}
        self._counts = {}
        self._events = deque()
        self._aborted = []
        self._resuming = set()
        self._replay_watermarks = {}
        self._replay_step = {}
        self._event("ready", settings=asdict(self.vvip))

    def _event(self, event, **fields):
        logger.warning("vvip_event=%s", json.dumps({
            "schema": "vvip.event/v1", "event": event, "boot_id": self._boot,
            "runtime_sha256": self._runtime_sha256,
            "monotonic": time.monotonic(), **fields,
        }, sort_keys=True))

    def add_request(self, request):
        existing = request.request_id in self.requests
        super().add_request(request)
        if not existing:
            self._queued[request.request_id] = time.monotonic()
            self._first_running.pop(request.request_id, None)
            self._counts.pop(request.request_id, None)
            self._resuming.discard(request.request_id)
            self._replay_watermarks.pop(request.request_id, None)

    def _blocks_released(self, request):
        managers = self.kv_cache_manager.coordinator.single_type_managers
        return bool(managers) and all(
            request.request_id not in manager.req_to_blocks for manager in managers
        ) and not self.deferred_frees

    def _maybe_preempt(self, now, throttle_prefills):
        policy = self.vvip
        if policy.disable_file:
            try:
                os.stat(policy.disable_file)
            except FileNotFoundError:
                pass
            except OSError:
                return  # An unreadable switch must not enable preemption.
            else:
                return
        if (policy.mode == "off" or not (self.waiting or self.skipped_waiting) or not self.running
                or self.pause_state != PauseState.UNPAUSED or throttle_prefills):
            return
        while self._events and now - self._events[0] >= 60:
            self._events.popleft()
        if (len(self._events) >= policy.max_per_minute or
                (self._events and (now - self._events[-1]) * 1000 < policy.min_interval_ms)):
            return
        request_queue = self._select_waiting_queue_for_scheduling()
        requester = request_queue.peek_request()
        ready = requester.status in {RequestStatus.WAITING, RequestStatus.PREEMPTED}
        if requester.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured = requester.structured_output_request
            grammar = structured.grammar if structured else None
            ready = grammar is not None and not isinstance(grammar, Exception)
        # Let super().schedule() perform the native blocked->ready transition.
        if (not ready
                or requester.resumable
                or requester.has_encoder_inputs
                or (now - self._queued[requester.request_id]) * 1000 < policy.trigger_wait_ms):
            return
        # Sequence pressure and decode/prefill token-budget pressure. KV-only
        # pressure remains vLLM's allocator's decision, not a guessed free count.
        chunk = self.scheduler_config.long_prefill_token_threshold or self.max_num_scheduled_tokens
        demand = sum(min(chunk, max(0, req.num_tokens_with_spec + req.num_output_placeholders
                                   - req.num_computed_tokens)) for req in self.running)
        if len(self.running) < self.max_num_running_reqs and demand < self.max_num_scheduled_tokens:
            return
        victim = choose_victim(requester, self.running, self._counts,
                               self._first_running, now, policy)
        if victim is None:
            return
        fields = dict(requester_id=requester.request_id, victim_id=victim.request_id,
                      requester_priority=requester.priority, victim_priority=victim.priority,
                      action=policy.action, generated_tokens=victim.num_output_tokens)
        fields["computed_tokens_before"] = victim.num_computed_tokens
        self._events.append(now)
        if policy.mode == "shadow":
            self._event("would_preempt", **fields)
            return
        if policy.action == "recompute":
            self._replay_watermarks[victim.request_id] = max(
                victim.num_computed_tokens,
                self._replay_watermarks.get(victim.request_id, 0))
            self.running.remove(victim)
            self._preempt_request(victim, now)
            # If resumed in this same iteration MRV1 must receive all token IDs.
            self.prev_step_scheduled_req_ids.discard(victim.request_id)
            self._resuming.add(victim.request_id)
        else:
            finished = self.finish_requests(victim.request_id, RequestStatus.FINISHED_ABORTED)
            if finished != [victim]:
                raise RuntimeError("VVIP victim changed before abort")
            self._aborted.append(victim)
        if (victim in self.running or not self._blocks_released(victim)
                or (policy.action == "abort" and victim.request_id in self.requests)):
            raise RuntimeError("VVIP native release invariant failed")
        self._counts[victim.request_id] = self._counts.get(victim.request_id, 0) + 1
        self._event("preempted", **fields, private_blocks_released=True,
                    terminal=policy.action == "abort")

    def schedule(self, throttle_prefills=False):
        now = time.monotonic()
        for request in self.running:
            self._first_running.setdefault(request.request_id, now)
        self._maybe_preempt(now, throttle_prefills)
        result = super().schedule(throttle_prefills)
        # Native schedule has advanced the computed cursor by the assigned work.
        # Count only re-evaluated positions, excluding prefix hits and new tokens.
        self._replay_step = {}
        for request_id in self._replay_watermarks.keys() & result.num_scheduled_tokens.keys():
            end = self.requests[request_id].num_computed_tokens
            start = end - result.num_scheduled_tokens[request_id]
            watermark = self._replay_watermarks[request_id]
            replay = max(0, min(end, watermark) - max(0, start))
            if replay:
                self._replay_step[request_id] = replay
            if end >= watermark:
                del self._replay_watermarks[request_id]
        for request in self.running:
            self._first_running.setdefault(request.request_id, now)
        for request_id in self._resuming & result.num_scheduled_tokens.keys():
            self._event("resumed", request_id=request_id)
            self._resuming.remove(request_id)
        return result

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        for request_id, tokens in self._replay_step.items():
            self._event("recomputed", request_id=request_id, tokens=tokens)
        self._replay_step.clear()
        # finish_requests alone frees scheduler state but does NOT send a final
        # output to the API consumer. Emit it once, after this worker step returns.
        for request in self._aborted:
            output = result.setdefault(request.client_index, EngineCoreOutputs())
            output.outputs.append(EngineCoreOutput(
                request_id=request.request_id, new_token_ids=[],
                finish_reason=FinishReason.ABORT, stop_reason="vvip_preempted",
                events=request.take_events(), trace_headers=request.trace_headers,
            ))
            output.finished_requests = (output.finished_requests or set()) | {request.request_id}
            self._event("abort_delivered", request_id=request.request_id)
        self._aborted.clear()
        # ponytail: O(live requests) cleanup per step; use finish callbacks if
        # profiling shows metadata scanning matters at very high concurrency.
        live = self.requests.keys()
        for mapping in (self._queued, self._first_running, self._counts, self._replay_watermarks):
            for request_id in mapping.keys() - live:
                del mapping[request_id]
        self._resuming.intersection_update(live)
        return result
