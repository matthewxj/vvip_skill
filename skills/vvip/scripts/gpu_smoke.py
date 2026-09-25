#!/usr/bin/env python3
"""Bounded real HTTP acceptance test, against a dedicated max-num-seqs=1 server.

Uses stdlib only. Requires fresh exact-request engine events in --server-log.
Never treats HTTP completion or aggregate metrics as a release proof.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid

from completion_client import completion_events


def stream(base, model, request_id, priority, tokens, timeout, first=None,
           prompt=None, first_after_tokens=1, *, ttft_timeout=None, idle_timeout=None):
    body = {"model": model, "prompt": prompt if prompt is not None else "List consecutive integers starting at 1, separated by commas:",
            "request_id": request_id, "priority": priority, "max_tokens": tokens,
            "stream": True, "temperature": 0, "seed": 42, "ignore_eos": True,
            "return_token_ids": True}
    result = {"request_id": request_id, "started": time.monotonic(), "token_ids": []}
    done, terminals = False, 0
    with completion_events(base, body, timeout, ttft_timeout=ttft_timeout,
                           idle_timeout=idle_timeout) as chunks:
        for chunk in chunks:
            if chunk is None:
                done = True
                break
            for choice in chunk.get("choices", []):
                result["token_ids"].extend(choice.get("token_ids") or [])
                if choice.get("text") or choice.get("token_ids"):
                    result.setdefault("first_token", time.monotonic())
                    if first and len(result["token_ids"]) >= first_after_tokens:
                        first.set()
                if choice.get("finish_reason"):
                    terminals += 1
                    result["finish_reason"] = choice["finish_reason"]
                    result["stop_reason"] = choice.get("stop_reason")
    result["ended"] = time.monotonic()
    result["terminal_count"] = terminals
    if not done or terminals != 1:
        raise RuntimeError("stream must contain exactly one terminal choice and [DONE]")
    if result["finish_reason"] == "length" and len(result["token_ids"]) != tokens:
        raise RuntimeError("stream did not contain the requested number of token IDs")
    if (result["finish_reason"] not in {"length", "abort"}
            or (result["finish_reason"] == "abort" and result.get("stop_reason") != "vvip_preempted")):
        raise RuntimeError("unexpected completion terminal")
    return result


def read_events(text):
    events = []
    for line in text.splitlines():
        if "vvip_event=" in line:
            # Logs may have an ANSI reset after the JSON.
            event, _ = json.JSONDecoder().raw_decode(line.split("vvip_event=", 1)[1])
            events.append(event)
    return events


def read_run_events(path, snapshot, run_id, boot_id=None):
    """Read only this run; refuse rotated/truncated logs and engine restarts."""
    import os
    with path.open("rb") as log:
        current = os.fstat(log.fileno())
        if ((current.st_dev, current.st_ino) != (snapshot.st_dev, snapshot.st_ino)
                or current.st_size < snapshot.st_size):
            raise RuntimeError("server log was rotated or truncated during the run")
        log.seek(snapshot.st_size)
        events = read_events(log.read().decode(errors="replace"))
    if any(e.get("event") == "ready" or (boot_id and e.get("boot_id") != boot_id)
           for e in events):
        raise RuntimeError("engine identity changed during the run")
    return [e for e in events if run_id in json.dumps(e)]


def verify(baseline, low, high, followup, events, mode, action):
    # Audited completion path adds -0 (one prompt), then InputProcessor adds an
    # 8-hex unique suffix unless native request-ID randomization is disabled.
    # Capture the sole actual ID; never synthesize an ID to send a cancellation.
    def matches(internal, external):
        return re.fullmatch(re.escape(f"cmpl-{external}-0") + r"(?:-[0-9a-f]{8})?", internal or "")
    matched = [e for e in events if matches(e.get("victim_id"), low["request_id"])
               and matches(e.get("requester_id"), high["request_id"])]
    related = [e for e in events if any(
        matches(e.get(key), request["request_id"])
        for key in ("victim_id", "requester_id", "request_id")
        for request in (low, high))]
    identities = {(e.get("boot_id"), e.get("runtime_sha256")) for e in related}
    if related and (len(identities) != 1 or any(not all(identity) for identity in identities)):
        raise AssertionError("missing or mixed engine/runtime identity")
    actual = [e for e in matched if e["event"] == "preempted"]
    if high["finish_reason"] != "length" or followup["finish_reason"] != "length":
        raise AssertionError("urgent/follow-up request did not complete normally")
    if mode == "enforce":
        if len(actual) != 1 or actual[0]["action"] != action:
            raise AssertionError("no unique exact-request preemption; check server flags/load")
        if actual[0].get("private_blocks_released") is not True:
            raise AssertionError("request-private block release was not observed")
        event = actual[0]
        if action == "abort":
            if low["finish_reason"] != "abort" or low.get("stop_reason") != "vvip_preempted":
                raise AssertionError("victim did not receive the VVIP abort terminal")
            if event.get("terminal") is not True or sum(
                e["event"] == "abort_delivered" and e.get("request_id") == event["victim_id"]
                and e.get("boot_id") == event["boot_id"] for e in events
            ) != 1:
                raise AssertionError("abort delivery was not observed on the same engine")
            return
        if event.get("terminal") is not False or high["ended"] >= low["ended"]:
            raise AssertionError("urgent request did not finish before resumed victim")
        if sum(e["event"] == "resumed" and e.get("request_id") == event["victim_id"]
               and e.get("boot_id") == event["boot_id"] for e in events) != 1:
            raise AssertionError("victim was not observed resuming")
    else:
        if any(e["event"] == "preempted" for e in related):
            raise AssertionError("off/shadow must not preempt")
        if mode == "shadow" and not any(e["event"] == "would_preempt" for e in matched):
            raise AssertionError("no exact shadow decision was observed")
    if low["finish_reason"] != "length" or not low["token_ids"]:
        raise AssertionError("victim did not complete with token IDs")
    if low["token_ids"] != baseline["token_ids"]:
        raise AssertionError("resumed/control token IDs differ; investigate model determinism")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--action", required=True, choices=["recompute", "abort"])
    parser.add_argument("--mode", default="enforce", choices=["off", "shadow", "enforce"])
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--ttft-timeout", type=float, help="headers through first real output; defaults to timeout")
    parser.add_argument("--idle-timeout", type=float, help="gap between real output chunks; defaults to timeout")
    parser.add_argument("--prompt-file", type=Path, help="optional synthetic UTF-8 prompt; report retains only its hash")
    parser.add_argument("--preempt-after-tokens", type=int, default=1)
    args = parser.parse_args()
    if not 32 <= args.tokens <= 8192 or not 0 < args.timeout <= 600:
        parser.error("tokens must be 32..8192 and timeout 0..600")
    if not 1 <= args.preempt_after_tokens < args.tokens:
        parser.error("preempt-after-tokens must be between 1 and tokens-1")
    import math
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.ttft_timeout, args.idle_timeout)):
        parser.error("phase timeouts must be positive finite seconds")
    if args.out.exists():
        parser.error("output already exists")
    prompt = args.prompt_file.read_text() if args.prompt_file else None
    if prompt is not None and not prompt.strip():
        parser.error("prompt file must not be empty")
    snapshot = args.server_log.stat()
    run_id = "vvip-smoke-" + uuid.uuid4().hex
    records, events, error, evidence_error, ready = {}, [], None, None, []
    phase_timeouts = dict(ttft_timeout=args.ttft_timeout, idle_timeout=args.idle_timeout)
    try:
        ready = [e for e in read_events(args.server_log.read_text()) if e.get("event") == "ready"]
        if (len(ready) != 1 or not ready[0].get("boot_id") or not ready[0].get("runtime_sha256")
                or ready[0].get("settings", {}).get("mode") != args.mode
                or ready[0].get("settings", {}).get("action") != args.action):
            raise RuntimeError("fresh matching engine ready event required")
        def capture(name, priority, tokens, **kwargs):
            try:
                result = stream(args.base_url, args.model, run_id + "-" + name,
                                priority, tokens, args.timeout, **phase_timeouts, **kwargs)
            except Exception as exc:
                records[name] = {"request_id": run_id + "-" + name, "error": type(exc).__name__}
                raise
            records[name] = result
            return result

        capture("baseline", 0, args.tokens, prompt=prompt)
        first = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            low_future = executor.submit(capture, "low", 0, args.tokens, first=first,
                                         prompt=prompt, first_after_tokens=args.preempt_after_tokens)
            # Wake promptly on failure instead of waiting a whole token budget.
            low_future.add_done_callback(lambda _: first.set())
            if not first.wait(args.timeout):
                raise TimeoutError("victim never reached the requested token threshold")
            if low_future.done():
                low_future.result()
                raise RuntimeError("victim completed before the VIP was submitted")
            capture("high", -10, 16)
            low_future.result(timeout=args.timeout)
        capture("followup", 0, 8)
    except Exception as exc:
        error = type(exc).__name__  # Do not publish server bodies, URLs, or credentials.
    try:
        events = read_run_events(args.server_log, snapshot, run_id,
                                 ready[0].get("boot_id") if len(ready) == 1 else None)
        if error is None:
            verify(records["baseline"], records["low"], records["high"], records["followup"],
                   events, args.mode, args.action)
    except (AssertionError, RuntimeError, ValueError, OSError) as exc:
        evidence_error = str(exc) if isinstance(exc, AssertionError) else type(exc).__name__
    verdict = error is None and evidence_error is None
    for result in records.values():
        if "token_ids" not in result:
            continue
        token_ids = result.pop("token_ids")
        result.update(token_count=len(token_ids), token_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest())
    report = {"schema": "vvip.smoke/v1", "passed": verdict, "error": error,
              "engine_identity": {key: ready[0].get(key) for key in ("boot_id", "runtime_sha256")}
              if len(ready) == 1 else None,
              "evidence_error": evidence_error, "timeouts": {"total": args.timeout, **phase_timeouts},
              "run_id": run_id, "mode": args.mode, "action": args.action,
              "preempt_after_tokens": args.preempt_after_tokens,
              "custom_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None,
              "records": records, "events": [e for e in events if run_id in json.dumps(e)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({"passed": verdict, "error": error, "evidence_error": evidence_error,
                      "report": str(args.out.resolve())}))
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
