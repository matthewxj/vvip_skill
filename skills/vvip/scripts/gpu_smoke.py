#!/usr/bin/env python3
"""Bounded real HTTP acceptance test, against a dedicated max-num-seqs=1 server.

Uses stdlib only. Requires fresh exact-request engine events in --server-log.
Never treats HTTP completion or aggregate metrics as a release proof.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid


def stream(base, model, request_id, priority, tokens, timeout, first=None,
           prompt=None, first_after_tokens=1):
    body = {"model": model, "prompt": prompt if prompt is not None else "List consecutive integers starting at 1, separated by commas:",
            "request_id": request_id, "priority": priority, "max_tokens": tokens,
            "stream": True, "temperature": 0, "seed": 42, "ignore_eos": True,
            "return_token_ids": True}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("VVIP_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + "/v1/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    result = {"request_id": request_id, "started": time.monotonic(), "token_ids": []}
    done, terminals = False, 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if time.monotonic() - result["started"] > timeout:
                raise TimeoutError("generation exceeded test deadline")
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"SSE error: {chunk['error']}")
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
    return result


def read_events(text):
    events = []
    for line in text.splitlines():
        if "vvip_event=" in line:
            # Logs may have an ANSI reset after the JSON.
            event, _ = json.JSONDecoder().raw_decode(line.split("vvip_event=", 1)[1])
            events.append(event)
    return events


def verify(baseline, low, high, followup, events, mode, action):
    # Audited completion path adds -0 (one prompt), then InputProcessor adds an
    # 8-hex unique suffix unless native request-ID randomization is disabled.
    # Capture the sole actual ID; never synthesize an ID to send a cancellation.
    def matches(internal, external):
        return re.fullmatch(re.escape(f"cmpl-{external}-0") + r"(?:-[0-9a-f]{8})?", internal or "")
    matched = [e for e in events if matches(e.get("victim_id"), low["request_id"])
               and matches(e.get("requester_id"), high["request_id"])]
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
            if event.get("terminal") is not True or not any(
                e["event"] == "abort_delivered" and e.get("request_id") == event["victim_id"]
                and e.get("boot_id") == event["boot_id"] for e in events
            ):
                raise AssertionError("abort delivery was not observed on the same engine")
            return
        if event.get("terminal") is not False or high["ended"] >= low["ended"]:
            raise AssertionError("urgent request did not finish before resumed victim")
        if not any(e["event"] == "resumed" and e.get("request_id") == event["victim_id"]
                   and e.get("boot_id") == event["boot_id"] for e in events):
            raise AssertionError("victim was not observed resuming")
    else:
        if actual:
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
    parser.add_argument("--prompt-file", type=Path, help="optional synthetic UTF-8 prompt; report retains only its hash")
    parser.add_argument("--preempt-after-tokens", type=int, default=1)
    args = parser.parse_args()
    if not 32 <= args.tokens <= 8192 or not 0 < args.timeout <= 600:
        parser.error("tokens must be 32..8192 and timeout 0..600")
    if not 1 <= args.preempt_after_tokens < args.tokens:
        parser.error("preempt-after-tokens must be between 1 and tokens-1")
    prompt = args.prompt_file.read_text() if args.prompt_file else None
    if prompt is not None and not prompt.strip():
        parser.error("prompt file must not be empty")
    offset = args.server_log.stat().st_size
    run_id = "vvip-smoke-" + uuid.uuid4().hex
    baseline = stream(args.base_url, args.model, run_id + "-baseline", 0, args.tokens, args.timeout, prompt=prompt)
    first = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        low_future = executor.submit(stream, args.base_url, args.model, run_id + "-low", 0,
                                     args.tokens, args.timeout, first, prompt, args.preempt_after_tokens)
        if not first.wait(args.timeout):
            raise TimeoutError("victim never reached the requested token threshold")
        high = stream(args.base_url, args.model, run_id + "-high", -10, 16, args.timeout)
        low = low_future.result(timeout=args.timeout)
    followup = stream(args.base_url, args.model, run_id + "-followup", 0, 8, args.timeout)
    with args.server_log.open("rb") as log:
        log.seek(offset)
        events = read_events(log.read().decode(errors="replace"))
    verdict, error = True, None
    try:
        verify(baseline, low, high, followup, events, args.mode, args.action)
    except AssertionError as exc:
        verdict, error = False, str(exc)
    records = {}
    for name, result in (("baseline", baseline), ("low", low), ("high", high), ("followup", followup)):
        token_ids = result.pop("token_ids")
        records[name] = {**result, "token_count": len(token_ids),
                         "token_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()}
    report = {"schema": "vvip.smoke/v1", "passed": verdict, "error": error,
              "run_id": run_id, "mode": args.mode, "action": args.action,
              "preempt_after_tokens": args.preempt_after_tokens,
              "custom_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None,
              "records": records, "events": [e for e in events if run_id in json.dumps(e)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({"passed": verdict, "error": error, "report": str(args.out.resolve())}))
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
