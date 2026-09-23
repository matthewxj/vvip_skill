---
name: vvip
description: Configure, inspect, and verify priority preemption of running requests on self-hosted vLLM. Supports recompute-and-resume or permanent abort through an engine-local scheduler. Use for urgent LLM requests blocked behind running lower-priority work; requires control of the vLLM server, not just an API key to a hosted model.
---

# VVIP

Resolve script paths relative to this skill directory. All scripts, runtime, and
references are self-contained. No Redis, Kubernetes, MCP server, or separate
gateway is required.

## Choose the operation

- Inspect/plan: `python3 scripts/vvip.py doctor`. Missing vLLM is a useful result;
  it does not require installing CUDA packages on the agent's machine.
- `recompute`: pause at an iteration boundary, release KV references, keep
  generated tokens, then let vLLM recompute and resume the same request.
- `abort`: permanently terminate with native `finish_reason="abort"` and
  `stop_reason="vvip_preempted"`. Never silently retry it.
- `--mode shadow` reports decisions without VVIP mutation; `off` disables VVIP.
  vLLM's own memory-pressure preemption is independent.

Default to recompute if unspecified; honor explicit abort or both. Before
configuring a server read [operation.md](references/operation.md), then
[compatibility.md](references/compatibility.md) for supported combinations.
Read [design.md](references/design.md) when extending the runtime or integrating
external admission/lease accounting.

## Workflow

1. Establish the actual server's vLLM build, model revision, hardware, parallelism,
   connectors, and whether it carries shared/live traffic. Inspect that server's
   environment, not just the agent's Python. A skill cannot change a third-party
   hosted API's GPU scheduling or Codex's own hosted inference.
2. Run doctor there. Matching source is **not GPU certification**. Never invent a
   compatible version, bypass a failing hash, or ignore unsupported topology.
3. Print a concrete plan: `python3 scripts/vvip.py serve MODEL --mode shadow
   --action recompute -- VLLM_ARGS`. It does not launch. Preserve the operator's
   model, tokenizer, template, quantization, and tool parsers.
4. For an authorized isolated test, add `--execute` **before** `--`. The launcher
   binds loopback and does not install packages. vLLM itself may download the
   model. Use an isolated environment; don't upgrade production in place.
5. Follow GPU acceptance in operation.md. Test both actions when requested. Require
   exact requester/victim correlation, resumed token equivalence for recompute,
   explicit abort terminal for abort, and a successful subsequent request.
   HTTP close, lower aggregate utilization, or CPU tests alone prove none of these.
   For performance use matched native-priority baseline and benchmark.py; report
   ordinary completion/throughput costs beside VIP latency. One smoke is not a
   performance result. gpu_guards.py checks equal/reverse priorities, a configured
   disable file, and client disconnect on an isolated single-sequence server.
6. Report implementation, source-check, CPU-check, and GPU-check status separately.
   Use existing session authorization; prepare concrete deployment changes before
   requesting any missing approval. Use the disable-file switch for an authorized
   live disable; it does not restore already aborted requests.

## Non-obvious constraints

- Smaller `priority` is more urgent; native default is `0`. Normal `0`, urgent
  `-10` is one convention. Same-priority victims are never selected.
- Priority is not authorization. Before sharing an endpoint, trusted authenticated
  ingress must assign/validate JSON `priority` **and** `X-Vllm-Priority`; vLLM gives
  the header precedence. An ordinary API key must not permit self-promotion.
- The scheduler owns queues and allocation. Local events are diagnostic evidence,
  **not signed/durable remote release receipts**; never clear external leases from
  these logs.
- Abort is terminal even when HTTP is 200. Consumers must inspect finish/stop
  reasons and cannot treat partial output as successful completion.
- Current adapter: experimental, source-pinned to vLLM `0.30.0` (35 files).
  Qwen3.8-27B-NVFP4 passed both actions, off/shadow, negative controls, and
  long-history replay on one RTX 6000D with V2/eager. See
  [validation.md](references/validation.md) for exact identities and limits.
  GPU acceptance is target-specific; broad applicability is not a tested model list.
- Hybrid models require actual native GDN_ATTN plus attention cache groups,
  prefix caching disabled, and mamba cache mode none. Never infer support from
  the model name or copy historical distributed release proofs to this adapter.
