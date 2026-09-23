# Compatibility and model evidence

VVIP adapts request state, cache ownership, and the native output protocol. It does not branch on model brand. A new model satisfying the same contract need not get its own scheduler, but **architectural applicability and GPU certification are different claims**.

## Executable contract

| Dimension | Requirement |
| --- | --- |
| Engine | V1, exact vLLM 0.30.0, 35 audited source files match SHA256 |
| Ordinary models | Autoregressive attention decoders; Dense/MoE does not change request selection |
| Hybrid state | Actual attention groups plus native GDN_ATTN groups; extra restrictions below |
| Input | Text in the published tests; encoder-input requests are not victims |
| API | Native completion/chat priority and output protocol |
| Parallelism | DP=PP=DCP=PCP=1; TP>1 requires separate target GPU acceptance |
| Execution | Synchronous, exactly one batch in flight; eager and graph evidence kept separate |
| Cache | Native local ownership; attention-only models can use native prefix caching |
| Actions | Recompute/resume and abort; off/shadow/enforce |
| Rejected | Pure attention-free models, unknown hybrid state, LoRA, speculative decoding, KV/EC connectors, P/D, pooling, diffusion, encoder-decoder |
| Streaming input | Session requests are not selected; output SSE is supported |
| Multiple replicas | Independent per-replica deployment; no global routing or cross-replica fairness |

For a GDN hybrid, every state spec must have `mamba_type.name == GDN_ATTN`, `mamba_cache_mode == none`, and `num_speculative_blocks == 0`. Effective configuration must disable prefix caching. State is rebuilt from token position zero. Checkpoint/CoW paths need separate review.

Quantization, GQA/MQA/MLA, sliding windows, and expert placement remain vLLM's responsibility. They do not automatically gain certification. A model's actual configuration may trigger another gate. Use doctor, startup checks, and target acceptance together. The upstream model catalog is a candidate set, not VVIP's tested-model list.

## Evidence matrix

| Model or family | State or historical deployment | Historical abort evidence | Current independent adapter |
| --- | --- | --- | --- |
| nvidia/Qwen3.8-27B-NVFP4 | 48 GDN + 16 attention layers; NVFP4/FP8 | Not inherited from another checkpoint | Both actions, off/shadow, four negative controls, 2,288-position replay, and 16-card single-GPU rotation complete; concurrent outputs are not guaranteed identical |
| Qwen3.8 NVFP4, historical checkpoint | Single engine | 6/6 | Requires its own checkpoint-specific acceptance |
| Kimi K3 | Historical single engine | 6/6 | KDA/other hybrid state is outside the current GDN gate; separate adaptation and tests required |
| GLM-5.3 NVFP4 | Historical single engine | 6/6 | Requires matching weights/version acceptance |
| GLM-5.3 NVFP4, P/D | Historical two-stage release | 6/6 | This adapter rejects P/D; historical release capability is not inherited |
| Llama, Mistral/Mixtral, Gemma, other Qwen Dense/MoE | Candidate attention decoders; actual configuration decides | Not audited here | Contract may apply; no published per-model GPU certification in this repository |
| Other native GDN hybrids | Must satisfy the real state-spec and no-prefix-state gates | Not audited here | Generic gate exists; each target configuration still needs tests |
| DeepSeek-v4.1-flash, MiniMax-M3 | Outside the historical success set | Not counted as passed | Uncertified |
| Unknown SSM, pure attention-free, externally owned state | Outside the current contract | Not applicable | Rejected pending a state-recovery design |

The 24 historical cases were checked for successful requesters, preempted victims, native running/KV release, and stage identifiers. The sanitized repository record is `docs/evidence/historical-abort-summary.json`. It proves bounded abort cases on the historical implementation, not this code's recompute behavior or long-term stability. See [validation](validation.md) and [design](design.md).

## Certifying another model

1. Pin revision, weight hashes, quantization, template, vLLM image/source, hardware, and effective configuration.
2. Prove native vLLM loads and completes requests first. Preserve native failures.
3. Run doctor and startup gates. Treat TP, graph execution, and runner changes as separate configurations.
4. Run off/shadow/recompute/abort. Require exact internal-ID correlation, native release evidence, and one client terminal.
5. Compare resumed tokens under fixed sampling; test long history, concurrent requests, post-abort reuse, and spare-capacity controls.
6. Repeat matched workloads, reporting ordinary-request costs with VIP latency. Retain failed and divergent results.

Finite tests cannot prove every existing or future model/configuration combination. A supported-model claim must name the actual evidence and its limits.

## Version upgrades

The [source profile](../scripts/vvip_runtime/compatibility.json) covers scheduler/request/queue, engine/output, runner, KV ownership, configuration, completion serving, GDN initialization/update, and quantization paths. Version text alone is insufficient. Drift is rejected even in shadow mode.

This is compatibility checking, not complete supply-chain verification. Unhashed dependencies, compiled kernels, hardware, and runtime file changes need an immutable deployment environment.

Review each new version in this order:

1. Priority/internal-ID propagation and blocked queue states.
2. `_preempt_request`, `finish_requests`, deferred free, private/shared ownership.
3. Scheduler output and active runner; same-step restoration of batch/token state.
4. Engine output through the API; exactly one abort terminal in normal process operation.
5. Extra model state reset/replay and effective quantization/configuration behavior.
6. CPU checks, target GPU controls, and repeated workloads; update evidence with the implementation.

There is no force option. If native vLLM gains the same semantics, run the same acceptance suite and prefer the native capability when it satisfies the contract.

Sources: [Scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py), [cache state definitions](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/kv_cache_interface.py), [target model](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4).
