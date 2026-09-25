# GPU validation — 2026-09-23

## Scope

**nvidia/Qwen3.8-27B-NVFP4 passed native loading, both actions, off/shadow, four runtime negative controls, and mid-generation replay.** These results bind the frozen runtime to a single RTX 6000D, vLLM 0.30.0, V2 runner, text input, and eager execution. They cover GDN hybrid state and mixed NVFP4/FP8 quantization.

The repeated performance study uses the same runtime across 16 cards. The one-sample smoke latencies below are correctness observations, not estimates of population-level performance.

The disconnect negative control below establishes HTTP reuse after closing a
stream, not exact native cancellation or private-KV release. The September 25
[client hardening](reliability.md) makes that limitation explicit and adds local
failure regressions. The scheduler/source-profile hash is unchanged; those new
client paths have not been rerun on GPU. Recorded September 23 measurements remain
historical evidence and have not been rewritten.

## Repeated workload summary

Four variants—native priority+sync, native priority+async, recompute, abort—rotate across all 16 cards, producing 256 reports and 1,664 measured requests. Each scenario/variant has 16 matched trials, concurrency four, and the default 100 ms wait threshold.

| Scenario | VIP TTFT p50: native sync → recompute | Completed-request throughput change |
| --- | --- | ---: |
| Saturated | 31.905 → 0.233 s | −0.54% |
| Burst | 31.706 → 0.248 s | −0.78% |
| Long prefill | 32.452 → 0.414 s | −1.41% |

Within these contended samples, VIP one-second target attainment changed from 0% to 100%. All 272 ordinary requests in the recompute arm completed. Abort had 50% ordinary completion in each contended scenario. No added benefit was established in the spare-capacity control.

Fixed synthetic workloads, a shared host, early VIP arrivals, and eager/concurrency-four execution limit extrapolation. Recompute output matched native sync for 223/416 complete sequences. Same-process off/off repetition also matched only 4/6. Single-sequence exact replay does not establish concurrent numerical or answer-quality equivalence. All differences are retained.

The full project includes the protocol, paired intervals, costs, and public dataset at `docs/EXPERIMENTS.md` and `docs/evidence/qwen38`. These limitations also apply when only the skill folder is copied.

## Reproducible identity

| Item | Value |
| --- | --- |
| Model | `nvidia/Qwen3.8-27B-NVFP4` |
| Revision | `482ca0f3832238542f8f5295dde86b5f22711d80` |
| Architecture | `Qwen3_5ForConditionalGeneration`, 48 GDN + 16 attention layers |
| Quantization / base dtype / KV dtype | `modelopt_mixed`, NVFP4/FP8; bfloat16; auto |
| vLLM / Torch / CUDA | `0.30.0` / `2.13.0+cu130` / `13.0` |
| GPU / driver | NVIDIA RTX 6000D, 85,651 MiB / 580.178.04 |
| Runner / GDN | V2; FlashInfer prefill, cuda decode |
| GEMM | FlashInferCutlassNvFp4LinearKernel / FlashInferFP8ScaledMMLinearKernel |
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.30.0` |
| Image digest | `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` |
| VVIP runtime SHA256 | `1422c1e1e030cdf3869c6714020e4c722b38d59afbe27b16fb9ad8b731875559` |
| Source profile | All 35 files matched; no bypass |

All 19 model files were checked against official LFS SHA256 / Git blob SHA1 and then recorded in a full SHA256 manifest. A second verifier repeated the check. The three weight shards are:

```text
model-00001-of-00003.safetensors  9965652544 bytes
7d0fd155118901373eb0fd13ed3aae68f95be747bc205be72ebc41739c25ee80
model-00002-of-00003.safetensors  9985757064 bytes
98a7e9486baa860c792c9463a770cb9d017696bdd17606300a5b9334149f4c27
model-00003-of-00003.safetensors  1970287672 bytes
0506ad35dc21469708e7813bd76c592cef08bd0317b4a1fe899745ebe2435271
```

The public download used no credentials. Weights are not distributed with the skill.

## Correctness configuration

Each case ran in a new container with no external network or host port mapping, read-only model/skill mounts, and an in-container loopback client. TP/DP/PP/DCP/PCP=1; no connectors, LoRA, or speculative decoding.

```text
--served-model-name qwen38-vvip
--max-model-len 8192
--max-num-seqs 1
--max-num-batched-tokens 1024
--gpu-memory-utilization 0.60
--enforce-eager
--no-enable-prefix-caching
--mamba-cache-mode none
--language-model-only
--limit-mm-per-prompt {"image":0,"video":0}
```

VVIP used priority+synchronous, `VVIP_TRIGGER_WAIT_MS=0`, and default remaining protections. Sampling used seed 42, temperature 0, `ignore_eos=true`, and `return_token_ids=true`. Each smoke ran a 512-token baseline, an ordinary request interrupted after output began, a 16-token VIP, and an 8-token reuse probe.

The native probe completed 16 tokens with one terminal. All VVIP clients required exactly one terminal and `[DONE]`; length-completed requests required the exact output-token count. Actual internal IDs, boot, and runtime hash were correlated. HTTP success or GPU utilization alone was not release evidence.

## Four modes

| Mode | Ordinary request | Engine evidence | VIP first token, one sample |
| --- | --- | --- | ---: |
| Recompute | 512 tokens, identical to baseline | One preemption; private references released; same-ID/boot resume; 13 replayed positions | 136.846 ms |
| Abort | Two tokens, then explicit abort | One preemption; release; same-ID/boot abort delivery | 136.874 ms |
| Off | 512 tokens, identical | No VVIP decisions/preemptions | 31,655.448 ms |
| Shadow | 512 tokens, identical | 60 shadow decisions, no actual preemption | 31,798.450 ms |

Every VIP and follow-up completed 16 and 8 tokens respectively. Baseline and recompute/off/shadow ordinary output hashes were:
`230ff0f0c3bb95ce814b9f63a3c52cc3e5cd5fb0d1a7240ae8e0456738d909de`.

The single running slot deliberately forces contention. The repeated study uses higher concurrency, the default 100 ms threshold, and both native sync/async controls.

## Longer history and mid-generation replay

A synthetic 64-row inventory prompt tokenized to **2,160 input tokens**. After a 512-token baseline, the VIP was submitted once a matching request produced at least 128 output tokens. Actual preemption occurred at 129 output tokens and computed position 2,288. The most recently sampled token had not yet been computed by the following step.

Native replay covered **1,024 + 1,024 + 240 = 2,288 positions**. The original request resumed, all 512 output IDs matched baseline, the VIP finished first, and the follow-up succeeded. No replacement ordinary request was submitted. This covers chunked replay and GDN/attention reconstruction for one fixed synthetic sample.

## Runtime negative controls

All four passed:

- Equal priorities: no preemption; both requests completed.
- Reverse priority: a less urgent arrival did not preempt more urgent running work.
- Configured disable file present: no new VVIP preemption; the test removed only its own file.
- Client disconnect after the first SSE token: a follow-up completed without extra VVIP preemption.

These do not prove every cancellation race or long-term freedom from leaks.

## Warnings and evidence

Logs retain FP4 lm_head shape 1/2 tuning-bucket fallback to Cutlass tactic −1 and shutdown EngineCore/resource-tracker semaphore warnings. No backend was changed to improve the numbers. This eager configuration is not a claim of peak model throughput. Containers stopped and independent snapshots confirmed GPU-memory release.

The six acceptance cases have 133 verified raw files; long-history replay adds 16. Public JSON retains synthetic IDs, timing, counts, model/runtime identity, and hashes. Private server/container snapshots are excluded from distribution.

Acceptance client SHA256: `ee68c78563556f844f8fda031b01ef4af14763598e4ca364a185a779b121cb85`.
The later prompt/trigger-capable client SHA256 is `4d9cf9a4a1cece4633d118dbe10efca59e85aa9be7543e01553f428fd3414c09`.
Both used the same frozen runtime. Doctor still reports `gpu_validated=false`: it checks source, while this document records configuration-specific GPU evidence.

CUDA graphs, TP>1, other weights/backends, actual tool calls/structured output, sustained production workloads, all disconnect races, and multi-host topology are not certified by these results. Unsupported configurations must follow the [contract review](compatibility.md).
