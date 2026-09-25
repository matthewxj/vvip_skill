# Quick start and operations

Run these commands from the complete `skills/vvip` directory. Planning and HTTP clients need Python 3.10+; inference needs a Linux GPU host, hardware compatible with the checkpoint, and **vLLM 0.30.0**. Read [compatibility](compatibility.md) before deploying. Installing the agent skill does not install CUDA or vLLM.

## 1. Inspect and plan

Copy the complete skill to the target host. You do not need to install its optional Python package. For a new engine environment, follow the [vLLM GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) and pin `0.30.0`. Custom builds with the same version string can still differ.

```sh
cd /path/to/vvip
python3 scripts/vvip.py doctor

# Inspect an exact source copy without importing torch or connecting to a GPU.
python3 scripts/vvip.py doctor --source-root /path/to/vllm/package
```

Exit 0 means the 35 audited files match; exit 2 means a missing installation or a version/source mismatch. `source_only=true` identifies a source-copy check. `gpu_validated=false` is intentional: doctor never runs GPU acceptance. It cannot inspect a remote Python environment over an inference HTTP endpoint.

Record model revision, file SHA256, quantization, template, image digest, and hardware. A served-model alias does not identify the weights. This example uses the tested hybrid checkpoint:

```sh
export VVIP_MODEL=/models/Qwen3.8-27B-NVFP4
mkdir -p artifacts
python3 scripts/vvip.py serve "$VVIP_MODEL" --mode shadow --action recompute -- \
  --served-model-name qwen38-vvip --port 8000 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.60 --enforce-eager \
  --no-enable-prefix-caching --mamba-cache-mode none \
  --language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'
```

This prints JSON only: no service launch or download. Memory and length limits are experiment settings, not universal defaults. Keep the checkpoint's quantization, tokenizer, chat template, and tool parsers. The published acceptance covers text only.

### Download pinned weights

Reuse existing weights when their identity matches. Otherwise, with the Hugging Face CLI installed in a download environment:

```sh
hf download nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir "$VVIP_MODEL"
```

See the [HF CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli). This public checkpoint did not require credentials in the experiment. Mirror downloads must still match the pinned upstream LFS SHA256 values. The three weight shards total about 21.92 GB; allow additional space for caches, logs, and temporary downloads.

### Isolated Docker acceptance

The tested image is `docker.m.daocloud.io/vllm/vllm-openai:v0.30.0`; its exact digest is in [validation](validation.md). Check source inside the container as well. The following example uses no external network or host port mapping:

```sh
export VVIP_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:v0.30.0
export VVIP_SKILL_DIR="$PWD"
mkdir -p artifacts

docker run --rm --network none \
  -v "$VVIP_SKILL_DIR:/skill:ro" --entrypoint python3 \
  "$VVIP_IMAGE" /skill/scripts/vvip.py doctor

docker run -d --name vvip-test --gpus device=0 --network none --shm-size=2g \
  -e VVIP_TRIGGER_WAIT_MS=0 -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -v "$VVIP_SKILL_DIR:/skill:ro" -v "$VVIP_MODEL:/model:ro" \
  -v "$VVIP_SKILL_DIR/artifacts:/evidence" \
  --entrypoint /bin/bash "$VVIP_IMAGE" -c '
    exec python3 /skill/scripts/vvip.py serve /model \
      --mode enforce --action recompute --execute -- \
      --served-model-name qwen38-vvip --max-model-len 8192 --max-num-seqs 1 \
      --max-num-batched-tokens 1024 --gpu-memory-utilization 0.60 --enforce-eager \
      --no-enable-prefix-caching --mamba-cache-mode none --language-model-only \
      --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
      > /evidence/recompute-server.log 2>&1'

docker exec vvip-test python3 -c \
  'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health").status)'
docker exec vvip-test python3 /skill/scripts/gpu_smoke.py \
  --model qwen38-vvip --action recompute --timeout 300 \
  --server-log /evidence/recompute-server.log --out /evidence/recompute.json

# After saving evidence, remove only this test container; retain images and weights.
docker stop vvip-test
docker rm vvip-test
```

Health checks can fail while loading. Inspect `artifacts/recompute-server.log` and wait for readiness before running the client. Choose Docker or the host-process approach below; do not launch both on the same GPU. Use fresh logs and report names for every action.

## 2. Recompute acceptance

On a **dedicated test instance**, `max_num_seqs=1` creates deterministic slot contention. Do not apply this low concurrency limit to a shared production service. `--execute` must precede the `--` separator.

Terminal A:

```sh
VVIP_TRIGGER_WAIT_MS=0 python3 scripts/vvip.py serve "$VVIP_MODEL" \
  --mode enforce --action recompute --execute -- \
  --served-model-name qwen38-vvip --port 8000 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.60 --enforce-eager \
  --no-enable-prefix-caching --mamba-cache-mode none \
  --language-model-only --limit-mm-per-prompt '{"image":0,"video":0}' \
  > artifacts/recompute-server.log 2>&1
```

The launcher owns loopback binding, native priority policy, synchronous execution, and scheduler selection; conflicting duplicate options are rejected. Once the model is ready, terminal B:

```sh
curl --fail http://127.0.0.1:8000/health
python3 scripts/gpu_smoke.py --model qwen38-vvip --action recompute \
  --timeout 300 --server-log artifacts/recompute-server.log \
  --out artifacts/recompute.json
```

The client runs a 512-token baseline, a 512-token ordinary request, a 16-token VIP after ordinary output begins, and an 8-token reuse probe. Passing requires correlated engine IDs/boot, real preemption and resume events, one complete SSE terminal, and resumed token IDs identical to the baseline. Reports retain counts and hashes, not generated text.

For longer history, submit the VIP only after at least 128 output tokens:

```sh
python3 scripts/gpu_smoke.py --model qwen38-vvip --action recompute \
  --prompt-file /path/to/synthetic-prompt.txt --preempt-after-tokens 128 \
  --tokens 512 --timeout 300 --server-log artifacts/recompute-server.log \
  --out artifacts/recompute-mid-generation.json
```

The baseline and victim use the same prompt; its SHA256 is recorded. Inspect `computed_tokens_before` and `recomputed.tokens`. Input plus output must fit `max_model_len`. Use synthetic prompts. The client accepts up to 8,192 output tokens and a 600-second per-request timeout.

Early completion, token mismatch, a disconnected HTTP stream, or missing correlated events do not pass. Quantization or batch shape can change numerical results; investigate and preserve differences rather than weakening the check and claiming identity.

### Client deadlines and failed runs

`--timeout` is an absolute per-request deadline, including HTTP headers and all
output. Optional `--ttft-timeout` covers the time through the first real output;
`--idle-timeout` limits gaps after output begins. Both default to the total budget.
Headers, heartbeats, and empty role/start frames do not start or reset output-idle
timing. For example, use `--timeout 600 --ttft-timeout 480 --idle-timeout 120` for
a bounded long-prefill test. A recompute pause must fit the chosen idle budget.
The total deadline never resets, even while the server trickles a partial line.

These flags also apply to `benchmark.py`; keep them identical across compared
variants. Reports record the budgets, and the analyzer rejects mismatched policies.
The clients connect directly without redirects, environment proxy discovery, or
automatic retries. Use loopback or a tunnel to the isolated native endpoint.

Smoke transport failures and benchmark warmup failures write failed reports.
Retain those alongside successful runs; do not reinterpret a missing terminal or
HTTP 200 as success. Use a fresh engine log and output path. Detected log rotation,
truncation, engine restart, or mixed smoke identities invalidates the evidence.
See [reliability and recovery](reliability.md) for limitations and ingress tests.

## 3. Abort and controls

Stop the dedicated recompute instance. Relaunch with the same model options but `--mode enforce --action abort` and a new `artifacts/abort-server.log`, then run:

```sh
python3 scripts/gpu_smoke.py --model qwen38-vvip --action abort \
  --timeout 300 --server-log artifacts/abort-server.log --out artifacts/abort.json
```

The victim must have run before receiving exactly one `finish_reason="abort"` and `stop_reason="vvip_preempted"`, correlated with `abort_delivered`. VIP and follow-up requests must succeed. Abort is not ordinary-request completion.

Run separate instances with `--mode off --action recompute` and `--mode shadow --action recompute`, matching the client mode and log path:

```sh
python3 scripts/gpu_smoke.py --model qwen38-vvip --mode off --action recompute \
  --timeout 300 --server-log artifacts/off-server.log --out artifacts/off.json
python3 scripts/gpu_smoke.py --model qwen38-vvip --mode shadow --action recompute \
  --timeout 300 --server-log artifacts/shadow-server.log --out artifacts/shadow.json
```

Off must have no actual VVIP preemption. Shadow may log `would_preempt` but must allow ordinary completion. Reports are not overwritten; reruns need new paths. For authenticated native endpoints, set `VVIP_API_KEY` in the client environment. Do not put credentials in arguments, reports, or Git.

### Runtime negative controls

Before starting a dedicated single-sequence enforce server, configure `VVIP_DISABLE_FILE` with an existing writable parent directory. After the four modes pass:

```sh
python3 scripts/gpu_guards.py --model qwen38-vvip \
  --server-log artifacts/recompute-server.log \
  --disable-file /absolute/operator-owned/vvip.disabled --out artifacts/guards.json
```

The script verifies the logged switch path, equal/reverse-priority behavior, disabled preemption, and a client disconnect after its first token followed by successful HTTP reuse. The switch file must initially be absent; the script removes only its own file. The disconnect record explicitly has `native_release_verified=false`: a successful follow-up does not distinguish native cancellation from the original request finishing naturally. Do not use this check as proof of private-KV release, reclaimed capacity, cancellation latency, or absence of leaks. Stronger claims require exact native evidence through the actual ingress; see [reliability](reliability.md).

## 4. Application requests

Native completion/chat requests accept `priority`. Smaller values are more urgent; ordinary `0` and VIP `-10` are a convention. This example needs concurrent lower-priority running work to create contention:

```sh
curl --fail-with-body -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-vvip","prompt":"Explain a binary search in one paragraph.","max_tokens":128,"stream":true,"priority":-10}'
```

An OpenAI-compatible SDK can pass `extra_body={"priority": -10}`. VVIP does not add Responses, tool-calling, structured-output, or vision capabilities absent from the model/server.

- **Recompute:** the ID, delivered history, and connection survive. Client timeouts must allow the pause.
- **Abort:** inspect the terminal even when HTTP is 200. Partial text is not a complete answer. VVIP does not retry, refund, or translate the response to HTTP 503.
- **Authorization:** trusted ingress must validate or assign both JSON `priority` and `X-Vllm-Priority`; the header takes precedence upstream. An ordinary API key must not allow self-promotion.

## 5. Policy and live disable switch

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `VVIP_MODE` | off | off/shadow/enforce; explicit launcher options take precedence |
| `VVIP_ACTION` | recompute | recompute/abort |
| `VVIP_TRIGGER_WAIT_MS` | 100 | Minimum wait in this engine before triggering |
| `VVIP_MIN_INTERVAL_MS` | 200 | Minimum interval between decisions |
| `VVIP_MAX_PER_MINUTE` | 60 | Rolling decision limit, including shadow decisions |
| `VVIP_MAX_PER_REQUEST` | 2 | Maximum interruptions per live request |
| `VVIP_PROTECT_AFTER_S` | 30 | Stop selecting a victim after this age from first scheduling |
| `VVIP_DISABLE_FILE` | unset | Operator-owned absolute path; presence disables new decisions |

Time is monotonic; protected age includes pauses. A protection value of zero immediately protects every request. Strict priority under unlimited VIP arrivals does not bound ordinary waiting; admission control remains external. These settings do not disable native memory-pressure preemption.

Configure the path before server startup, with an operator-controlled parent directory:

```sh
export VVIP_DISABLE_FILE=/absolute/operator-owned/vvip.disabled
# In another terminal sharing the same filesystem:
touch /absolute/operator-owned/vvip.disabled
# Remove your switch file to resume the configured mode:
rm /absolute/operator-owned/vvip.disabled
```

A present or unreadable file stops new decisions at the next scheduling check. It does not restore an aborted request or reverse completed preemption. Other environment settings are read at process startup. Containers must mount the switch directory consistently.

## 6. Repeated workloads

After correctness acceptance, use `max_num_seqs=4` and matched model/hardware/cache settings. Include native priority+sync, recompute, and abort. Remove `--scheduler-cls` for the native baseline while preserving the comparison settings. Native async is a separate opportunity-cost control, not part of the same baseline.

```sh
python3 scripts/benchmark.py --model qwen38-vvip --variant recompute \
  --scenario saturated --trials 8 --block 0 \
  --server-log artifacts/recompute-server.log --out artifacts/bench-recompute-b0.json
```

Scenarios are `spare`, `saturated`, `burst`, and `long-prefill`. Use identical block/trials/SLO across variants and save actual server arguments. Aggregate complete matching reports:

```sh
python3 scripts/analyze_benchmark.py artifacts/bench-*.json --out artifacts/summary.json
```

The report includes latency quantiles, VIP SLO, ordinary completion, total and completed-request throughput, and replay/interruption counts. Each trial drains before the next; finite bursts do not measure sustained capacity. Errors and aborted requests remain in the denominators.

For token-hash differences, run an additional same-process control with concurrency four, prefix caching off, enforce/recompute mode, and an initially absent configured switch file:

```sh
python3 scripts/gpu_determinism.py --model qwen38-vvip \
  --server-log artifacts/recompute-server.log \
  --disable-file /absolute/operator-owned/vvip.disabled --out artifacts/repeat-controls.json
```

After warmup, this runs off/off/recompute/off/recompute/recompute with identical workloads, retaining hashes and first divergence positions. `request_contract_passed` means terminal and switch/preemption checks passed—not token identity. Read `comparisons` for that result. Do not silently change kernels or quantization to suppress differences; a different backend requires a separate record. This diagnostic is outside the performance matrix.

## 7. Troubleshooting and rollback

| Symptom | Response |
| --- | --- |
| Source drift | Restore the pinned build or audit an upgrade; never bypass hashes |
| Native model loading fails | Resolve model/hardware/backend support before evaluating preemption |
| Unsupported configuration | Use a supported contract or implement an evidenced adaptation |
| No preemption event | Check actual pressure, strict priority, wait threshold, age protection, and rate limits |
| Token mismatch | Preserve evidence; compare native repeats, batch shapes, and state rebuilding |
| Disconnect without abort terminal | Treat as failure; inspect EngineCore-to-API delivery |
| Faster VIPs but lower useful throughput | Quantify replay/discard costs and choose policy with both tiers in view |

Rollback: create the disable file, drain requests, stop the dedicated service, then restart with native vLLM options. No installed vLLM source was patched. Remove only identified test containers/processes; retain images, weights, and evidence separately. Bounded acceptance does not replace production workload testing or a long-running soak.
