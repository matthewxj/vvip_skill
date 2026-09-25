# Reliability checks and incident recovery

Use this reference for stalled responses, disconnects, missing request evidence,
or deployment drift. The engine-local scheduler has no remote capacity ledger,
Redis scan, release controller, or receipt database. Do not add those components
to recover a local request or assume that a distributed-system fix applies here.

## Classify the failure before changing policy

| Observed failure | Independent skill boundary and required action |
| --- | --- |
| A remote slot remains reserved after HTTP completion | External ownership is outside VVIP. Preserve the exact request identity and require that owner's release protocol; a local log is not its receipt. |
| A release record disappeared after process restart | VVIP events are diagnostic and may be lost. Absence is unknown, not proof of release. Recompute does not survive an engine crash. |
| A controller repeatedly scans all resources or skips paginated candidates | No such scanner exists in this runtime. In an external integration, consume each page before advancing its cursor, bound retries, and keep blocking discovery off the request loop. Backoff never authorizes release. |
| Long prefill fails shortly after headers or an empty role frame | Separate absolute total/first-output budgets from output-idle timing. Headers, heartbeat, role, and start frames do not reset a deadline or establish an output boundary. |
| A completed response hangs during listener cleanup | Check the actual ingress's cancellation path. Cancel and join listeners without swallowing cancellation; if a polling API swallows it, check the task's cancellation state before polling again. VVIP does not patch an external gateway. |
| Client disconnect occurs after a successful terminal | Preserve that successful terminal; do not overwrite it with a later cancellation or infer a refund/retry. Billing and retries remain application responsibilities. |
| Request fails before the worker can be identified | Do not infer non-dispatch from connection errors, missing journal entries, or a 404. A separate dispatcher needs a durable, identity-bound non-dispatch result and replay prevention. |
| An incompatible worker is still serving after rollout | Inspect every actual target's source, image, runtime hash, and boot identity. A control-plane declaration or one healthy replica is not deployment proof. |

## What the bundled clients establish

`gpu_smoke.py` and `benchmark.py` accept `--timeout` (absolute per-request
budget), `--ttft-timeout` (request through first real output), and `--idle-timeout`
(gap after real output). Phase budgets default to the total budget; they never
extend it. A shared stdlib client enforces deadlines while HTTP headers or an
incomplete SSE line are being read. Empty frames do not count as output.

The client connects directly to the native completion endpoint: HTTP or
certificate-verified HTTPS, no redirects, proxy discovery, or automatic retries.
Use loopback or a tunnel for an isolated test. Python socket timeouts do not bound
the OS hostname resolver; resolve an unresponsive host separately. CLI total
budgets remain capped at 600 seconds. A model requiring a longer experiment needs
a separately reviewed bounded harness, not a reduced input presented as equivalent.

Smoke and benchmark warmup failures produce failed JSON reports. Completed smoke
records and available engine events remain in the report after a later request
fails. Error types are retained without copying server response bodies or URLs.
An output file must be writable and absent; a killed process cannot promise a report.
Do not discard failed reports or count a failed warmup as a valid performance run.

Evidence readers reject detected log rotation/truncation and new engine startup
events during a run. Smoke evidence must have the same nonempty boot/runtime
identity, a unique preemption, and the matching single resume or abort delivery.
Save logs with the report. These local checks do not authenticate an external log
or replace a durable remote receipt.

`gpu_guards.py` checks equal/reverse priorities, the disable switch, and whether a
new HTTP request completes after closing an output stream. Its disconnect record
explicitly says `native_release_verified=false`: the original task might have
finished naturally before the probe ran. This is **not** proof of native abort,
private-KV release, capacity reclamation, or bounded cancellation latency.

For a deployment that needs those stronger claims, obtain exact native request
status and resource ownership from the actual engine, including its boot identity.
Test disconnect before first output, during output, and after successful terminal
through the real ingress. A direct-engine test cannot certify gateway/protocol
conversion, authentication, billing, or distributed transfer behavior.

## Recovery and rollout

1. Preserve the target identity, effective configuration, original input size,
   timing phase, request IDs, and logs. Keep customer content and infrastructure
   identities out of shared reports. A short health probe cannot reproduce a large
   prefill or long-running request.
2. Disable new VVIP decisions with the configured switch if needed. This does not
   cancel all work, undo an abort, release an external reservation, or make a
   missing receipt appear. Never bulk-delete ownership records to make counts zero.
3. Use an isolated instance to reproduce the specific failure, then run
   off/shadow/recompute/abort with matched inputs and explicit failure accounting.
   Bound test concurrency so the test itself does not overload a shared ingress.
4. Before an authorized restart, stop new admissions and establish the drain or
   maintenance boundary. In an external dispatcher, an old execution generation
   being gone is useful only with identity-bound replay prevention and the
   dispatcher's guarded ownership transition. A restart alone is not a receipt.
5. After rollout, verify every serving instance and test the user-facing path.
   Report retained uncertainty separately from successful new requests. Do not
   certify unsupported P/D, connectors, async execution, or untested models by
   reusing unrelated historical evidence.

## September 2026 hardening scope

Regression tests reproduce partial-line deadline extension, missing failure
reports, protocol-only frames, slow headers, output-idle stalls, cancellation of
the client watchdog, log replacement/restart, mixed engine identities, duplicate
abort delivery, and lost events after client failure. Configuration tests keep
distributed transfer and multiple in-flight batches rejected.

These are CPU/local-HTTP checks. The audited scheduler and its 35-file source
profile are unchanged. The [September 23 GPU evidence](validation.md) retains its
original runtime hash and measured scope; the hardened clients have not yet been
rerun on GPU. Historical disconnect/reuse results must not be promoted to native
release certification.
