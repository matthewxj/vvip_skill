"""Exact-source gates are compatibility checks, not GPU certification."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def manifest():
    return json.loads(Path(__file__).with_name("compatibility.json").read_text())


def inspect_source(root=None):
    expected = manifest()
    installed_version = None
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.origin:
            return {"compatible": False, "errors": ["vllm is not installed"]}
        root = Path(spec.origin).parent
        installed_version = importlib.metadata.version("vllm")
    root = Path(root).resolve()
    errors = []
    if installed_version is not None and installed_version != expected["vllm_version"]:
        errors.append(f"version {installed_version} != {expected['vllm_version']}")
    for name, digest in expected["sha256"].items():
        try:
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        except OSError:
            actual = "missing"
        if actual != digest:
            errors.append(f"source drift: {name} ({actual})")
    return {"compatible": not errors, "source_root": str(root),
            "vllm_version": installed_version or expected["vllm_version"],
            "source_only": installed_version is None,
            "gpu_validated": False, "errors": errors}


def require_compatible():
    report = inspect_source()
    if not report["compatible"]:
        raise RuntimeError("VVIP refuses unsupported vLLM: " + "; ".join(report["errors"]))


def check_config(config, kv_cache_config=None):
    """No permissive defaults for scheduler/parallel safety invariants."""
    sc, pc, mc = config.scheduler_config, config.parallel_config, config.model_config
    failures = []
    if sc.policy != "priority" or sc.async_scheduling is not False:
        failures.append("priority policy and --no-async-scheduling required")
    for name in ("data_parallel_size", "pipeline_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(pc, name) != 1:
            failures.append(f"{name} must equal 1")
    if config.max_concurrent_batches != 1:
        failures.append("exactly one in-flight batch required")
    for name in ("kv_transfer_config", "ec_transfer_config", "speculative_config", "lora_config"):
        if getattr(config, name) is not None:
            failures.append(f"{name} is not supported yet")
    if mc.runner_type != "generate" or mc.is_encoder_decoder or mc.is_diffusion:
        failures.append("autoregressive decoder generation required")
    if mc.is_attention_free:
        failures.append("attention-free models require separate validation")
    if mc.is_hybrid:
        # Audited GDN path: release all state groups, restart at token zero.
        # Prefix checkpoints/CoW have a different state restoration contract.
        cache = config.cache_config
        specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups] if kv_cache_config else []
        states = [spec for spec in specs if hasattr(spec, "mamba_type")]
        if (not states or len(states) == len(specs)
                or any(getattr(spec.mamba_type, "name", None) != "GDN_ATTN"
                       or spec.mamba_cache_mode != "none"
                       or spec.num_speculative_blocks != 0 for spec in states)
                or cache.enable_prefix_caching is not False
                or cache.mamba_cache_mode != "none"):
            failures.append("hybrid requires native GDN_ATTN + attention cache groups, "
                            "--no-enable-prefix-caching and --mamba-cache-mode none")
    if failures:
        raise RuntimeError("VVIP unsupported configuration: " + "; ".join(failures))
