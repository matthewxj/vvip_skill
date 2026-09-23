"""Engine-independent selection; smaller integer means higher priority."""
from dataclasses import dataclass
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    mode: str = "off"
    action: str = "recompute"
    trigger_wait_ms: float = 100
    min_interval_ms: float = 200
    max_per_minute: int = 60
    max_per_request: int = 2
    protect_after_s: float = 30
    disable_file: str = ""

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError("mode must be off, shadow, or enforce")
        if self.action not in {"recompute", "abort"}:
            raise ValueError("action must be recompute or abort")
        if self.disable_file and not Path(self.disable_file).is_absolute():
            raise ValueError("disable_file must be an absolute path")
        for name in ("trigger_wait_ms", "min_interval_ms", "protect_after_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_per_minute", "max_per_request"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(**{
            name: type(getattr(defaults, name))(
                os.environ.get("VVIP_" + name.upper(), getattr(defaults, name)))
            for name in cls.__dataclass_fields__
        })


def choose_victim(requester, running, counts, first_running, now, settings):
    """Use actual scheduler objects; no model names or platform lease metadata."""
    candidates = (
        req for req in running
        if req.priority > requester.priority
        and not req.resumable
        and not req.has_encoder_inputs
        and counts.get(req.request_id, 0) < settings.max_per_request
        and now - first_running[req.request_id] < settings.protect_after_s
        and req.num_in_flight_tokens == 0
    )
    return max(candidates, key=lambda req: (
        req.priority,
        -req.num_output_tokens / max(1, req.max_tokens),
        req.num_computed_tokens,
        req.arrival_time,
        req.request_id,
    ), default=None)
