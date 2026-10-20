from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass

from .catalog import ModelSpec

GIB = 1 << 30

# Calibrate these on the target GPU with a real model; every estimate derives from them.
KV_FULL_SEQS = 2  # sequences of full context length the KV cache must hold
KV_MARGIN = 1.2  # block rounding
# vLLM reserves several times the raw recurrent state per sequence for hybrid models. Qwen3.5-9B needed
# 245 MB per sequence for 51.5 MB of raw state (fitted from its 8k and 32k KV cache logs), so 5.0 leaves a margin.
LINEAR_STATE_FACTOR = 5.0
# CUDA context, CUDA graphs and activations. Measured about 2 GiB on top of weights + KV for Qwen3.5-9B.
OVERHEAD_BYTES = 4 * GIB
SAFETY_BYTES = 2 * GIB  # headroom that must stay free on top of the estimate


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    gpu_memory_utilization: float

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.overhead_bytes

    @property
    def required_free_bytes(self) -> int:
        return self.total_bytes + SAFETY_BYTES


def plan_memory(spec: ModelSpec, gpu_total_bytes: int) -> MemoryPlan:
    per_sequence = spec.linear_state_bytes * LINEAR_STATE_FACTOR + spec.kv_bytes_per_token * (spec.context_length or 0)
    kv_cache_bytes = math.ceil(per_sequence * KV_FULL_SEQS * KV_MARGIN)
    total = spec.weights_bytes + kv_cache_bytes + OVERHEAD_BYTES
    utilization = min(1.0, math.ceil(total * 100 / gpu_total_bytes) / 100)
    return MemoryPlan(spec.weights_bytes, kv_cache_bytes, OVERHEAD_BYTES, utilization)


def _parse_smi(output: str) -> tuple[int, int]:
    free_mib, total_mib = (int(part.strip()) for part in output.strip().splitlines()[0].split(","))
    return free_mib << 20, total_mib << 20


def query_gpu_memory(index: int = 0) -> tuple[int, int]:
    """Return (free, total) bytes of the device as seen by the whole GPU, including other users."""

    try:
        import pynvml
    except ImportError:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return _parse_smi(result.stdout)
    pynvml.nvmlInit()
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
        return int(info.free), int(info.total)
    finally:
        pynvml.nvmlShutdown()
