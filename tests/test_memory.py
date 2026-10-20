from __future__ import annotations

import math

import pytest

from vllm_gateway.catalog import ModelSpec
from vllm_gateway.memory import (
    GIB,
    KV_FULL_SEQS,
    KV_MARGIN,
    OVERHEAD_BYTES,
    SAFETY_BYTES,
    _parse_smi,
    plan_memory,
)


def spec(weights_gib: float, kv_per_token: int, context: int) -> ModelSpec:
    return ModelSpec(
        name="m",
        model_path="/m",
        context_length=context,
        weights_bytes=int(weights_gib * GIB),
        kv_bytes_per_token=kv_per_token,
    )


def test_plan_adds_weights_kv_pool_and_overhead() -> None:
    plan = plan_memory(spec(20, 16384, 32768), 96 * GIB)

    expected_kv = int(16384 * 32768 * KV_FULL_SEQS * KV_MARGIN)
    assert plan.kv_cache_bytes == pytest.approx(expected_kv, abs=1)
    assert plan.weights_bytes == 20 * GIB
    assert plan.total_bytes == 20 * GIB + plan.kv_cache_bytes + OVERHEAD_BYTES
    assert plan.required_free_bytes == plan.total_bytes + SAFETY_BYTES


def test_kv_pool_scales_with_context_only_by_the_per_token_size() -> None:
    short = plan_memory(spec(10, 8192, 4096), 96 * GIB)
    long = plan_memory(spec(10, 8192, 131072), 96 * GIB)

    assert long.kv_cache_bytes == pytest.approx(short.kv_cache_bytes * 32, rel=1e-6)
    assert long.weights_bytes == short.weights_bytes


def test_utilization_is_rounded_up_to_two_decimals_and_capped() -> None:
    plan = plan_memory(spec(10, 0, 4096), 100 * GIB)
    assert plan.gpu_memory_utilization == pytest.approx(math.ceil((10 * GIB + OVERHEAD_BYTES) / GIB) / 100)
    assert plan_memory(spec(10.5, 0, 4096), 100 * GIB).gpu_memory_utilization == pytest.approx(
        math.ceil((10.5 * GIB + OVERHEAD_BYTES) / GIB) / 100
    )

    assert plan_memory(spec(200, 0, 4096), 100 * GIB).gpu_memory_utilization == 1.0


def test_parse_nvidia_smi_output_in_mib() -> None:
    assert _parse_smi("57700, 97887\n") == (57700 * 1024**2, 97887 * 1024**2)
