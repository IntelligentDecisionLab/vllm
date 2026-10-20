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


def test_hybrid_pool_covers_what_vllm_needs_for_one_request() -> None:
    """Qwen3.5-9B, from the vLLM logs: one 8192-token request needs 0.37 GiB, one 32768-token request ~0.853 GB."""

    raw_state = 24 * (32 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 32 * 128) * 2)
    per_token = 2 * 9 * 4 * 256

    def pool(context: int) -> int:
        model = ModelSpec(
            name="m",
            model_path="/m",
            context_length=context,
            weights_bytes=18 * GIB,
            kv_bytes_per_token=per_token,
            linear_state_bytes=raw_state,
        )
        return plan_memory(model, 96 * GIB).kv_cache_bytes

    assert pool(8192) >= 2 * int(0.37 * GIB)
    assert pool(32768) >= 2 * 853_000_000


def test_dense_model_has_no_recurrent_state_term() -> None:
    plan = plan_memory(spec(1, 57344, 4096), 96 * GIB)

    assert plan.kv_cache_bytes == math.ceil(57344 * 4096 * KV_FULL_SEQS * KV_MARGIN)
