from __future__ import annotations

import pytest

from vllm_gateway.diagnostics import BackendOutput, BackendStartupError, diagnostic_from_exception


def test_kv_signal_survives_chunks_ansi_and_tail_overflow() -> None:
    output = BackendOutput()
    for chunk in (b"\xff\x1b[31mValueError: No available mem", b"ory for the cache blocks\x1b[0m\n"):
        output.feed(chunk)
    output.feed(b"x" * 100_000)
    assert len(output.tail) <= 65536
    assert output.code == "kv_cache_memory_insufficient"


def test_kv_pool_smaller_than_one_request_is_classified() -> None:
    output = BackendOutput()
    output.feed(
        b"ValueError: To serve at least one request with the model's max seq len (8192), (0.37 GiB KV cache is "
        b"needed, which is larger than the available KV cache memory (0.34 GiB)."
    )
    assert output.code == "kv_cache_memory_insufficient"


@pytest.mark.parametrize("text", [b"CUDA out of memory", b"torch.OutOfMemoryError: secret /home/user"])
def test_cuda_oom_has_fixed_safe_summary(text: bytes) -> None:
    output = BackendOutput()
    output.feed(text)
    diagnostic = diagnostic_from_exception(RuntimeError("secret /home/user"), code=output.code)
    assert diagnostic.code == "cuda_out_of_memory"
    assert "secret" not in diagnostic.message
    assert "/home" not in diagnostic.message
    assert len(diagnostic.diagnostic_id) == 32


def test_unknown_and_timeout_do_not_echo_internal_exception() -> None:
    output = BackendOutput()
    output.feed(b"WARNING low memory; api_key=secret")
    assert output.code is None
    unknown = diagnostic_from_exception(RuntimeError("secret traceback"))
    timeout = diagnostic_from_exception(TimeoutError("secret endpoint"))
    assert unknown.code == "backend_startup_failed"
    assert timeout.code == "backend_startup_timeout"
    assert "secret" not in unknown.message + timeout.message
    error = BackendStartupError("internal code 1", unknown)
    assert error.diagnostic is unknown
    assert str(error) == "internal code 1"
