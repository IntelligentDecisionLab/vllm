from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import httpx
import pytest

from vllm_gateway.catalog import ModelCatalog, ModelSpec
from vllm_gateway.diagnostics import BackendOutput, BackendStartupError
from vllm_gateway.manager import (
    Backend,
    BackendBusyError,
    BackendLease,
    BackendState,
    CapacityError,
    ModelManager,
    PortAllocator,
    PortUnavailable,
)

pytestmark = pytest.mark.anyio

MODEL_DIR = str(Path(__file__).parent)  # any existing directory satisfies the model files check


class FakeProcess:
    next_pid = 20000

    def __init__(self) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode: int | None = None
        self._finished = asyncio.Event()

    async def wait(self) -> int:
        await self._finished.wait()
        assert self.returncode is not None
        return self.returncode

    def finish(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._finished.set()


def make_catalog() -> ModelCatalog:
    return ModelCatalog(
        [
            ModelSpec(name="model-a", model_path=MODEL_DIR, repo_id="org/a", context_length=4096),
            ModelSpec(name="model-b", model_path=MODEL_DIR, repo_id="org/b", context_length=8192),
        ]
    )


def install_fake_processes(
    monkeypatch: pytest.MonkeyPatch,
    manager: ModelManager,
    *,
    gates: dict[str, asyncio.Event] | None = None,
) -> list[str]:
    starts: list[str] = []

    async def fake_start(backend: Backend) -> None:
        starts.append(backend.spec.name)
        if gates is not None:
            await gates[backend.spec.name].wait()
        backend.process = FakeProcess()

    async def fake_stop(backend: Backend) -> None:
        process = backend.process
        if isinstance(process, FakeProcess) and process.returncode is None:
            process.finish(0)
        backend.process = None

    monkeypatch.setattr(manager, "_start_backend", fake_start)
    monkeypatch.setattr(manager, "_stop_backend", fake_stop)
    return starts


async def test_same_model_concurrent_acquire_starts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=2, shutdown_timeout=0.1)
    gate = asyncio.Event()
    starts = install_fake_processes(monkeypatch, manager, gates={"model-a": gate})

    tasks = [asyncio.create_task(manager.acquire("model-a")) for _ in range(20)]
    while not starts:
        await asyncio.sleep(0)
    assert starts == ["model-a"]

    gate.set()
    leases = await asyncio.gather(*tasks)
    assert len({id(lease.backend) for lease in leases}) == 1
    assert leases[0].backend.active_requests == 20

    await asyncio.gather(*(lease.release() for lease in leases))
    assert leases[0].backend.active_requests == 0
    await manager.shutdown()


async def test_different_models_start_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=2, shutdown_timeout=0.1)
    gates = {"model-a": asyncio.Event(), "model-b": asyncio.Event()}
    starts = install_fake_processes(monkeypatch, manager, gates=gates)

    task_a = asyncio.create_task(manager.acquire("model-a"))
    task_b = asyncio.create_task(manager.acquire("model-b"))
    while len(starts) < 2:
        await asyncio.sleep(0)
    assert set(starts) == {"model-a", "model-b"}

    gates["model-a"].set()
    gates["model-b"].set()
    lease_a, lease_b = await asyncio.gather(task_a, task_b)
    assert lease_a.backend.port != lease_b.backend.port

    await lease_a.release()
    await lease_b.release()
    health = await manager.health()
    assert health["loaded_models"] == ["model-a", "model-b"]
    assert health["loaded_model"] == "model-a"
    assert health["capacity"] == {"max_loaded_models": 2, "resident_count": 2}
    await manager.shutdown()


async def test_shutdown_drains_active_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=1.0)
    install_fake_processes(monkeypatch, manager)
    lease = await manager.acquire("model-a")

    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0.01)
    assert not shutdown.done()

    await lease.release()
    await asyncio.wait_for(shutdown, timeout=1.0)
    status = await manager.backend_status("model-a")
    assert status["state"] == BackendState.STOPPED.value


async def test_drain_timeout_retains_port_until_late_lease_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.01)
    install_fake_processes(monkeypatch, manager)
    lease = await manager.acquire("model-a")
    port = lease.backend.port

    await manager.shutdown()
    status = await manager.backend_status("model-a")
    assert status["state"] == BackendState.FAILED.value
    assert manager._ports._by_model["model-a"] == port

    await lease.release()
    assert "model-a" not in manager._ports._by_model


async def test_unload_rejects_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.1)
    install_fake_processes(monkeypatch, manager)
    lease = await manager.acquire("model-a")

    with pytest.raises(BackendBusyError, match="active request"):
        await manager.unload("model-a")

    await lease.release()
    assert await manager.unload("model-a") is True
    assert await manager.unload("model-a") is False
    await manager.shutdown()


async def test_cancelled_unload_request_does_not_cancel_backend_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.5)
    install_fake_processes(monkeypatch, manager)
    lease = await manager.acquire("model-a")
    await lease.release()

    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()

    async def slow_stop(backend: Backend) -> None:
        stop_started.set()
        await allow_stop.wait()
        process = backend.process
        if isinstance(process, FakeProcess) and process.returncode is None:
            process.finish(0)
        backend.process = None

    monkeypatch.setattr(manager, "_stop_backend", slow_stop)
    unloading = asyncio.create_task(manager.unload("model-a"))
    await stop_started.wait()
    unloading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await unloading

    allow_stop.set()
    for _ in range(20):
        status = await manager.backend_status("model-a")
        if status["state"] == BackendState.STOPPED.value:
            break
        await asyncio.sleep(0)
    assert status["state"] == BackendState.STOPPED.value
    await manager.shutdown()


async def test_unload_cancels_in_progress_startup_before_releasing_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.1)
    gate = asyncio.Event()
    starts = install_fake_processes(monkeypatch, manager, gates={"model-a": gate})

    acquiring = asyncio.create_task(manager.acquire("model-a"))
    while not starts:
        await asyncio.sleep(0)

    assert await manager.unload("model-a") is True
    with pytest.raises(RuntimeError, match="loading model 'model-a' was cancelled"):
        await acquiring
    status = await manager.backend_status("model-a")
    assert status["state"] == BackendState.STOPPED.value
    assert status["pid"] is None

    gate.set()
    lease = await manager.acquire("model-a")
    assert starts == ["model-a", "model-a"]
    await lease.release()
    await manager.shutdown()


async def test_process_crash_marks_failed_and_can_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(
        make_catalog(),
        max_loaded_models=2,
        failure_cooldown=0.0,
        shutdown_timeout=0.1,
    )
    starts = install_fake_processes(monkeypatch, manager)
    lease = await manager.acquire("model-a")
    old_port = lease.backend.port
    process = lease.backend.process
    assert isinstance(process, FakeProcess)

    process.finish(17)
    for _ in range(20):
        status = await manager.backend_status("model-a")
        if status["state"] == BackendState.FAILED.value:
            break
        await asyncio.sleep(0)
    assert status["state"] == BackendState.FAILED.value
    assert "code 17" in status["error"]
    assert manager._ports._by_model["model-a"] == old_port

    with pytest.raises(BackendStartupError, match="still releasing") as still_releasing:
        await manager.acquire("model-a")
    assert still_releasing.value.diagnostic is lease.backend.diagnostic
    assert still_releasing.value.diagnostic.diagnostic_id == status["diagnostic"]["diagnostic_id"]
    other = await manager.acquire("model-b")
    assert other.backend.port != old_port
    await other.release()

    await lease.release()
    assert "model-a" not in manager._ports._by_model
    second = await manager.acquire("model-a")
    assert starts == ["model-a", "model-b", "model-a"]
    await second.release()
    await manager.shutdown()


async def test_cancelled_waiter_does_not_cancel_shared_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.1)
    gate = asyncio.Event()
    starts = install_fake_processes(monkeypatch, manager, gates={"model-a": gate})

    cancelled = asyncio.create_task(manager.acquire("model-a"))
    survivor = asyncio.create_task(manager.acquire("model-a"))
    while not starts:
        await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    gate.set()
    lease = await survivor
    assert starts == ["model-a"]
    await lease.release()
    await manager.shutdown()


def test_port_allocator_never_takes_foreign_listener() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        allocator = PortAllocator(port, 1)
        with pytest.raises(PortUnavailable, match="externally occupied"):
            allocator.reserve("model-a")
        assert listener.fileno() >= 0
    finally:
        listener.close()


async def test_startup_diagnostic_drains_exit_tail_and_reuses_cooldown_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(make_catalog(), failure_cooldown=60)
    outputs = iter([b"ValueError: No available memory for the cache blocks\n", b"unknown error secret\n"])

    async def fake_exec(*args: object, **kwargs: object) -> FakeProcess:
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.STDOUT
        process = FakeProcess()
        process.stdout = asyncio.StreamReader()  # type: ignore[attr-defined]
        process.stdout.feed_data(next(outputs))  # type: ignore[attr-defined]
        process.stdout.feed_eof()  # type: ignore[attr-defined]
        process.finish(1)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(BackendStartupError) as first:
        await manager.acquire("model-a")
    diagnostic = first.value.diagnostic
    assert diagnostic.code == "kv_cache_memory_insufficient"
    with pytest.raises(BackendStartupError) as cooldown:
        await manager.acquire("model-a")
    assert cooldown.value.diagnostic is diagnostic
    status = await manager.backend_status("model-a")
    assert status["diagnostic"] == diagnostic.as_dict()
    backend = manager._backends["model-a"]
    assert backend.output_task is None
    assert "model-a" not in manager._ports._by_model

    manager.failure_cooldown = 0
    with pytest.raises(BackendStartupError) as restart:
        await manager.acquire("model-a")
    assert restart.value.diagnostic.code == "backend_startup_failed"
    assert restart.value.diagnostic.diagnostic_id != diagnostic.diagnostic_id
    await manager.shutdown()


async def test_timeout_and_parallel_backend_diagnostics_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), startup_timeout=0, max_loaded_models=2)

    async def fake_start(backend: Backend) -> None:
        if backend.spec.name == "model-a":
            backend.output.feed(b"CUDA out of memory")
            raise RuntimeError("exited code 1 secret")
        raise TimeoutError("secret URL")

    monkeypatch.setattr(manager, "_start_backend", fake_start)
    results = await asyncio.gather(manager.acquire("model-a"), manager.acquire("model-b"), return_exceptions=True)
    assert all(isinstance(result, BackendStartupError) for result in results)
    statuses = [await manager.backend_status(name) for name in ("model-a", "model-b")]
    assert statuses[0]["diagnostic"]["code"] == "cuda_out_of_memory"
    assert statuses[1]["diagnostic"]["code"] == "backend_startup_timeout"
    assert statuses[0]["diagnostic"]["diagnostic_id"] != statuses[1]["diagnostic"]["diagnostic_id"]
    await manager.shutdown()


async def test_non_eof_output_cleanup_is_bounded_and_cancellation_keeps_no_diagnostic() -> None:
    manager = ModelManager(make_catalog())
    backend = Backend(manager.catalog.get("model-a"), 30999)
    reader = asyncio.StreamReader()
    task = asyncio.create_task(manager._read_output(reader, backend.output, "model-a", 1))
    backend.output_task = task
    await asyncio.wait_for(manager._stop_backend(backend), timeout=2)
    assert task.done()
    assert backend.output_task is None
    assert backend.diagnostic is None
    await manager.shutdown()


async def test_successful_backend_continues_draining_large_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=1)
    script = (
        "import sys,time; sys.stdout.write('x'*1000000); sys.stdout.flush(); "
        "sys.stderr.write('END-OF-OUTPUT\\n'); sys.stderr.flush(); time.sleep(120)"
    )
    monkeypatch.setattr(manager, "_command", lambda *args: [sys.executable, "-c", script])
    monkeypatch.setattr(PortAllocator, "_is_available", staticmethod(lambda port: True))

    async def ready(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    monkeypatch.setattr(manager._client, "get", ready)
    seen = asyncio.Event()
    backend_output_feed = manager._read_output

    async def observe(reader: asyncio.StreamReader, output: BackendOutput, name: str, generation: int) -> None:
        original_feed = output.feed

        def feed(chunk: bytes) -> None:
            original_feed(chunk)
            if b"END-OF-OUTPUT" in output.tail:
                seen.set()

        output.feed = feed
        await backend_output_feed(reader, output, name, generation)

    monkeypatch.setattr(manager, "_read_output", observe)
    lease = await manager.acquire("model-a")
    try:
        await asyncio.wait_for(seen.wait(), timeout=5)
        assert lease.backend.state is BackendState.READY
        assert len(lease.backend.output.tail) <= 65536
        assert lease.backend.output_task is not None and not lease.backend.output_task.done()
    finally:
        await lease.release()
        await manager.shutdown()
    assert lease.backend.output_task is None


async def test_cancelled_startup_finishes_reader_without_recording_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.1)
    entered = asyncio.Event()
    process = FakeProcess()
    process.stdout = asyncio.StreamReader()  # type: ignore[attr-defined]

    async def spawn(*args: object, **kwargs: object) -> FakeProcess:
        return process

    async def blocked_health(*args: object, **kwargs: object) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    def terminate(pid: int, sig: int) -> None:
        assert pid == process.pid
        process.finish()
        process.stdout.feed_eof()  # type: ignore[attr-defined]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(manager._client, "get", blocked_health)
    monkeypatch.setattr("vllm_gateway.manager.os.killpg", terminate)
    acquiring = asyncio.create_task(manager.acquire("model-a"))
    await entered.wait()
    assert await manager.unload("model-a")
    with pytest.raises(RuntimeError, match="was cancelled"):
        await acquiring
    backend = manager._backends["model-a"]
    assert backend.output_task is None
    assert backend.diagnostic is None
    assert backend.state is BackendState.STOPPED
    await manager.shutdown()


async def test_crash_with_inherited_pipe_does_not_block_watcher() -> None:
    manager = ModelManager(make_catalog())

    class PipeHoldingProcess(FakeProcess):
        async def wait(self) -> int:
            await asyncio.Event().wait()
            raise AssertionError("pipe never closes")

    process = PipeHoldingProcess()
    process.returncode = 1
    reader = asyncio.StreamReader()
    process.stdout = reader  # type: ignore[attr-defined]
    reader.feed_data(b"torch.OutOfMemoryError: private path\n")
    backend = Backend(manager.catalog.get("model-a"), 30999, state=BackendState.READY, process=process)
    manager._backends["model-a"] = backend
    backend.output_task = asyncio.create_task(manager._read_output(reader, backend.output, "model-a", 0))
    await asyncio.wait_for(manager._watch_process(backend, 0, process), timeout=2)
    assert backend.state is BackendState.FAILED
    assert backend.diagnostic is not None
    assert backend.diagnostic.code == "cuda_out_of_memory"
    assert backend.output_task is None
    await manager.shutdown()


def flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_command_uses_fixed_profile_and_alias_settings() -> None:
    spec = ModelSpec(
        name="my-alias",
        model_path="/snap/qwen",
        repo_id="org/qwen",
        context_length=32768,
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        mtp=True,
    )
    command = ModelManager(ModelCatalog([spec]))._command(spec, 30123)

    assert command[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert flag_value(command, "--model") == "/snap/qwen"
    assert flag_value(command, "--served-model-name") == "my-alias"
    assert flag_value(command, "--host") == "127.0.0.1"
    assert flag_value(command, "--port") == "30123"
    assert flag_value(command, "--max-model-len") == "32768"
    assert flag_value(command, "--kv-cache-dtype") == "fp8_e4m3"
    assert flag_value(command, "--attention-backend") == "flashinfer"
    assert flag_value(command, "--max-num-seqs") == "16"
    assert flag_value(command, "--max-num-batched-tokens") == "2048"
    assert flag_value(command, "--reasoning-parser") == "qwen3"
    assert flag_value(command, "--tool-call-parser") == "qwen3_coder"
    assert "--enable-auto-tool-choice" in command
    assert json.loads(flag_value(command, "--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    for flag in ("--async-scheduling", "--enable-flashinfer-autotune", "--enable-prefix-caching"):
        assert flag in command
    assert "--device-ids" not in command


def test_command_omits_family_flags_for_unknown_family() -> None:
    spec = ModelSpec(name="plain", model_path="/snap/plain", context_length=4096)
    command = ModelManager(ModelCatalog([spec]))._command(spec, 30000)

    for flag in ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice", "--speculative-config"):
        assert flag not in command


async def test_lease_works_as_async_context_manager() -> None:
    manager = ModelManager(make_catalog())
    backend = Backend(manager.catalog.get("model-a"), 30000, active_requests=1)

    async with BackendLease(manager, backend) as lease:
        assert lease.backend.active_requests == 1

    assert backend.active_requests == 0
    await manager.shutdown()


async def test_capacity_rejects_second_active_model_then_evicts_idle_one(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=1, shutdown_timeout=0.1)
    install_fake_processes(monkeypatch, manager)

    first = await manager.acquire("model-a")
    with pytest.raises(CapacityError, match="active or loading"):
        await manager.acquire("model-b")

    await first.release()
    second = await manager.acquire("model-b")
    assert second.backend.spec.name == "model-b"
    assert manager._backends["model-a"].state is BackendState.STOPPED
    await second.release()
    await manager.shutdown()


async def test_restart_uses_the_current_spec_of_a_recreated_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), shutdown_timeout=0.1)
    install_fake_processes(monkeypatch, manager)

    lease = await manager.acquire("model-a")
    await lease.release()
    assert await manager.unload("model-a")

    manager.catalog = ModelCatalog([ModelSpec(name="model-a", model_path=MODEL_DIR, context_length=16384)])
    lease = await manager.acquire("model-a")

    assert lease.backend.spec.context_length == 16384
    await lease.release()
    await manager.shutdown()
