from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from sglang_service.catalog import ModelCatalog, ModelSpec
from sglang_service.manager import (
    Backend,
    BackendBusyError,
    BackendState,
    ModelManager,
    PortAllocator,
    PortUnavailable,
)

pytestmark = pytest.mark.anyio


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
            ModelSpec(name="model-a", model_path="org/model-a", aliases=("a",)),
            ModelSpec(name="model-b", model_path="org/model-b", aliases=("b",)),
        ],
        Path("models.toml"),
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

    tasks = [asyncio.create_task(manager.acquire("a")) for _ in range(20)]
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

    with pytest.raises(RuntimeError, match="still releasing"):
        await manager.acquire("model-a")
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


def test_catalog_rejects_normalized_and_alias_collisions() -> None:
    with pytest.raises(ValueError, match="lookup collision"):
        ModelCatalog(
            [
                ModelSpec(name="model.a", model_path="org/a"),
                ModelSpec(name="model-a", model_path="org/b"),
            ],
            Path("models.toml"),
        )

    with pytest.raises(ValueError, match="lookup collision"):
        ModelCatalog(
            [
                ModelSpec(name="model-a", model_path="org/a", aliases=("shared",)),
                ModelSpec(name="model-b", model_path="org/b", aliases=("shared",)),
            ],
            Path("models.toml"),
        )


def test_command_rejects_gateway_managed_extra_args() -> None:
    catalog = ModelCatalog(
        [
            ModelSpec(
                name="model-a",
                model_path="org/a",
                vllm_args=("--port=31000",),
            )
        ],
        Path("models.toml"),
    )
    manager = ModelManager(catalog)
    with pytest.raises(ValueError, match="--port"):
        manager._command(catalog.default, "org/a", 30000)

    underscore_spec = ModelSpec(
        name="model-b",
        model_path="org/b",
        vllm_args=("--tensor_parallel_size=8",),
    )
    underscore_manager = ModelManager(ModelCatalog([underscore_spec], Path("models.toml")))
    with pytest.raises(ValueError, match="--tensor_parallel_size"):
        underscore_manager._command(underscore_spec, "org/b", 30001)


def test_global_idle_timeout_overrides_per_model_zero() -> None:
    spec = ModelSpec(name="model-a", model_path="org/a", idle_seconds=0.0)
    manager = ModelManager(ModelCatalog([spec], Path("models.toml")), idle_seconds=60.0)
    assert manager._idle_threshold(spec) == 60.0


def test_command_defaults_to_single_gpu_when_parallelism_is_unspecified() -> None:
    spec = ModelSpec(name="model-a", model_path="org/a")
    manager = ModelManager(ModelCatalog([spec], Path("models.toml")))
    command = manager._command(spec, "org/a", 30000)
    assert command[command.index("--tensor-parallel-size") + 1] == "1"


async def test_reload_keeps_loaded_model_and_adds_new(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=2, shutdown_timeout=0.1)
    install_fake_processes(monkeypatch, manager)

    lease = await manager.acquire("model-a")
    old_process = lease.backend.process
    new_catalog = ModelCatalog(
        [
            ModelSpec(name="model-a", model_path="org/model-a", aliases=("a",)),
            ModelSpec(name="model-c", model_path="org/model-c"),
        ],
        Path("models.toml"),
    )
    result = await manager.reload_catalog(new_catalog)

    assert result == {
        "reloaded": True,
        "config_file": str(new_catalog.config_file),
        "default_model": "model-a",
        "added": ["model-c"],
        "removed": ["model-b"],
        "updated": [],
        "kept_loaded": ["model-a"],
    }
    assert manager.catalog is new_catalog
    assert manager._backends["model-a"].process is old_process
    health = await manager.health()
    assert health["models"] == ["model-a", "model-c"]

    await lease.release()
    await manager.shutdown()


async def test_reload_stops_backend_of_removed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=2, shutdown_timeout=0.1)
    install_fake_processes(monkeypatch, manager)

    lease = await manager.acquire("model-b")
    await lease.release()
    removed_process = manager._backends["model-b"].process
    assert removed_process is not None and removed_process.returncode is None

    new_catalog = ModelCatalog(
        [ModelSpec(name="model-a", model_path="org/model-a")],
        Path("models.toml"),
    )
    result = await manager.reload_catalog(new_catalog)

    assert result["removed"] == ["model-b"]
    assert "model-b" not in manager._backends
    assert removed_process is not None and removed_process.returncode == 0
    with pytest.raises(KeyError):
        await manager.acquire("model-b")
    await manager.shutdown()


async def test_reload_restarts_updated_model_with_new_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ModelManager(make_catalog(), max_loaded_models=2, shutdown_timeout=0.1)
    starts = install_fake_processes(monkeypatch, manager)

    lease = await manager.acquire("model-a")
    await lease.release()
    old_pid = manager._backends["model-a"].process.pid  # type: ignore[union-attr]

    new_catalog = ModelCatalog(
        [
            ModelSpec(name="model-a", model_path="org/model-a", aliases=("a",), context_length=8192),
            ModelSpec(name="model-b", model_path="org/model-b", aliases=("b",)),
        ],
        Path("models.toml"),
    )
    result = await manager.reload_catalog(new_catalog)

    assert result["updated"] == ["model-a"]
    assert "model-a" not in manager._backends

    fresh_lease = await manager.acquire("model-a")
    assert fresh_lease.backend.spec.context_length == 8192
    assert fresh_lease.backend.process.pid != old_pid  # type: ignore[union-attr]
    assert starts == ["model-a", "model-a"]

    await fresh_lease.release()
    await manager.shutdown()


async def test_reload_recomputes_default_max_loaded_models_only() -> None:
    four_models = ModelCatalog(
        [
            ModelSpec(name=f"model-{index}", model_path=f"org/model-{index}")
            for index in range(4)
        ],
        Path("models.toml"),
    )
    one_model = ModelCatalog(
        [ModelSpec(name="model-a", model_path="org/model-a")],
        Path("models.toml"),
    )

    defaulted = ModelManager(four_models)
    assert defaulted.max_loaded_models == 2
    await defaulted.reload_catalog(one_model)
    assert defaulted.max_loaded_models == 1
    await defaulted.shutdown()

    pinned = ModelManager(four_models, max_loaded_models=4)
    await pinned.reload_catalog(one_model)
    assert pinned.max_loaded_models == 4
    await pinned.shutdown()
