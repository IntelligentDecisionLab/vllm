from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from .catalog import ModelCatalog, ModelSpec, resolve_model_path

LOGGER = logging.getLogger("vllm_gateway")


class BackendState(StrEnum):
    STOPPED = "stopped"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"


class CapacityError(RuntimeError):
    pass


class BackendBusyError(RuntimeError):
    pass


class PortUnavailable(RuntimeError):
    pass


class PortAllocator:
    """Assign stable ports without taking ownership of foreign processes."""

    def __init__(self, base_port: int, count: int = 64):
        if not 1 <= base_port <= 65535:
            raise ValueError("backend_port must be between 1 and 65535")
        if count < 1 or base_port + count - 1 > 65535:
            raise ValueError("backend_port_count exceeds the valid TCP port range")
        self.base_port = base_port
        self.count = count
        self._by_model: dict[str, int] = {}
        self._by_port: dict[int, str] = {}

    @staticmethod
    def _is_available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        finally:
            sock.close()
        return True

    def reserve(self, model_name: str) -> int:
        existing = self._by_model.get(model_name)
        if existing is not None:
            return existing

        digest = hashlib.sha256(model_name.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") % self.count
        occupied: list[int] = []
        for step in range(self.count):
            port = self.base_port + ((offset + step) % self.count)
            if port in self._by_port:
                continue
            if not self._is_available(port):
                occupied.append(port)
                continue
            self._by_model[model_name] = port
            self._by_port[port] = model_name
            return port

        detail = f"; externally occupied ports: {occupied}" if occupied else ""
        raise PortUnavailable(
            f"no backend port is available in {self.base_port}-{self.base_port + self.count - 1}{detail}"
        )

    def release(self, model_name: str, port: int) -> None:
        if self._by_model.get(model_name) == port:
            self._by_model.pop(model_name, None)
        if self._by_port.get(port) == model_name:
            self._by_port.pop(port, None)


@dataclass(slots=True)
class Backend:
    spec: ModelSpec
    port: int
    state: BackendState = BackendState.STOPPED
    process: asyncio.subprocess.Process | None = None
    active_requests: int = 0
    last_used: float = 0.0
    generation: int = 0
    start_task: asyncio.Task[None] | None = None
    watcher_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None
    error: str | None = None
    failed_at: float | None = None


class BackendLease:
    def __init__(self, manager: ModelManager, backend: Backend):
        self._manager = manager
        self.backend = backend
        self._released = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._manager.release(self.backend)

    async def __aenter__(self) -> BackendLease:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.release()


class ModelManager:
    """Own independent vLLM subprocesses and route leases by canonical model."""

    _RESIDENT_STATES = {
        BackendState.LOADING,
        BackendState.READY,
        BackendState.DRAINING,
    }
    _PROTECTED_VLLM_FLAGS = {
        "--model",
        "--model-path",
        "--host",
        "--port",
        "--served-model-name",
        "--tensor-parallel-size",
        "--data-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--kv-cache-dtype",
        "--load-format",
        "--runner",
        "--reasoning-parser",
        "--tool-call-parser",
        "--chat-template",
        "--attention-backend",
        "--hf-overrides",
        "--device-ids",
        "--async-scheduling",
        "--enable-flashinfer-autotune",
        "--speculative-config",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
    }

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        backend_port: int = 30000,
        backend_port_count: int = 64,
        max_loaded_models: int | None = None,
        tensor_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        attention_backend: str | None = None,
        startup_timeout: float = 900.0,
        shutdown_timeout: float = 30.0,
        idle_seconds: float = 0.0,
        context_length_override: int | None = None,
        failure_cooldown: float = 2.0,
        vllm_args: tuple[str, ...] = (),
    ):
        if tensor_parallel_size is not None and tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        if data_parallel_size is not None and data_parallel_size < 1:
            raise ValueError("data_parallel_size must be at least 1")
        configured_limit = min(2, len(catalog.specs)) if max_loaded_models is None else max_loaded_models
        if configured_limit < 1:
            raise ValueError("max_loaded_models must be at least 1")
        self._max_loaded_models_defaulted = max_loaded_models is None
        self._validate_extra_args(vllm_args, "global vLLM arguments")

        self.catalog = catalog
        self.backend_port = backend_port  # compatibility: now the allocator range start
        self.backend_port_count = backend_port_count
        self.max_loaded_models = configured_limit
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.attention_backend = attention_backend
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.idle_seconds = idle_seconds
        self.context_length_override = context_length_override
        self.failure_cooldown = failure_cooldown
        self.vllm_args = vllm_args

        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._backends: dict[str, Backend] = {}
        self._ports = PortAllocator(backend_port, backend_port_count)
        self._stopping = False
        self._started = False
        self._idle_task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0))

    @classmethod
    def _validate_extra_args(cls, args: Iterable[str], source: str) -> None:
        for arg in args:
            if not arg.startswith("--"):
                continue
            flag = arg.split("=", 1)[0]
            normalized_flag = flag.replace("_", "-")
            if normalized_flag.startswith("--no-"):
                normalized_flag = "--" + normalized_flag.removeprefix("--no-")
            if normalized_flag in cls._PROTECTED_VLLM_FLAGS:
                raise ValueError(f"{source} cannot override gateway-managed option {flag}")

    async def start(self, preload: str | Iterable[str] | None = None) -> None:
        self._started = True
        if self.idle_seconds > 0 or any(spec.idle_seconds and spec.idle_seconds > 0 for spec in self.catalog.specs):
            self._idle_task = asyncio.create_task(self._idle_reaper(), name="vllm-idle-reaper")
        if preload:
            names = [preload] if isinstance(preload, str) else list(preload)
            await asyncio.gather(*(self.load(name) for name in names))

    async def shutdown(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)
            self._idle_task = None

        async with self._lock:
            self._stopping = True
            startup_tasks = [
                backend.start_task
                for backend in self._backends.values()
                if backend.start_task and not backend.start_task.done()
            ]
            for task in startup_tasks:
                task.cancel()
            self._changed.notify_all()

        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

        async with self._lock:
            stop_tasks: list[asyncio.Task[None]] = []
            for backend in self._backends.values():
                if backend.state in self._RESIDENT_STATES or backend.process is not None:
                    backend.state = BackendState.DRAINING
                    if not backend.stop_task or backend.stop_task.done():
                        backend.stop_task = asyncio.create_task(
                            self._drain_and_stop(backend, backend.generation, self.shutdown_timeout),
                            name=f"vllm-stop-{backend.spec.name}",
                        )
                    stop_tasks.append(backend.stop_task)
            self._changed.notify_all()

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        watcher_tasks = [
            backend.watcher_task
            for backend in self._backends.values()
            if backend.watcher_task and not backend.watcher_task.done()
        ]
        if watcher_tasks:
            await asyncio.gather(*watcher_tasks, return_exceptions=True)
        await self._client.aclose()
        self._started = False

    async def reload_catalog(self, new_catalog: ModelCatalog) -> dict[str, Any]:
        """Swap in a new catalog, stopping backends whose spec changed or was removed.

        Unchanged models with resident backends keep running untouched; newly added
        models start lazily on their first request.
        """
        old_specs = {spec.name: spec for spec in self.catalog.specs}
        new_specs = {spec.name: spec for spec in new_catalog.specs}
        added = sorted(set(new_specs) - set(old_specs))
        removed = sorted(set(old_specs) - set(new_specs))
        updated = sorted(name for name in set(old_specs) & set(new_specs) if old_specs[name] != new_specs[name])
        kept = sorted(set(old_specs) & set(new_specs) - set(updated))

        cancelled_startups: list[asyncio.Task[None]] = []
        stop_tasks: list[asyncio.Task[None]] = []
        async with self._lock:
            if self._stopping:
                raise RuntimeError("model manager is shutting down")
            for name in (*removed, *updated):
                backend = self._backends.pop(name, None)
                if backend is None:
                    continue
                if backend.start_task and not backend.start_task.done():
                    backend.state = BackendState.DRAINING
                    backend.start_task.cancel()
                    cancelled_startups.append(backend.start_task)
                elif backend.state in self._RESIDENT_STATES or backend.process is not None:
                    backend.state = BackendState.DRAINING
                    if not backend.stop_task or backend.stop_task.done():
                        backend.stop_task = asyncio.create_task(
                            self._drain_and_stop(backend, backend.generation, self.shutdown_timeout),
                            name=f"vllm-reload-{name}",
                        )
                    stop_tasks.append(backend.stop_task)
            kept_loaded = sorted(
                name
                for name in kept
                if (backend := self._backends.get(name)) and backend.state in self._RESIDENT_STATES
            )
            self.catalog = new_catalog
            if self._max_loaded_models_defaulted:
                self.max_loaded_models = min(2, len(new_catalog.specs))
            self._changed.notify_all()

        if cancelled_startups:
            await asyncio.gather(*cancelled_startups, return_exceptions=True)
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        return {
            "reloaded": True,
            "config_file": str(new_catalog.config_file),
            "default_model": new_catalog.default.name,
            "added": added,
            "removed": removed,
            "updated": updated,
            "kept_loaded": kept_loaded,
        }

    async def load(self, model_name: str | None) -> dict[str, Any]:
        lease = await self.acquire(model_name)
        name = lease.backend.spec.name
        await lease.release()
        return await self.backend_status(name)

    async def unload(self, model_name: str | None) -> bool:
        spec = self.catalog.get(model_name)
        startup_task: asyncio.Task[None] | None = None
        stop_task: asyncio.Task[None] | None = None
        async with self._lock:
            backend = self._backends.get(spec.name)
            if not backend or backend.state in {BackendState.STOPPED, BackendState.FAILED}:
                return False
            if backend.active_requests:
                raise BackendBusyError(f"model {spec.name!r} has {backend.active_requests} active request(s)")
            backend.state = BackendState.DRAINING
            generation = backend.generation
            if backend.start_task and not backend.start_task.done():
                startup_task = backend.start_task
                startup_task.cancel()
            else:
                if not backend.stop_task or backend.stop_task.done():
                    backend.stop_task = asyncio.create_task(
                        self._drain_and_stop(backend, generation, self.shutdown_timeout),
                        name=f"vllm-unload-{spec.name}",
                    )
                stop_task = backend.stop_task
            self._changed.notify_all()

        if startup_task is not None:
            await asyncio.gather(startup_task, return_exceptions=True)
            return True
        if stop_task is not None:
            await asyncio.shield(stop_task)
        return True

    async def acquire(self, model_name: str | None) -> BackendLease:
        spec = self.catalog.get(model_name)
        while True:
            startup_task: asyncio.Task[None] | None = None
            eviction_task: asyncio.Task[None] | None = None
            async with self._lock:
                if self._stopping:
                    raise RuntimeError("model manager is shutting down")

                backend = self._backends.get(spec.name)
                if (
                    backend
                    and backend.state is BackendState.READY
                    and backend.process
                    and backend.process.returncode is None
                ):
                    backend.active_requests += 1
                    backend.last_used = time.monotonic()
                    return BackendLease(self, backend)

                if backend and backend.state is BackendState.READY:
                    backend.state = BackendState.FAILED
                    backend.error = "vLLM backend process exited"
                    backend.failed_at = time.monotonic()
                    self._ports.release(backend.spec.name, backend.port)

                if backend and backend.state is BackendState.LOADING:
                    startup_task = backend.start_task
                elif backend and backend.state is BackendState.DRAINING:
                    await self._changed.wait()
                    continue
                else:
                    if backend and backend.state is BackendState.FAILED and backend.active_requests > 0:
                        raise RuntimeError(
                            f"model {spec.name!r} backend failed while "
                            f"{backend.active_requests} request(s) are still releasing"
                        )
                    if (
                        backend
                        and backend.state is BackendState.FAILED
                        and backend.failed_at is not None
                        and time.monotonic() - backend.failed_at < self.failure_cooldown
                    ):
                        raise RuntimeError(
                            f"model {spec.name!r} is in failure cooldown: {backend.error or 'startup failed'}"
                        )

                    resident = [item for item in self._backends.values() if item.state in self._RESIDENT_STATES]
                    if len(resident) >= self.max_loaded_models:
                        candidates = [
                            item for item in resident if item.state is BackendState.READY and item.active_requests == 0
                        ]
                        if candidates:
                            victim = min(candidates, key=lambda item: item.last_used)
                            victim.state = BackendState.DRAINING
                            if not victim.stop_task or victim.stop_task.done():
                                victim.stop_task = asyncio.create_task(
                                    self._drain_and_stop(victim, victim.generation, self.shutdown_timeout),
                                    name=f"vllm-evict-{victim.spec.name}",
                                )
                            eviction_task = victim.stop_task
                            self._changed.notify_all()
                        elif any(item.state is BackendState.DRAINING for item in resident):
                            await self._changed.wait()
                            continue
                        else:
                            raise CapacityError(f"all {self.max_loaded_models} backend slot(s) are active or loading")
                    else:
                        port = self._ports.reserve(spec.name)
                        if backend is None:
                            backend = Backend(spec=spec, port=port)
                            self._backends[spec.name] = backend
                        else:
                            backend.port = port
                        backend.generation += 1
                        backend.state = BackendState.LOADING
                        backend.error = None
                        backend.failed_at = None
                        backend.last_used = time.monotonic()
                        generation = backend.generation
                        backend.start_task = asyncio.create_task(
                            self._start_generation(backend, generation),
                            name=f"vllm-start-{spec.name}",
                        )
                        startup_task = backend.start_task
                        self._changed.notify_all()

            if eviction_task is not None:
                await asyncio.shield(eviction_task)
                continue
            if startup_task is None:
                continue
            try:
                await asyncio.shield(startup_task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                raise RuntimeError(f"loading model {spec.name!r} was cancelled") from None

    async def release(self, backend: Backend | None) -> None:
        if backend is None:
            return
        async with self._lock:
            backend.active_requests = max(0, backend.active_requests - 1)
            backend.last_used = time.monotonic()
            if backend.active_requests == 0 and backend.state is BackendState.FAILED and backend.process is None:
                self._ports.release(backend.spec.name, backend.port)
            self._changed.notify_all()

    async def backend_status(self, model_name: str | None) -> dict[str, Any]:
        spec = self.catalog.get(model_name)
        async with self._lock:
            backend = self._backends.get(spec.name)
            return self._backend_dict(backend, spec.name)

    async def health(self) -> dict[str, Any]:
        async with self._lock:
            ready = sorted(
                backend.spec.name
                for backend in self._backends.values()
                if backend.state is BackendState.READY and backend.process and backend.process.returncode is None
            )
            active_requests = sum(backend.active_requests for backend in self._backends.values())
            backend_snapshots = {name: self._backend_dict(backend, name) for name, backend in self._backends.items()}
            resident_count = sum(backend.state in self._RESIDENT_STATES for backend in self._backends.values())
            stopping = self._stopping
        details = [
            backend_snapshots.get(spec.name) or self._backend_dict(None, spec.name) for spec in self.catalog.specs
        ]
        return {
            "status": "ok" if not stopping else "stopping",
            "service": "vllm-model-router",
            "loaded_model": (
                self.catalog.default.name if self.catalog.default.name in ready else (ready[0] if ready else None)
            ),
            "loaded_models": ready,
            "active_requests": active_requests,
            "models": [spec.name for spec in self.catalog.specs],
            "resident_models": details,
            "capacity": {
                "max_loaded_models": self.max_loaded_models,
                "resident_count": resident_count,
            },
            "stopping": stopping,
        }

    @staticmethod
    def _backend_dict(backend: Backend | None, name: str) -> dict[str, Any]:
        if backend is None:
            return {
                "name": name,
                "state": BackendState.STOPPED.value,
                "port": None,
                "pid": None,
                "active_requests": 0,
                "last_used": None,
                "error": None,
            }
        process = backend.process
        return {
            "name": name,
            "state": backend.state.value,
            "port": backend.port,
            "pid": process.pid if process and process.returncode is None else None,
            "active_requests": backend.active_requests,
            "last_used": backend.last_used,
            "error": backend.error,
        }

    async def send(
        self,
        backend: Backend,
        method: str,
        path: str,
        *,
        query: str = "",
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> httpx.Response:
        url = f"http://127.0.0.1:{backend.port}/{path}"
        if query:
            url = f"{url}?{query}"
        request = self._client.build_request(method, url, headers=headers, content=content)
        return await self._client.send(request, stream=True)

    async def _start_generation(self, backend: Backend, generation: int) -> None:
        try:
            await self._start_backend(backend)
        except BaseException as exc:
            await self._stop_backend(backend)
            async with self._lock:
                if backend.generation == generation:
                    backend.start_task = None
                    backend.process = None
                    backend.error = None if isinstance(exc, asyncio.CancelledError) else str(exc)
                    backend.failed_at = None if isinstance(exc, asyncio.CancelledError) else time.monotonic()
                    backend.state = (
                        BackendState.STOPPED
                        if self._stopping or backend.state is BackendState.DRAINING
                        else BackendState.FAILED
                    )
                    self._ports.release(backend.spec.name, backend.port)
                    self._changed.notify_all()
            raise

        should_stop = False
        async with self._lock:
            if backend.generation != generation or self._stopping or backend.state is not BackendState.LOADING:
                should_stop = True
            else:
                backend.state = BackendState.READY
                backend.start_task = None
                backend.error = None
                backend.failed_at = None
                backend.last_used = time.monotonic()
                process = backend.process
                if process:
                    backend.watcher_task = asyncio.create_task(
                        self._watch_process(backend, generation, process),
                        name=f"vllm-watch-{backend.spec.name}",
                    )
                self._changed.notify_all()

        if should_stop:
            await self._stop_backend(backend)
            async with self._lock:
                if backend.generation == generation:
                    backend.state = BackendState.STOPPED
                    backend.start_task = None
                    backend.process = None
                    self._ports.release(backend.spec.name, backend.port)
                    self._changed.notify_all()

    async def _watch_process(
        self,
        backend: Backend,
        generation: int,
        process: asyncio.subprocess.Process,
    ) -> None:
        returncode = await process.wait()
        async with self._lock:
            if backend.generation != generation or backend.process is not process:
                return
            backend.process = None
            if backend.active_requests == 0:
                self._ports.release(backend.spec.name, backend.port)
            if backend.state not in {BackendState.DRAINING, BackendState.STOPPED}:
                backend.state = BackendState.FAILED
                backend.error = f"vLLM backend exited with code {returncode}"
                backend.failed_at = time.monotonic()
            self._changed.notify_all()

    async def _start_backend(self, backend: Backend) -> None:
        if not PortAllocator._is_available(backend.port):
            raise PortUnavailable(f"backend port {backend.port} for model {backend.spec.name!r} became unavailable")

        model_path = await asyncio.to_thread(resolve_model_path, backend.spec)
        command = self._command(backend.spec, model_path, backend.port)
        LOGGER.info("starting vLLM backend for %s: %s", backend.spec.name, " ".join(command))
        env = os.environ.copy()
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        if os.path.isdir(cuda_home):
            env["CUDA_HOME"] = cuda_home
            cuda_bin = os.path.join(cuda_home, "bin")
            current_path = env.get("PATH", "")
            if cuda_bin not in current_path.split(os.pathsep):
                env["PATH"] = f"{cuda_bin}{os.pathsep}{current_path}"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        backend.process = await asyncio.create_subprocess_exec(
            *command,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

        deadline = time.monotonic() + self.startup_timeout
        last_error = "backend did not become ready"
        while time.monotonic() < deadline:
            if backend.process.returncode is not None:
                raise RuntimeError(
                    f"vLLM backend for {backend.spec.name!r} exited with "
                    f"code {backend.process.returncode}; inspect its stderr"
                )
            try:
                health = await self._client.get(
                    f"http://127.0.0.1:{backend.port}/health",
                    timeout=httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0),
                )
                if health.status_code != 200:
                    last_error = f"HTTP {health.status_code} from /health"
                    await asyncio.sleep(1.0)
                    continue

                models = await self._client.get(
                    f"http://127.0.0.1:{backend.port}/v1/models",
                    timeout=httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0),
                )
                if models.status_code != 200:
                    last_error = f"HTTP {models.status_code} from /v1/models"
                    await asyncio.sleep(1.0)
                    continue
                payload = models.json()
                served = {
                    str(item.get("id"))
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id") is not None
                }
                if backend.spec.name not in served:
                    last_error = f"/v1/models did not advertise expected model {backend.spec.name!r}"
                    await asyncio.sleep(1.0)
                    continue
                LOGGER.info("vLLM backend ready: %s on port %d", backend.spec.name, backend.port)
                return
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last_error = str(exc)
            await asyncio.sleep(1.0)

        raise TimeoutError(
            f"timed out after {self.startup_timeout:.0f}s waiting for vLLM backend {backend.spec.name!r} ({last_error})"
        )

    def _command(self, spec: ModelSpec, model_path: str, port: int) -> list[str]:
        self._validate_extra_args(spec.vllm_args, f"vLLM arguments for model {spec.name!r}")
        context_length = self.context_length_override or spec.context_length
        tp_size = self.tensor_parallel_size or spec.tensor_parallel_size or 1
        dp_size = self.data_parallel_size or spec.data_parallel_size or 1
        gpu_mem = (
            self.gpu_memory_utilization
            if self.gpu_memory_utilization is not None
            else (spec.gpu_memory_utilization if spec.gpu_memory_utilization is not None else 0.90)
        )
        max_requests = (
            self.max_num_seqs
            if self.max_num_seqs is not None
            else (spec.max_num_seqs if spec.max_num_seqs is not None else 256)
        )
        batched_tokens = (
            self.max_num_batched_tokens
            if self.max_num_batched_tokens is not None
            else (spec.max_num_batched_tokens if spec.max_num_batched_tokens is not None else 4096)
        )

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            spec.name,
            "--tensor-parallel-size",
            str(tp_size),
            "--data-parallel-size",
            str(dp_size),
            "--gpu-memory-utilization",
            str(gpu_mem),
            "--max-num-seqs",
            str(max_requests),
            "--max-num-batched-tokens",
            str(batched_tokens),
        ]
        command.extend(("--kv-cache-dtype", spec.kv_cache_dtype or "auto"))
        if context_length:
            command.extend(("--max-model-len", str(context_length)))
        if spec.reasoning_parser:
            command.extend(("--reasoning-parser", spec.reasoning_parser))
        if spec.tool_call_parser:
            command.extend(("--enable-auto-tool-choice", "--tool-call-parser", spec.tool_call_parser))
        if spec.chat_template:
            command.extend(("--chat-template", spec.chat_template))
        attention_backend = self.attention_backend or spec.attention_backend
        if attention_backend:
            command.extend(("--attention-backend", attention_backend))
        if spec.device_ids is not None:
            command.extend(("--device-ids", ",".join(str(device) for device in spec.device_ids)))
        if spec.async_scheduling is not None:
            command.append("--async-scheduling" if spec.async_scheduling else "--no-async-scheduling")
        if spec.flashinfer_autotune is not None:
            command.append(
                "--enable-flashinfer-autotune" if spec.flashinfer_autotune else "--no-enable-flashinfer-autotune"
            )
        if spec.enable_prefix_caching is not None:
            command.append("--enable-prefix-caching" if spec.enable_prefix_caching else "--no-enable-prefix-caching")
        if spec.speculative_method is not None:
            speculative_config: dict[str, Any] = {"method": spec.speculative_method}
            if spec.num_speculative_tokens is not None:
                speculative_config["num_speculative_tokens"] = spec.num_speculative_tokens
            command.extend(("--speculative-config", json.dumps(speculative_config)))

        override_args: dict[str, Any] = {}
        if spec.rope_scaling or spec.rope_scale is not None or spec.yarn_orig_ctx is not None:
            rope_dict: dict[str, Any] = {}
            if spec.rope_scaling:
                rope_dict["rope_type"] = spec.rope_scaling.lower()
            if spec.rope_scale is not None:
                rope_dict["factor"] = spec.rope_scale
            if spec.yarn_orig_ctx is not None:
                rope_dict["original_max_position_embeddings"] = spec.yarn_orig_ctx
            override_args["rope_scaling"] = rope_dict
        if spec.rope_theta is not None:
            override_args["rope_theta"] = spec.rope_theta
        if override_args:
            command.extend(("--hf-overrides", json.dumps(override_args)))

        command.extend(spec.vllm_args)
        command.extend(self.vllm_args)
        return command

    async def _drain_and_stop(
        self,
        backend: Backend,
        generation: int,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        async with self._lock:
            while backend.generation == generation and backend.active_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                except TimeoutError:
                    break

        await self._stop_backend(backend)
        async with self._lock:
            if backend.generation == generation:
                backend.process = None
                backend.stop_task = None
                backend.start_task = None
                if backend.active_requests > 0:
                    backend.state = BackendState.FAILED
                    backend.error = f"backend stopped while {backend.active_requests} request(s) were still active"
                    backend.failed_at = time.monotonic()
                else:
                    backend.state = BackendState.STOPPED
                    self._ports.release(backend.spec.name, backend.port)
                self._changed.notify_all()

    async def _stop_backend(self, backend: Backend) -> None:
        process = backend.process
        if not process or process.returncode is not None:
            return
        LOGGER.info("stopping vLLM backend for %s", backend.spec.name)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self.shutdown_timeout)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    def _idle_threshold(self, spec: ModelSpec) -> float:
        if self.idle_seconds > 0:
            return self.idle_seconds
        return spec.idle_seconds or 0.0

    async def _idle_reaper(self) -> None:
        while True:
            thresholds = [threshold for spec in self.catalog.specs if (threshold := self._idle_threshold(spec)) > 0]
            minimum = min(thresholds) if thresholds else 60.0
            await asyncio.sleep(min(max(minimum / 2.0, 1.0), 60.0))

            async with self._lock:
                now = time.monotonic()
                tasks: list[asyncio.Task[None]] = []
                for backend in self._backends.values():
                    threshold = self._idle_threshold(backend.spec)
                    if (
                        backend.state is BackendState.READY
                        and backend.active_requests == 0
                        and threshold > 0
                        and now - backend.last_used >= threshold
                    ):
                        backend.state = BackendState.DRAINING
                        backend.stop_task = asyncio.create_task(
                            self._drain_and_stop(backend, backend.generation, self.shutdown_timeout),
                            name=f"vllm-idle-{backend.spec.name}",
                        )
                        tasks.append(backend.stop_task)
                if tasks:
                    self._changed.notify_all()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
