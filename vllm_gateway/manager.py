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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

import httpx

from .catalog import ModelCatalog, ModelSpec
from .diagnostics import BackendDiagnostic, BackendOutput, BackendStartupError, diagnostic_from_exception

LOGGER = logging.getLogger("vllm_gateway")

IDLE_SECONDS = 3600.0
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 2048
KV_CACHE_DTYPE = "fp8_e4m3"
ATTENTION_BACKEND = "flashinfer"
NUM_SPECULATIVE_TOKENS = 2


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


def diagnose_backend_error(exc: BaseException) -> BackendDiagnostic:
    code = None
    if isinstance(exc, CapacityError):
        code = "backend_capacity_exceeded"
    elif isinstance(exc, PortUnavailable):
        code = "backend_port_unavailable"
    return diagnostic_from_exception(exc, code=code)


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
    diagnostic: BackendDiagnostic | None = None
    diagnostic_id: str = field(default_factory=lambda: uuid4().hex)
    output: BackendOutput = field(default_factory=BackendOutput)
    output_task: asyncio.Task[None] | None = None


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

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        backend_port: int = 30000,
        backend_port_count: int = 64,
        max_loaded_models: int = 2,
        startup_timeout: float = 900.0,
        shutdown_timeout: float = 30.0,
        idle_seconds: float = IDLE_SECONDS,
        failure_cooldown: float = 2.0,
    ):
        if max_loaded_models < 1:
            raise ValueError("max_loaded_models must be at least 1")

        self.catalog = catalog
        self.backend_port = backend_port
        self.backend_port_count = backend_port_count
        self.max_loaded_models = max_loaded_models
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.idle_seconds = idle_seconds
        self.failure_cooldown = failure_cooldown

        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._backends: dict[str, Backend] = {}
        self._ports = PortAllocator(backend_port, backend_port_count)
        self._stopping = False
        self._started = False
        self._idle_task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0))

    async def start(self) -> None:
        self._started = True
        if self.idle_seconds > 0:
            self._idle_task = asyncio.create_task(self._idle_reaper(), name="vllm-idle-reaper")

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
                        message = (
                            f"model {spec.name!r} backend failed while "
                            f"{backend.active_requests} request(s) are still releasing"
                        )
                        if backend.diagnostic is not None:
                            raise BackendStartupError(message, backend.diagnostic)
                        raise RuntimeError(message)
                    if (
                        backend
                        and backend.state is BackendState.FAILED
                        and backend.failed_at is not None
                        and time.monotonic() - backend.failed_at < self.failure_cooldown
                    ):
                        message = f"model {spec.name!r} is in failure cooldown: {backend.error or 'startup failed'}"
                        if backend.diagnostic is not None:
                            raise BackendStartupError(message, backend.diagnostic)
                        raise RuntimeError(message)

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
                            backend.spec = spec
                            backend.port = port
                        backend.generation += 1
                        backend.state = BackendState.LOADING
                        backend.error = None
                        backend.failed_at = None
                        backend.diagnostic = None
                        backend.diagnostic_id = uuid4().hex
                        backend.output = BackendOutput()
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
            "loaded_model": ready[0] if ready else None,
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
                "diagnostic": None,
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
            "diagnostic": backend.diagnostic.as_dict() if backend.diagnostic else None,
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
            diagnostic = None
            if not isinstance(exc, asyncio.CancelledError):
                diagnostic = diagnostic_from_exception(
                    exc,
                    code=backend.output.code or diagnose_backend_error(exc).code,
                    diagnostic_id=backend.diagnostic_id,
                )
                LOGGER.error(
                    "backend failure model=%s generation=%d diagnostic_id=%s code=%s: %s",
                    backend.spec.name,
                    generation,
                    diagnostic.diagnostic_id,
                    diagnostic.code,
                    exc,
                )
            async with self._lock:
                if backend.generation == generation:
                    backend.start_task = None
                    backend.process = None
                    backend.error = None if isinstance(exc, asyncio.CancelledError) else str(exc)
                    backend.diagnostic = diagnostic
                    backend.failed_at = None if isinstance(exc, asyncio.CancelledError) else time.monotonic()
                    backend.state = (
                        BackendState.STOPPED
                        if self._stopping or backend.state is BackendState.DRAINING
                        else BackendState.FAILED
                    )
                    self._ports.release(backend.spec.name, backend.port)
                    self._changed.notify_all()
            if diagnostic is not None:
                raise BackendStartupError(str(exc), diagnostic) from exc
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
        returncode = await self._wait_for_exit(process)
        if backend.generation != generation or backend.process is not process:
            return
        await self._finish_output(backend, process)
        async with self._lock:
            if backend.generation != generation or backend.process is not process:
                return
            backend.process = None
            if backend.active_requests == 0:
                self._ports.release(backend.spec.name, backend.port)
            if backend.state not in {BackendState.DRAINING, BackendState.STOPPED}:
                backend.state = BackendState.FAILED
                backend.error = f"vLLM backend exited with code {returncode}"
                backend.diagnostic = diagnostic_from_exception(
                    RuntimeError(backend.error),
                    code=backend.output.code or "backend_exited",
                    diagnostic_id=backend.diagnostic_id,
                )
                LOGGER.error(
                    "backend failure model=%s generation=%d diagnostic_id=%s code=%s: %s",
                    backend.spec.name,
                    generation,
                    backend.diagnostic.diagnostic_id,
                    backend.diagnostic.code,
                    backend.error,
                )
                backend.failed_at = time.monotonic()
            self._changed.notify_all()

    async def _start_backend(self, backend: Backend) -> None:
        if not PortAllocator._is_available(backend.port):
            raise PortUnavailable(f"backend port {backend.port} for model {backend.spec.name!r} became unavailable")

        if not await asyncio.to_thread(os.path.isdir, backend.spec.model_path):
            raise FileNotFoundError(f"model files for {backend.spec.name!r} do not exist: {backend.spec.model_path}")
        command = self._command(backend.spec, backend.port)
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        reader = getattr(backend.process, "stdout", None)
        if reader is not None:
            backend.output_task = asyncio.create_task(
                self._read_output(reader, backend.output, backend.spec.name, backend.generation),
                name=f"vllm-output-{backend.spec.name}-{backend.generation}",
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

    def _command(self, spec: ModelSpec, port: int) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            spec.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            spec.name,
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--max-num-batched-tokens",
            str(MAX_NUM_BATCHED_TOKENS),
            "--kv-cache-dtype",
            KV_CACHE_DTYPE,
            "--max-model-len",
            str(spec.context_length),
            "--attention-backend",
            ATTENTION_BACKEND,
            "--async-scheduling",
            "--enable-flashinfer-autotune",
            "--enable-prefix-caching",
        ]
        if spec.reasoning_parser:
            command.extend(("--reasoning-parser", spec.reasoning_parser))
        if spec.tool_call_parser:
            command.extend(("--enable-auto-tool-choice", "--tool-call-parser", spec.tool_call_parser))
        if spec.mtp:
            speculative_config = {"method": "mtp", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS}
            command.extend(("--speculative-config", json.dumps(speculative_config)))
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

    @staticmethod
    async def _read_output(
        reader: asyncio.StreamReader,
        output: BackendOutput,
        name: str,
        generation: int,
    ) -> None:
        try:
            while chunk := await reader.read(4096):
                output.feed(chunk)
                LOGGER.info(
                    "backend output model=%s generation=%d: %s",
                    name,
                    generation,
                    chunk.decode("utf-8", errors="replace"),
                )
        except (OSError, ValueError) as exc:
            LOGGER.warning("backend output reader failed for %s: %s", name, exc)

    @staticmethod
    async def _wait_for_exit(process: asyncio.subprocess.Process) -> int:
        # Process.wait() can wait for inherited pipes even after the child exits.
        waiter = asyncio.create_task(process.wait())
        try:
            while process.returncode is None:
                done, _ = await asyncio.wait({waiter}, timeout=0.1)
                if done:
                    return waiter.result()
            return process.returncode
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    @staticmethod
    async def _finish_output(backend: Backend, process: asyncio.subprocess.Process | None) -> None:
        task = backend.output_task
        if task is None:
            return
        done, _ = await asyncio.wait({task}, timeout=0.5)
        if not done:
            task.cancel()
            # StreamReader has no public close API. Close only our own pipe's
            # transport when a descendant keeps stdout open past the deadline.
            reader = getattr(process, "stdout", None)
            transport = getattr(reader, "_transport", None)
            if transport is not None:
                transport.close()
        await asyncio.gather(task, return_exceptions=True)
        if backend.output_task is task:
            backend.output_task = None

    async def _stop_backend(self, backend: Backend) -> None:
        process = backend.process
        try:
            if not process or process.returncode is not None:
                return
            LOGGER.info("stopping vLLM backend for %s", backend.spec.name)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(self._wait_for_exit(process), timeout=self.shutdown_timeout)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(self._wait_for_exit(process), timeout=self.shutdown_timeout)
        finally:
            await self._finish_output(backend, process)

    async def _idle_reaper(self) -> None:
        while True:
            await asyncio.sleep(min(max(self.idle_seconds / 2.0, 1.0), 60.0))

            async with self._lock:
                now = time.monotonic()
                tasks: list[asyncio.Task[None]] = []
                for backend in self._backends.values():
                    if (
                        backend.state is BackendState.READY
                        and backend.active_requests == 0
                        and now - backend.last_used >= self.idle_seconds
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
