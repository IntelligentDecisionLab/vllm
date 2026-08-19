from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .catalog import ModelCatalog, ModelSpec, resolve_model_path

LOGGER = logging.getLogger("sglang_service")


@dataclass(slots=True)
class Backend:
    spec: ModelSpec
    port: int
    process: asyncio.subprocess.Process | None = None
    active_requests: int = 0
    last_used: float = 0.0


class BackendLease:
    def __init__(self, manager: "ModelManager", backend: Backend):
        self._manager = manager
        self.backend = backend
        self._released = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._manager.release(self.backend)


class ModelManager:
    """Own one SGLang subprocess at a time and route requests to it.

    One backend is deliberate: the configured GGUF weights are large enough
    that keeping all presets resident would defeat the available VRAM.  The
    active backend itself still handles concurrent requests through SGLang's
    scheduler.
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        backend_port: int = 30000,
        tensor_parallel_size: int = 2,
        data_parallel_size: int = 1,
        mem_fraction_static: float = 0.90,
        max_running_requests: int = 256,
        chunked_prefill_size: int = 4096,
        cuda_graph_backend_prefill: str = "tc_piecewise",
        attention_backend: str | None = "triton",
        startup_timeout: float = 900.0,
        idle_seconds: float = 0.0,
        context_length_override: int | None = None,
        sglang_args: tuple[str, ...] = (),
    ):
        if tensor_parallel_size < 1 or data_parallel_size < 1:
            raise ValueError("tensor_parallel_size and data_parallel_size must be at least 1")
        self.catalog = catalog
        self.backend_port = backend_port
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.chunked_prefill_size = chunked_prefill_size
        self.cuda_graph_backend_prefill = cuda_graph_backend_prefill
        self.attention_backend = attention_backend
        self.startup_timeout = startup_timeout
        self.idle_seconds = idle_seconds
        self.context_length_override = context_length_override
        self.sglang_args = sglang_args
        self._lock = asyncio.Lock()
        self._idle = asyncio.Condition(self._lock)
        self._current: Backend | None = None
        self._stopping = False
        self._idle_task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(
            # Inference and SSE responses may legitimately take much longer
            # than a health probe.  Readiness probes below set their own short
            # timeout explicitly.
            timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
        )

    async def start(self, preload: str | None = None) -> None:
        if self.idle_seconds > 0 or any(spec.idle_seconds and spec.idle_seconds > 0 for spec in self.catalog.specs):
            self._idle_task = asyncio.create_task(self._idle_reaper(), name="sglang-idle-reaper")
        if preload:
            lease = await self.acquire(preload)
            await lease.release()

    async def shutdown(self) -> None:
        self._stopping = True
        if self._idle_task:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)
        async with self._lock:
            backend = self._current
            self._current = None
        if backend:
            await self._stop_backend(backend)
        await self._client.aclose()

    async def acquire(self, model_name: str | None) -> BackendLease:
        spec = self.catalog.get(model_name)
        async with self._lock:
            if self._stopping:
                raise RuntimeError("model manager is shutting down")

            while self._current and self._current.spec.name != spec.name:
                if self._current.active_requests:
                    await self._idle.wait()
                    continue
                old = self._current
                self._current = None
                await self._stop_backend(old)

            if self._current is None:
                backend = Backend(spec=spec, port=self.backend_port, last_used=time.monotonic())
                self._current = backend
                try:
                    await self._start_backend(backend)
                except BaseException:
                    self._current = None
                    raise
            elif self._current.process is None or self._current.process.returncode is not None:
                backend = self._current
                await self._stop_backend(backend)
                try:
                    await self._start_backend(backend)
                except BaseException:
                    self._current = None
                    raise
            else:
                backend = self._current

            backend.active_requests += 1
            backend.last_used = time.monotonic()
            return BackendLease(self, backend)

    async def release(self, backend: Backend | None) -> None:
        if backend is None:
            return
        async with self._lock:
            backend.active_requests = max(0, backend.active_requests - 1)
            backend.last_used = time.monotonic()
            self._idle.notify_all()

    async def health(self) -> dict[str, Any]:
        async with self._lock:
            current = self._current
            loaded = current.spec.name if current and current.process and current.process.returncode is None else None
            active_requests = current.active_requests if current else 0
        return {
            "status": "ok",
            "service": "sglang-model-router",
            "loaded_model": loaded,
            "active_requests": active_requests,
            "models": [spec.name for spec in self.catalog.specs],
        }

    async def _start_backend(self, backend: Backend) -> None:
        model_path = await asyncio.to_thread(resolve_model_path, backend.spec)
        command = self._command(backend.spec, model_path, backend.port)
        LOGGER.info("starting SGLang backend for %s: %s", backend.spec.name, " ".join(command))
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
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
                    f"SGLang backend for {backend.spec.name!r} exited with "
                    f"code {backend.process.returncode}; inspect its stderr"
                )
            for endpoint in ("/health", "/v1/models"):
                try:
                    response = await self._client.get(
                        f"http://127.0.0.1:{backend.port}{endpoint}",
                        timeout=httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0),
                    )
                    if response.status_code < 500:
                        LOGGER.info("SGLang backend ready: %s", backend.spec.name)
                        return
                    last_error = f"HTTP {response.status_code} from {endpoint}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
            await asyncio.sleep(1.0)

        await self._stop_backend(backend)
        raise TimeoutError(
            f"timed out after {self.startup_timeout:.0f}s waiting for SGLang backend "
            f"{backend.spec.name!r} ({last_error})"
        )

    def _command(self, spec: ModelSpec, model_path: str, port: int) -> list[str]:
        context_length = self.context_length_override or spec.context_length
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            spec.name,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--data-parallel-size",
            str(self.data_parallel_size),
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--max-running-requests",
            str(self.max_running_requests),
            "--chunked-prefill-size",
            str(self.chunked_prefill_size),
            "--cuda-graph-backend-prefill",
            self.cuda_graph_backend_prefill,
        ]
        if spec.is_gguf:
            command.extend(("--load-format", "gguf"))
        if spec.embeddings:
            command.append("--is-embedding")
        if spec.kv_cache_dtype:
            command.extend(("--kv-cache-dtype", spec.kv_cache_dtype))
        if context_length:
            command.extend(("--context-length", str(context_length)))
        if self.attention_backend:
            command.extend(("--attention-backend", self.attention_backend))

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
            command.extend(("--json-model-override-args", json.dumps(override_args)))

        command.extend(self.sglang_args)
        return command

    async def _stop_backend(self, backend: Backend) -> None:
        process = backend.process
        backend.process = None
        if not process or process.returncode is not None:
            return
        LOGGER.info("stopping SGLang backend for %s", backend.spec.name)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def _idle_reaper(self) -> None:
        while True:
            await asyncio.sleep(min(max(self.idle_seconds / 2, 10.0), 60.0))
            async with self._lock:
                backend = self._current
                if backend and backend.active_requests == 0:
                    threshold = backend.spec.idle_seconds if backend.spec.idle_seconds is not None else self.idle_seconds
                    if threshold > 0 and time.monotonic() - backend.last_used >= threshold:
                        self._current = None
                        await self._stop_backend(backend)
                        self._idle.notify_all()
