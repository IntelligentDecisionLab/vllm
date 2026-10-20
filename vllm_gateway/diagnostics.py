from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from uuid import uuid4

_MESSAGES = {
    "kv_cache_memory_insufficient": "Not enough GPU memory to allocate KV cache blocks.",
    "cuda_out_of_memory": "The backend ran out of GPU memory.",
    "backend_startup_timeout": "The backend did not become ready before the startup deadline.",
    "backend_startup_failed": "The backend failed to start; consult the gateway log with the diagnostic ID.",
    "backend_exited": "The backend process exited unexpectedly.",
    "insufficient_gpu_memory": "Not enough free GPU memory to load the model.",
    "backend_port_unavailable": "No backend port is available.",
    "model_files_unavailable": "Required model files are unavailable.",
    "invalid_backend_configuration": "The backend configuration is invalid.",
}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True, slots=True)
class BackendDiagnostic:
    code: str
    message: str
    diagnostic_id: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class BackendStartupError(RuntimeError):
    def __init__(self, message: str, diagnostic: BackendDiagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


class BackendOutput:
    """Bounded raw tail; recognized failure evidence survives tail eviction."""

    LIMIT = 65536

    def __init__(self) -> None:
        self.tail = b""
        self.code: str | None = None

    def feed(self, chunk: bytes) -> None:
        # Check before eviction so a match at a chunk boundary is retained.
        text = _ANSI.sub("", (self.tail + chunk).decode("utf-8", errors="replace")).lower()
        if "cuda out of memory" in text or "torch.outofmemoryerror" in text:
            self.code = "cuda_out_of_memory"
        elif (
            "no available memory for the cache blocks" in text or "larger than the available kv cache memory" in text
        ) and self.code is None:
            self.code = "kv_cache_memory_insufficient"
        self.tail = (self.tail + chunk)[-self.LIMIT :]


def diagnostic_from_exception(
    exc: BaseException,
    *,
    code: str | None = None,
    diagnostic_id: str | None = None,
    message: str | None = None,
) -> BackendDiagnostic:
    if isinstance(exc, BackendStartupError):
        return exc.diagnostic
    if code is None:
        if isinstance(exc, TimeoutError):
            code = "backend_startup_timeout"
        elif isinstance(exc, FileNotFoundError):
            code = "model_files_unavailable"
        elif isinstance(exc, ValueError):
            code = "invalid_backend_configuration"
        else:
            code = "backend_startup_failed"
    return BackendDiagnostic(code, message or _MESSAGES[code], diagnostic_id or uuid4().hex)
