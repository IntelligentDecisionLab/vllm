from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from .memory import GIB

LOGGER = logging.getLogger("vllm_gateway")

ALLOW_PATTERNS = ("*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken")
RESERVE_BYTES = 20 * GIB  # disk space that must stay free after a download
MAX_FINISHED_JOBS = 50
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
_REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")


class DownloadState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DownloadError(Exception):
    """A download request that cannot be accepted. `status_code` is the HTTP status to report."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(slots=True)
class DownloadJob:
    id: str
    repo_id: str
    revision: str | None
    size_bytes: int
    state: DownloadState = DownloadState.QUEUED
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "state": self.state.value,
            "size_bytes": self.size_bytes,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


def hub_remote_files(repo_id: str, revision: str | None) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError as exc:
        raise DownloadError(403, "repository is gated; the gateway needs an HF_TOKEN with access") from exc
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise DownloadError(404, f"repository or revision not found: {repo_id}") from exc
    except (HfHubHTTPError, OSError) as exc:
        raise DownloadError(502, f"cannot reach Hugging Face: {type(exc).__name__}") from exc
    return [(item.rfilename, item.size or 0) for item in info.siblings or []]


def hub_fetch(repo_id: str, revision: str | None, cache_dir: Path, allow_patterns: tuple[str, ...]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=list(allow_patterns))


class DownloadManager:
    """Background Hugging Face downloads into the shared cache, one at a time."""

    def __init__(
        self,
        hub_cache: Path,
        *,
        reserve_bytes: int = RESERVE_BYTES,
        remote_files: Callable[[str, str | None], list[tuple[str, int]]] = hub_remote_files,
        fetch: Callable[[str, str | None, Path, tuple[str, ...]], None] = hub_fetch,
        disk_free: Callable[[Path], int] | None = None,
    ):
        self.hub_cache = hub_cache
        self.reserve_bytes = reserve_bytes
        self._remote_files = remote_files
        self._fetch = fetch
        self._disk_free = disk_free or self._default_disk_free
        self._jobs: dict[str, DownloadJob] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._slot = asyncio.Semaphore(1)

    @staticmethod
    def _default_disk_free(path: Path) -> int:
        while not path.exists() and path != path.parent:
            path = path.parent
        return shutil.disk_usage(path).free

    def _active(self, repo_id: str, revision: str | None) -> DownloadJob | None:
        for job in self._jobs.values():
            if (
                job.repo_id == repo_id
                and job.revision == revision
                and job.state in {DownloadState.QUEUED, DownloadState.RUNNING}
            ):
                return job
        return None

    def active_repos(self) -> set[str]:
        return {
            job.repo_id for job in self._jobs.values() if job.state in {DownloadState.QUEUED, DownloadState.RUNNING}
        }

    def get(self, job_id: str) -> DownloadJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown download job {job_id!r}")
        return job

    def jobs(self) -> list[DownloadJob]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    async def submit(self, repo_id: object, revision: object = None) -> DownloadJob:
        if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id):
            raise DownloadError(400, "repo_id must look like 'org/name'")
        if revision is not None and (not isinstance(revision, str) or not _REVISION.fullmatch(revision)):
            raise DownloadError(400, "revision is invalid")
        existing = self._active(repo_id, revision)
        if existing is not None:
            return existing

        files = await asyncio.to_thread(self._remote_files, repo_id, revision)
        wanted = [(name, size) for name, size in files if any(fnmatch.fnmatch(name, p) for p in ALLOW_PATTERNS)]
        if not any(name.endswith(".safetensors") for name, _ in wanted):
            raise DownloadError(400, "repository has no safetensors weights")
        size = sum(size for _, size in wanted)
        free = await asyncio.to_thread(self._disk_free, self.hub_cache)
        if size > free - self.reserve_bytes:
            raise DownloadError(
                507,
                f"download needs {size / GIB:.1f} GiB but only {free / GIB:.1f} GiB is free "
                f"(keeping {self.reserve_bytes / GIB:.0f} GiB in reserve)",
            )

        existing = self._active(repo_id, revision)  # another request may have registered while we waited
        if existing is not None:
            return existing
        job = DownloadJob(uuid4().hex, repo_id, revision, size, created_at=time.time())
        self._jobs[job.id] = job
        self._prune()
        task = asyncio.create_task(self._run(job), name=f"download-{repo_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _prune(self) -> None:
        finished = [job for job in self.jobs() if job.state in {DownloadState.SUCCEEDED, DownloadState.FAILED}]
        for job in finished[MAX_FINISHED_JOBS:]:
            del self._jobs[job.id]

    async def _run(self, job: DownloadJob) -> None:
        async with self._slot:
            job.state = DownloadState.RUNNING
            try:
                await asyncio.to_thread(self._fetch, job.repo_id, job.revision, self.hub_cache, ALLOW_PATTERNS)
            except Exception as exc:
                LOGGER.exception("download of %s failed", job.repo_id)
                job.state = DownloadState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"[:500]
            else:
                job.state = DownloadState.SUCCEEDED
            finally:
                job.finished_at = time.time()

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
