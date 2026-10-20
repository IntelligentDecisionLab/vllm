from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from vllm_gateway.downloads import (
    ALLOW_PATTERNS,
    DownloadError,
    DownloadManager,
    DownloadState,
)
from vllm_gateway.memory import GIB

pytestmark = pytest.mark.anyio

FILES = [("model.safetensors", 4 * GIB), ("config.json", 100), ("tokenizer.json", 900), ("pytorch_model.bin", 10**12)]


def make_manager(tmp_path: Path, *, free: int = 100 * GIB, fetch=None, files=FILES) -> DownloadManager:
    return DownloadManager(
        tmp_path,
        remote_files=lambda repo, revision: list(files),
        fetch=fetch or (lambda repo, revision, cache, patterns: None),
        disk_free=lambda path: free,
    )


async def wait_for_state(manager: DownloadManager, job_id: str, state: DownloadState) -> None:
    for _ in range(200):
        if manager.get(job_id).state is state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job stayed {manager.get(job_id).state}")


async def test_size_counts_only_files_that_will_be_downloaded(tmp_path: Path) -> None:
    calls: list[tuple] = []
    manager = make_manager(tmp_path, fetch=lambda *args: calls.append(args))

    job = await manager.submit("Qwen/Qwen3-0.6B", "main")
    await wait_for_state(manager, job.id, DownloadState.SUCCEEDED)

    assert job.size_bytes == 4 * GIB + 1000
    assert calls == [("Qwen/Qwen3-0.6B", "main", tmp_path, ALLOW_PATTERNS)]
    assert job.finished_at is not None
    assert manager.active_repos() == set()


async def test_same_repo_is_deduplicated_while_active(tmp_path: Path) -> None:
    release = threading.Event()
    manager = make_manager(tmp_path, fetch=lambda *args: release.wait(timeout=5))

    first = await manager.submit("org/model")
    second = await manager.submit("org/model")
    await asyncio.sleep(0.05)

    assert second is first
    assert manager.active_repos() == {"org/model"}
    release.set()
    await wait_for_state(manager, first.id, DownloadState.SUCCEEDED)
    third = await manager.submit("org/model")
    assert third is not first
    await wait_for_state(manager, third.id, DownloadState.SUCCEEDED)


async def test_downloads_run_one_at_a_time(tmp_path: Path) -> None:
    release = threading.Event()
    order: list[str] = []

    def fetch(repo, revision, cache, patterns) -> None:
        order.append(repo)
        release.wait(timeout=5)

    manager = make_manager(tmp_path, fetch=fetch)
    first = await manager.submit("org/one")
    second = await manager.submit("org/two")
    await wait_for_state(manager, first.id, DownloadState.RUNNING)
    await asyncio.sleep(0.05)

    assert second.state is DownloadState.QUEUED
    assert order == ["org/one"]
    release.set()
    await wait_for_state(manager, second.id, DownloadState.SUCCEEDED)
    assert order == ["org/one", "org/two"]


async def test_failed_download_reports_the_error(tmp_path: Path) -> None:
    def fetch(*args) -> None:
        raise OSError("disk exploded")

    manager = make_manager(tmp_path, fetch=fetch)
    job = await manager.submit("org/model")
    await wait_for_state(manager, job.id, DownloadState.FAILED)

    assert job.error == "OSError: disk exploded"
    assert manager.active_repos() == set()


async def test_rejects_downloads_that_would_exhaust_the_disk(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, free=23 * GIB)  # 4 GiB needed + 20 GiB reserve > 23 GiB

    with pytest.raises(DownloadError) as error:
        await manager.submit("org/model")

    assert error.value.status_code == 507
    assert "GiB" in error.value.message
    assert manager.jobs() == []


@pytest.mark.parametrize("repo_id", ["", "no-slash", "a/b/c", "../etc/passwd", "org/na me", None, 7])
async def test_rejects_malformed_repo_ids(tmp_path: Path, repo_id: object) -> None:
    with pytest.raises(DownloadError) as error:
        await make_manager(tmp_path).submit(repo_id)
    assert error.value.status_code == 400


async def test_rejects_repos_without_safetensors_and_bad_revisions(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="no safetensors"):
        await make_manager(tmp_path, files=[("pytorch_model.bin", 10), ("config.json", 1)]).submit("org/model")
    with pytest.raises(DownloadError, match="revision"):
        await make_manager(tmp_path).submit("org/model", "bad revision!")


async def test_finished_jobs_are_pruned(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    ids = []
    for index in range(55):
        job = await manager.submit(f"org/m{index}")
        await wait_for_state(manager, job.id, DownloadState.SUCCEEDED)
        ids.append(job.id)
    await manager.submit("org/last")

    assert len(manager.jobs()) <= 52
    with pytest.raises(KeyError):
        manager.get(ids[0])
