# Development and verification

## Environment

The project requires Python 3.12 or newer and uses uv to manage the lock file. The unit tests use FastAPI, httpx and fake processes, so they need neither a GPU nor vLLM. vLLM is an optional runtime dependency (`--extra runtime`) because the Docker image already contains it.

```bash
uv sync --extra test --extra dev
```

## Entry points

```bash
uv run python -m vllm_gateway --help
```

The `vllm-service` and `vllm-gateway` scripts in `pyproject.toml` call the same main function. `docker compose up -d --build` runs the same module inside the image.

## Test areas

- `tests/test_catalog.py`: cache scan, `config.json` parsing (dense, hybrid, MTP), family profiles, alias validation and persistence.
- `tests/test_memory.py`: the memory estimate and the `nvidia-smi` fallback parser.
- `tests/test_downloads.py`: download jobs, deduplication, disk check and pruning, with fake Hub functions.
- `tests/test_app.py`: HTTP routing, model/alias/download endpoints, stream lease cleanup, diagnostics. Uses `FakeManager`.
- `tests/test_manager.py`: asynchronous lifecycle, startup serialization, memory admission and idle unload with a fake GPU, draining, crashes, ports.
- `tests/test_diagnostics.py`: failure classification.

Narrow the loop according to the change:

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py tests/test_memory.py tests/test_downloads.py
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
```

Full verification:

```bash
uv run --extra test python -m pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

GitHub Actions runs the same three commands on pushes to `main` and on pull requests, on a CPU-only runner. Install the hooks with `uv tool install pre-commit && pre-commit install`. The repository has no type-check gate.

## Change workflow

### Backend option or alias field

1. Add the field to `ModelSpec` and to `ModelCatalog` (resolution and the persisted JSON).
2. Expose it in the alias API in `app.py` if users should choose it.
3. Emit the flag in `ModelManager._command()` and cover it in `tests/test_manager.py`.
4. If it changes memory use, update `plan_memory()` and its tests.

### Proxy or lifecycle

- Write down which of the invariants in [architecture](architecture.md#backend-lifecycle) the change touches before editing, and add a test for each.
- Every successful acquire path in the proxy must release its lease, including HTTP errors, SSE close, client cancellation and unexpected exceptions.
- Manager tests use `FakeProcess`, `asyncio.Event`, `FakeGpu` and monkeypatching to control startup and stop without vLLM.
- Do not infer ownership with `fuser`, `pgrep`, port regexes or a machine-wide process scan.

## Backend failures

When a load fails, the API returns 503 with `error.code`, a safe `message` and a `diagnostic_id`. The `diagnostic_id` is also logged with the failure and the backend's merged stdout/stderr, which the gateway writes to its own log and keeps (at most 64 KiB) in memory:

```bash
docker compose logs gateway | grep -F '<diagnostic_id>'
```

| `error.code` | Meaning |
| --- | --- |
| `insufficient_gpu_memory` | Not enough free GPU memory. Loaded backends are not stopped to make room. The message gives GiB needed and free |
| `kv_cache_memory_insufficient` | vLLM could not allocate KV cache blocks |
| `cuda_out_of_memory` | Explicit CUDA/PyTorch OOM |
| `backend_startup_timeout` | Readiness was not reached before the startup deadline |
| `backend_startup_failed` | Unrecognized startup failure. No cause is inferred |
| `backend_exited` | A started backend exited unexpectedly |
| `backend_port_unavailable` | No backend port is available |
| `model_files_unavailable` | The snapshot directory no longer exists |
| `invalid_backend_configuration` | The backend configuration is invalid |

`GET /aliases` and `GET /health` keep the last failure of each backend. The same ID is reused during the retry cooldown and a new attempt gets a new ID. Transport errors between the gateway and a backend return 502 with `backend_request_failed`. Bodies and SSE streams from a backend are forwarded unchanged.

## Deploying and the GPU smoke test

Docker is the only deployment path. On the server:

```bash
git clone -b <branch> https://github.com/IntelligentDecisionLab/vllm.git && cd vllm
cp .env.example .env     # set GATEWAY_UID, GATEWAY_GID, GATEWAY_GPU
docker compose up -d --build
```

Changes to command generation, memory estimation or the image need this smoke test on the target GPU. The GPU may be shared, so record its memory and other users' processes first and compare at the end.

1. `nvidia-smi` baseline for the gateway's GPU.
2. `GET /health` returns 200 and shows the GPU's free and total memory.
3. Download a small dense model and a hybrid one (for example `Qwen/Qwen3-0.6B` and `Qwen/Qwen3.5-9B`) and wait for the jobs to succeed. Confirm that the new files are writable by the cache's group.
4. Create an alias for the hybrid model and load it. In the log, `reserved … GiB memory for KV Cache` should match the estimate and the measured GPU use should stay below it. Send one non-streaming and one streaming chat completion, and one long prompt with several concurrent requests.
5. Create more aliases than fit. Loading the last one must fail with 503 `insufficient_gpu_memory` and leave every loaded backend `ready`, and it must succeed after one of them is unloaded. An alias that cannot fit an empty GPU must be rejected with 400 at creation.
6. `docker compose restart` keeps the aliases. `docker compose down` returns quickly, and the GPU memory returns to the baseline.
7. Inside the image, `nvcc --version` must be CUDA 12.9 or newer for Blackwell (SM 12.x) FlashInfer JIT. To debug a backend by hand, set `CUDA_HOME=/usr/local/cuda` and put `/usr/local/cuda/bin` on `PATH`.
