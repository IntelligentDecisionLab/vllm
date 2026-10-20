# Gateway architecture

## System boundary

This repository implements an OpenAI-compatible gateway. It discovers models in the Hugging Face cache, stores user aliases, starts and stops independent vLLM API server subprocesses, forwards HTTP requests, and tracks a request lease for each backend. Model execution, the scheduler, CUDA kernels and weight formats remain the responsibility of upstream vLLM.

The gateway runs in a Docker container that can see exactly one GPU. The gateway starts vLLM with `sys.executable`, so it is installed into the vLLM image's own Python.

## Components

- `vllm_gateway/cli.py`: parses CLI arguments and environment variables, builds the catalog, manager and app, and starts Uvicorn.
- `vllm_gateway/catalog.py`: scans the Hugging Face cache, reads `config.json` to derive the family profile, KV size and MTP support, and owns the persisted alias registry (`ModelSpec`, `ModelCatalog`).
- `vllm_gateway/memory.py`: the memory estimate for one model and context length, and the GPU memory query.
- `vllm_gateway/manager.py`: allocates private ports, creates vLLM subprocesses, runs readiness probes, and manages admission, leases, eviction, idle unload and shutdown.
- `vllm_gateway/downloads.py`: background Hugging Face downloads with a disk-space check.
- `vllm_gateway/diagnostics.py`: classification of backend failures into safe codes with a diagnostic ID.
- `vllm_gateway/app.py`: HTTP routes for models, aliases, downloads and health, plus a catch-all proxy.

`python -m vllm_gateway`, `vllm-service` and `vllm-gateway` all enter `vllm_gateway.cli:main`.

## Request flow

1. `app.py` reads `model` from the JSON body. A missing or non-string `model` returns 400.
2. `ModelCatalog.get()` accepts only an exact alias and returns its `ModelSpec`. An unknown name returns 404.
3. The app rewrites `model` to the alias (the name vLLM serves) and obtains a `BackendLease` from `ModelManager.acquire()`.
4. `acquire()` returns a lease for a ready backend, shares the startup task of a loading one, or admits and starts a new one.
5. The manager forwards the request to `127.0.0.1:<private-port>`. The app releases the lease after a normal response is fully read, an SSE stream closes, or the proxy raises an exception.

Leases tell shutdown, unload, delete and eviction when it is safe to stop a backend. Every successful acquire must be released exactly once.

## Backend lifecycle

```text
STOPPED -> LOADING -> READY -> DRAINING -> STOPPED
                    \-> FAILED
```

Invariants:

- I1 Every successful acquire is released exactly once on every path (normal, SSE, HTTP error, cancellation, exception).
- I2 Eviction only picks a backend that is `READY` with zero active leases.
- I3 Signals go only to the process group the manager created. Other processes on the GPU are never touched.
- I4 Concurrent first requests for one alias share a single startup. Cancelling a waiter does not cancel it.
- I5 `ModelManager._lock` is never held while waiting for a subprocess, HTTP, or an NVML call.
- I6 At most one backend is `LOADING`. The startup gate is taken before the free-memory measurement and released by the start task when it finishes or is cancelled, so the next measurement sees the memory the previous model really uses.
- A private port is derived from the alias, with deterministic probing inside a bounded range. A port held by a foreign process is skipped.
- A backend becomes `READY` only when `/health` and `/v1/models` (which must list the alias) both succeed.
- A `DRAINING` backend accepts no new leases. The manager waits for active leases, then stops the process group with SIGTERM and SIGKILL after the timeout.

## Admission and eviction

Before starting a model, `_make_room()` measures the free memory of the whole GPU, which includes other users, and compares it with the estimate plus a safety margin (see [configuration](configuration.md#memory-estimate)). While it does not fit, it drains the least recently used idle backend, waits for the memory to be released and measures again. When no idle backend is left it raises `CapacityError`, which the API reports as 503 `insufficient_gpu_memory`.

`validate_fits()` rejects, at alias creation, a model that would not fit even on an empty GPU.

## Aliases

`ModelCatalog` keeps aliases in `aliases.json` under the state directory, written through a temporary file and `os.replace`. At startup each alias is resolved again against the cache. An alias whose model has disappeared stays in the file and is reported as unavailable instead of being deleted.

An alias stores only the repo id and context length. The snapshot path, weight size, KV size, parsers and MTP flag are derived from the cache each time, so a re-download is picked up on the next restart.

## Fixed backend options

`ModelManager._command()` builds the same command for every model, apart from the alias, snapshot path, context length, memory plan and the family-dependent flags (reasoning parser, tool-call parser, speculative config). Users cannot pass other vLLM options. To make an option configurable, add it to `ModelSpec`, the alias API, `_command()` and its tests.
