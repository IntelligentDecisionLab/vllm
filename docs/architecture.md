# Gateway architecture

## System boundary

This repository implements an OpenAI-compatible gateway. The gateway parses the catalog, starts and stops independent vLLM API server subprocesses, forwards HTTP requests, and tracks a request lease for each backend. Model execution, the scheduler, CUDA kernels, and weight formats remain the responsibility of upstream vLLM.

The companion script is in `scripts/vllm` in the repository and is installed to `~/.local/bin/vllm` for deployment. It manages the gateway PID and log; the gateway's `ModelManager` owns the backend process group. Tests execute the repository copy of the script to ensure that it does not scan for or terminate processes that do not belong to the gateway.

## Components

- `vllm_gateway/cli.py`: parses CLI arguments and environment variables, loads the catalog, creates `ModelManager` and the FastAPI app, and starts Uvicorn.
- `vllm_gateway/catalog.py`: parses TOML into immutable `ModelSpec` objects and handles defaults, aliases, name normalization, and model paths.
- `vllm_gateway/manager.py`: allocates private ports, creates vLLM subprocesses, runs readiness probes, and manages capacity, leases, LRU/idle eviction, reload, and shutdown.
- `vllm_gateway/app.py`: provides health, model listing, localhost-only management APIs, and a catch-all proxy; it canonicalizes model identity before forwarding.
- `models.toml`: the model catalog and deployment parameters currently used by the repository.

`python -m vllm_gateway`, `vllm-service`, and `vllm-gateway` all ultimately enter `vllm_gateway.cli:main`.

## Request flow

1. `app.py` reads `model` from the JSON body. If it is not specified, the current default model is used.
2. `ModelCatalog.get()` accepts a canonical name, normalized name, alias, model path, or Hugging Face source and returns the unique `ModelSpec`.
3. The app changes the outgoing `model` to the canonical name, applies request-level model defaults, and obtains a `BackendLease` from the manager.
4. `ModelManager.acquire()` shares the startup task already running for the same model. If capacity is full, it unloads only the least-recently-used `READY` backend with no active requests.
5. The manager forwards the request stream to `127.0.0.1:<private-port>`. The app must release the lease after a normal response is fully read, an SSE stream closes, or the proxy raises an exception.

Leases tell shutdown, unload, reload, and idle eviction when it is safe to stop a backend. When modifying a proxy error path, preserve the invariant that every successful acquire is released exactly once.

## Backend lifecycle

Backend states are:

```text
STOPPED -> LOADING -> READY -> DRAINING -> STOPPED
                    \-> FAILED
```

Important invariants:

- A private port is mapped deterministically from the canonical model name; collisions are probed within the bounded range.
- If an external process occupies a port, the gateway skips it or reports an error and does not take ownership of that process.
- A backend must pass both `/health` and a `/v1/models` check containing the expected canonical model before it can enter `READY`.
- Concurrent first requests for the same model share one startup task; cancelling a waiter must not cancel the shared startup.
- A `DRAINING` backend accepts no new leases. The manager waits for active leases to reach zero and stops the process only after the timeout.
- The gateway starts the backend in a new process session and, on stop, sends SIGTERM only to the process group it created, followed by SIGKILL after the timeout.

## Catalog reload

`POST /admin/reload` reparses the configuration path used at startup. If parsing fails, the app retains the old catalog and returns 400. After a successful parse:

- New model: added to the catalog and loaded on the first request.
- Removed model: its resident backend is stopped after startup cancellation or draining.
- Changed `ModelSpec`: the old backend is stopped; the next request starts it with the new spec.
- Identical spec: the current backend process is retained.

If the CLI specifies `--default-model`, reload does not change the request default; otherwise, the app follows the new catalog default.

## Gateway-managed vLLM options

`ModelManager._command()` consistently generates identity, network, parallelism, memory, context, KV cache, parser, chat template, attention backend, and Hugging Face override arguments. These options cannot be hidden in `vllm_args` or the global `--vllm-arg`.

This restriction once directly caused a model load failure: the catalog put `--attention-backend` in `vllm_args`, and the manager refused to start because it was a protected option. `attention_backend` is now an official `ModelSpec` field; when adding other managed options, create a typed field, CLI override, and command-generation test rather than bypassing protected-flag validation.
