# vLLM multi-model gateway

This project provides a single OpenAI-compatible endpoint (default `:11432`) and starts multiple independent vLLM backends on demand from the repository's `models.toml`. Each resident model has its own private port, process, request lease, and idle policy; by default, up to two models can remain resident at the same time.

The current catalog includes:

- `qwen3.8-27b` (default, `Qwen/Qwen3.8-27B-FP8`, dual GPU)
- `qwen3.5-9b` (`Qwen/Qwen3.5-9B`, GPU 1)

The gateway canonicalizes aliases, changes the request's `model` to the canonical ID, and then sends it to the corresponding backend. Requests for the same model share one vLLM scheduler; different models can be loaded and served in parallel.

For detailed design and maintenance information:

- [Gateway architecture](docs/architecture.md)
- [Configuration and file placement](docs/configuration.md)
- [Development and verification](docs/development.md)

## Installation

```bash
uv sync --extra runtime --extra test
```

The current runtime contract and lock file use vLLM 0.27.1. The gateway itself can run unit tests without a GPU runtime; actually loading models still requires a compatible NVIDIA/CUDA environment.

## Model catalog

`models.toml` supports global defaults and per-model overrides. The repository's current core configuration is shown below; see [`models.toml`](models.toml) for the complete GPU placement and runtime fields:

```toml
[default]
context_length = 262144
kv_cache_dtype = "fp8_e4m3"
tensor_parallel_size = 1
data_parallel_size = 1
gpu_memory_utilization = 0.45
max_num_seqs = 16
max_num_batched_tokens = 2048
idle_seconds = 1800
reasoning_parser = "qwen3"
tool_call_parser = "qwen3_coder"
attention_backend = "flashinfer"
async_scheduling = true
flashinfer_autotune = true
enable_prefix_caching = true
speculative_method = "mtp"
num_speculative_tokens = 2

["qwen3.8-27b"]
model_path = "Qwen/Qwen3.8-27B-FP8"
gpu_memory_utilization = 0.35
tensor_parallel_size = 2
max_num_seqs = 64
idle_seconds = 0.0
default = true
aliases = ["qwen3.8-27b-fp8"]
num_speculative_tokens = 3

["qwen3.5-9b"]
model_path = "Qwen/Qwen3.5-9B"
context_length = 131072
gpu_memory_utilization = 0.35
device_ids = [1]
max_num_seqs = 128
idle_seconds = 0.0
aliases = ["qwen3.5-9b"]
```

`model_path` can be a Hugging Face repo ID or a local directory containing Transformers/Safetensors weights. If the canonical name, normalized name, alias, model path, or HF source conflicts across models, the gateway reports an explicit error at startup rather than silently routing to the wrong model.

`attention_backend` is an official catalog field; do not put `--attention-backend` in `vllm_args`, because the gateway rejects managed parameters there.

When `reasoning = "off"`, the gateway adds the following to a chat request when the request does not specify it:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

## Starting the gateway

Run directly:

```bash
uv run python -m vllm_gateway
```

Preload the default model:

```bash
uv run python -m vllm_gateway --preload
```

Preload multiple models at once:

```bash
uv run python -m vllm_gateway \
  --preload-model qwen3.8-27b \
  --preload-model qwen3.5-9b
```

Important options:

```text
--backend-port 30000          start of the private port range
--backend-port-count 64       number of ports available to the allocator
--max-loaded-models 2         hard limit for resident/loading backends
--idle-seconds 0              global idle unload; 0 means keep loaded
--startup-timeout 900         backend model load timeout
--backend-shutdown-timeout 30 request drain/process stop timeout
--tensor-parallel-size N      explicitly override TP for all models
--data-parallel-size N        explicitly override DP for all models
--vllm-arg ARG                additional safe vLLM argument; repeatable
```

Identity, network, and catalog-managed resource flags cannot be overridden through `--vllm-arg`; examples include `--port`, `--model`, `--served-model-name`, TP/DP, memory, context, and KV cache. Use the catalog or the gateway's official CLI options for these settings.

## Companion script (`scripts/vllm`)

The companion script is located at `scripts/vllm` in the repository. Install it to `~/.local/bin/vllm` the first time:

```bash
install -m 755 scripts/vllm ~/.local/bin/vllm
```

The script uses `/home/sam/Projects/vllm` by default and can also be configured with `VLLM_PROJECT_DIR`. It manages only the gateway PID; `ModelManager` stops all backend children during graceful shutdown and does not guess process ownership using `fuser`, `pgrep`, or port regexes.

```bash
vllm start
vllm status
vllm models
vllm load qwen3.8-27b
vllm load qwen3.5-9b
vllm unload qwen3.8-27b
vllm restart
vllm stop
vllm logs 200
vllm download
```

`start` waits for `/health`; when preload is configured, it then waits for `/ready`. If a model still has active requests, `unload` returns 409 and does not forcibly interrupt them.

Common environment variables:

```text
VLLM_PROJECT_DIR
VLLM_CONFIG_FILE
VLLM_SERVICE_HOST / VLLM_SERVICE_PORT
VLLM_BACKEND_PORT / VLLM_BACKEND_PORT_COUNT
VLLM_MAX_LOADED_MODELS
VLLM_PRELOAD / VLLM_PRELOAD_MODELS (comma-separated)
VLLM_MODEL_IDLE_SECONDS
VLLM_TP_SIZE / VLLM_DP_SIZE
VLLM_GPU_MEMORY_UTILIZATION
VLLM_GRACEFUL_SHUTDOWN_TIMEOUT
VLLM_PID_FILE / VLLM_LOG_FILE
```

## API

Liveness and resident state:

```bash
curl http://127.0.0.1:11432/health
curl http://127.0.0.1:11432/ready
curl http://127.0.0.1:11432/v1/models
```

- `/health`: gateway liveness, `loaded_models`, each backend's state/port/PID/active requests, and capacity.
- `/ready`: returns 200 when at least one backend has completed strict readiness; otherwise returns 503.
- `/v1/models`: the complete catalog; a model's `loaded` value means that its backend is currently ready.

Inference:

```bash
curl http://127.0.0.1:11432/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Introduce vLLM in one sentence"}],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }'
```

A JSON request without `model` uses the catalog default. OpenAI SDK clients can set `base_url` to `http://127.0.0.1:11432/v1`.

Local management API (accepts loopback clients only):

```bash
curl http://127.0.0.1:11432/admin/models
curl -X POST http://127.0.0.1:11432/admin/models/qwen3.8-27b/load
curl -X POST http://127.0.0.1:11432/admin/models/qwen3.8-27b/unload
```

Hot-reload the configuration file:

```bash
curl -X POST http://127.0.0.1:11432/admin/reload
```

The gateway reparses `models.toml` and atomically replaces the catalog without restarting the service: newly added models are loaded lazily on their first request; backends for changed or removed specs drain and then stop; a loaded model with an identical spec is unaffected (its process is not restarted). A parse failure returns 400 and the old configuration continues serving. The response includes an `added` / `removed` / `updated` / `kept_loaded` diff summary. If `--default-model` was specified at startup, reload does not change the default model.

## Backend lifecycle and security

- A backend port is mapped deterministically from the canonical model name into a bounded private range; collisions use deterministic probing.
- If an external process occupies a port, the gateway only skips it or reports an error; it never kills that process.
- A backend enters ready only when `/health == 200` and `/v1/models == 200` and contains the expected canonical model.
- Concurrent first requests for the same model share one startup task; different models can load in parallel when capacity permits.
- When capacity is full, only a backend without active requests is unloaded using LRU; if all backends are active/loading, a new model load returns 503.
- Shutdown and idle unload first enter draining; after active leases complete, the gateway-owned process group is stopped.

## Tests

```bash
uv run --extra test python -m pytest -q
```

Run GPU smoke tests after confirming hardware capacity. If model loading fails because of VRAM or a runtime flag, first inspect `vllm logs` and the backend error diagnostics in `/health`; do not arbitrarily change context, KV dtype, or memory fraction before verifying the cause.
