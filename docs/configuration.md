# Configuration and file placement

## Configuration sources and precedence

The gateway reads `models.toml` from the repository root by default. `--config` can specify another TOML file.

The configuration precedence for a single backend command is:

1. Global overrides provided by the gateway CLI or the corresponding `VLLM_*` environment variables.
2. Per-model fields in `models.toml`.
3. Values merged from `[default]` into the model entry.
4. Built-in fallbacks in `ModelManager`.

TP, DP, GPU memory, sequence/token budgets, attention backend, and context are all resolved according to these rules. Global overrides affect every model, so verify before deployment that they do not break GPU isolation settings.

## `models.toml`

`[default]` contains values shared by all models; `["canonical-name"]` contains the model source and overrides. The current parser also accepts a `[models.<name>]` map and a `[[models]]` array of tables, but the repository itself uses root-level named tables.

Common typed fields:

| Category | Fields |
| --- | --- |
| Identity | `model_path`, `hf_repo`, `aliases`, `default` |
| Capacity | `tensor_parallel_size`, `data_parallel_size`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens` |
| Context/KV | `context_length`, `kv_cache_dtype`, `rope_scaling`, `rope_scale`, `yarn_orig_ctx`, `rope_theta` |
| Request behavior | `reasoning`, `reasoning_parser`, `tool_call_parser`, `chat_template` |
| Lifecycle | `idle_seconds` |
| Runtime | `attention_backend`, `device_ids`, `async_scheduling`, `flashinfer_autotune`, `enable_prefix_caching`, `vllm_args` |
| Speculative decoding | `speculative_method`, `num_speculative_tokens` |

`device_ids` uses a TOML integer array (for example, `[0, 1]`); `async_scheduling`, `flashinfer_autotune`, and `enable_prefix_caching` use booleans. When unset, no flag is emitted; `false` emits the corresponding `--no-` flag to override a default of `true`. `speculative_method` is a string and `num_speculative_tokens` is an integer; `--speculative-config` is generated only when a method is set, and the token count may be omitted. These new fields support `[default]` and per-model values and have no additional CLI/environment override.

`model_path` can point to a local Transformers/Safetensors directory or a Hugging Face repo ID. If it looks like an explicit local path but does not exist, the gateway reports an error.

The canonical name, normalized name, alias, model path, source, and Hugging Face repo must each be unique across the entire catalog. A name conflict stops the gateway before it binds the public port.

### Shared GPU memory budget

`gpu_memory_utilization` is converted to vLLM's `--gpu-memory-utilization`; its value is a fraction of the GPU's **total memory**, not of the free memory at startup. At startup, vLLM requires free memory to cover at least `total memory × gpu_memory_utilization`; if an external process on the same GPU has already consumed too much memory, the backend exits before loading weights.

In the `qwen3.5-9b` incident on 2026-08-27, GPU 1 had 30.64 GiB remaining, but `0.35` required 33.24 GiB, so `vllm load` returned 503. After lowering that model's value to `0.30` (a 28.49 GiB budget) and hot-reloading, the backend completed weight loading, CUDA graph capture, and readiness.

When you encounter a similar error:

1. Use `nvidia-smi` to check free memory on the target GPU and the other processes using it.
2. Find the free/desired GiB reported by vLLM in the backend stderr; the gateway's 503 only summarizes the subprocess exit.
3. Remove processes that should not coexist, or lower that model's `gpu_memory_utilization`, while preserving the space needed for weights, activations, CUDA graphs, and the KV cache.
4. Run `POST /admin/reload` and load the model again; confirm that `/health` reaches `ready`, then send a completion smoke test.

Do not convert all currently free memory into a fraction. Shared GPU usage changes, and operating near the limit can cause the next startup to fail again.

### `vllm_args` boundary

`vllm_args` is only for safe upstream parameters for which the gateway has no typed field. The gateway manages the following categories, which cannot be overridden in `vllm_args` or with `--vllm-arg`:

- model identity, host, port, and served model name
- TP/DP, GPU memory, context, sequence/token budget, and KV dtype
- load format, runner, reasoning/tool parser, and chat template
- attention backend and Hugging Face overrides
- `--device-ids`, `--async-scheduling`, `--enable-flashinfer-autotune`, and `--speculative-config`
- `--enable-chunked-prefill`, `--enable-prefix-caching`, and `--enable-auto-tool-choice`

Protected flags are also rejected in underscore spelling, `--flag=value` form, and `--no-<flag>` form.

The current catalog's generative models use the V1 chunked-prefill default, so the duplicate `--enable-chunked-prefill` has been removed. The prefix-caching default depends on model capabilities and the hybrid architecture; Qwen hybrid models retain `enable_prefix_caching = true` to preserve the behavior that was explicitly enabled before migration.

The gateway retains its own fallbacks: `gpu_memory_utilization = 0.90`, `max_num_seqs = 256`, and `max_num_batched_tokens = 4096`. In vLLM 0.27.1, the GPU memory default is `0.92`; the `SchedulerConfig` seq/token dataclass defaults are `128` / `2048`, but `EngineArgs` adjusts them automatically at startup based on the device and usage. Catalog or global overrides still take precedence over gateway fallbacks.

When adding a managed parameter, update `ModelSpec`, the catalog parser, CLI/manager precedence, and command-generation tests. Do not remove a flag from the protected set to bypass validation.

## Environment variables

The gateway source directly supports the main `VLLM_*` settings, including:

```text
VLLM_CONFIG_FILE
VLLM_DEFAULT_MODEL
VLLM_SERVICE_HOST / VLLM_SERVICE_PORT
VLLM_ADMIN_API_KEY
VLLM_BACKEND_PORT / VLLM_BACKEND_PORT_COUNT
VLLM_MAX_LOADED_MODELS
VLLM_TP_SIZE / VLLM_DP_SIZE
VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS / VLLM_MAX_NUM_BATCHED_TOKENS
VLLM_ATTENTION_BACKEND
VLLM_CONTEXT_LENGTH
VLLM_STARTUP_TIMEOUT / VLLM_BACKEND_SHUTDOWN_TIMEOUT
VLLM_GRACEFUL_SHUTDOWN_TIMEOUT
VLLM_MODEL_IDLE_SECONDS
VLLM_SERVICE_LOG_LEVEL
```

The deployment companion script supports additional lifecycle variables such as preload, PID/log paths, and stop timeout. The script is in `scripts/vllm` in the repository and is installed to `~/.local/bin/vllm` for deployment. It also supports `VLLM_ADMIN_KEY_FILE` for specifying a local key file; by default, it looks for `vllm-gateway/admin.key` under `XDG_CONFIG_HOME` (`~/.config` when unset). The gateway Python code does not read this file itself.

## File placement

| Content | Location | Readers |
| --- | --- | --- |
| Version-controlled model catalog | `models.toml` or the file specified by `VLLM_CONFIG_FILE` | `ModelCatalog`, companion script |
| Python package metadata and locked dependencies | `pyproject.toml`, `uv.lock` | uv, build backend |
| Architecture and development guidelines | `docs/`, each level's `AGENTS.md` | people and coding agents |
| Model weights and Hugging Face cache | local model path or Hugging Face cache outside the repository | vLLM, huggingface-hub |
| Gateway PID/log | `~/.cache/vllm-gateway.pid`, `~/.cache/vllm-gateway.log` by default | companion script |
| Management API key | `~/.config/vllm-gateway/admin.key` or `VLLM_ADMIN_KEY_FILE`; not version-controlled | companion script; directory 700, file 600 |
| Python/test/build artifacts | `.venv/`, `__pycache__/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info` | development tools; do not commit |
| Agent harness state | `.agents/`, `.pi/` | agent tooling; excluded by `.gitignore` |

## Secrets

Management endpoints require `VLLM_ADMIN_API_KEY` (it must be set to a non-empty value before startup) and remain restricted to loopback sources; the gateway refuses to start when it is unset. Calls to `GET /admin/models`, `POST /admin/reload`, `POST /admin/models/{model}/load`, and `POST /admin/models/{model}/unload` must include `Authorization: Bearer <key>`; the `models`, `load`, and `unload` commands in `scripts/vllm` use the same environment variable. `/health`, `/ready`, `/v1/*`, and other proxy routes do not require this key; the Authorization header on inference requests is forwarded to the backend as usual. Changing the key requires restarting the gateway; catalog reload does not update the key.

For convenient local operation, `scripts/vllm` prefers a non-empty `VLLM_ADMIN_API_KEY`; when it is unset, it reads the local key file and exports it to the gateway. The file contains only a single-line key, may have a trailing newline, must not contain `export` or an assignment, and the script does not execute its contents. Restrict the directory to 700 and the file to 600; updating or reinstalling the script does not modify this file, so `vllm models`, `vllm load`, and `vllm restart` can be used without repeatedly exporting the key manually. When running `python -m vllm_gateway` directly, you must still provide the environment variable yourself.

Use a sufficiently long random value and pass it through a local key file with restricted permissions, the process environment, or an external secret manager. Avoid putting the key in CLI arguments, `models.toml`, README, docs, test fixtures, or log examples. A bearer key does not protect against users on the same host who can read the gateway process environment or memory of a process with the same UID; connections across hosts must not rely only on loopback checks and require TLS, a reverse proxy, and network isolation. Deployment secrets such as Hugging Face tokens and proxy credentials must likewise not be committed.

## Safe update workflow

1. After editing the catalog, first run the catalog and command-generation tests.
2. Use the management key to call `POST /admin/reload` (for example, `printf 'Authorization: Bearer %s\n' "$VLLM_ADMIN_API_KEY" | curl --header @- -X POST http://127.0.0.1:11432/admin/reload`); first inspect the `added`, `removed`, `updated`, and `kept_loaded` fields in the response.
3. Check `/health` and `/v1/models`. If weights need to be loaded, then run a GPU smoke test for the target model.
4. When reload fails, do not restart to mask the parser problem; the service retains the old catalog, so first correct the configuration error indicated by the response and log.
