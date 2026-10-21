# Shared vLLM gateway

An OpenAI-compatible endpoint (default `:11432`) for a lab server. It serves the models in a shared Hugging Face cache through independent vLLM backends, one per user alias, on a single GPU. It runs in Docker and has no authentication, so it belongs on an internal network only.

## How it works

- A **model** is a complete Hugging Face repo in the cache (`/data/huggingface/hub`). `GET /models/local` lists them and `POST /models/download` fetches more.
- An **alias** binds a model to a context length. Create one per use (`POST /aliases`) and call it through the OpenAI API by putting the alias in `model`.
- Each alias gets its own vLLM process, started on its first request or by `POST /aliases/{alias}/load`.
- Everything except the model and context length is fixed. Parsers and speculative decoding come from the model's `config.json`, and the GPU memory share is computed from the model size and context length.
- A backend that sits idle for one hour is unloaded. Loaded backends are never stopped to make room for another model, so when a new model does not fit in the free GPU memory the load fails with `insufficient_gpu_memory` until a backend is unloaded or times out.
- Only one backend loads at a time (about 5 minutes for a 9B model), so a request for a second unloaded model waits for the first load to finish.

## Deploy with Docker

The image is built on `vllm/vllm-openai`, so the host needs the NVIDIA container toolkit.

```bash
cp .env.example .env     # set GATEWAY_UID, GATEWAY_GID and GATEWAY_GPU
docker compose up -d --build
docker compose logs -f gateway
```

`.env` settings:

| Variable | Meaning |
| --- | --- |
| `GATEWAY_UID`, `GATEWAY_GID` | User and group the container runs as. Use the owner of the shared cache and its group (`hfusers`) so downloads stay writable for everyone |
| `GATEWAY_GPU` | UUID of the one GPU the gateway may use (`nvidia-smi -L`) |
| `GATEWAY_PORT` | Published port, default 11432 |
| `HF_CACHE_DIR` | Host path of the shared cache, default `/data/huggingface` |
| `HF_TOKEN` | Only for gated or private repositories |

Aliases are stored in the `gateway-state` volume and survive restarts. `docker compose stop` drains active requests and stops every backend (it waits up to 120 seconds).

The GPU may be shared with other people's processes. The gateway reads the free memory of the whole GPU before every load and never signals processes it did not start.

## API

A guide for people who only send requests is in [docs/usage.md](docs/usage.md). Examples assume `http://140.109.21.51:11432`.

```bash
# Models available in the cache, with size, maximum context and the aliases that use them
curl http://140.109.21.51:11432/models/local

# Download a model in the background, then poll the job
curl -X POST http://140.109.21.51:11432/models/download -H 'content-type: application/json' \
  -d '{"repo_id": "Qwen/Qwen3.5-9B"}'
curl http://140.109.21.51:11432/models/downloads/<job id>

# Create an alias, optionally load it, and use it
curl -X POST http://140.109.21.51:11432/aliases -H 'content-type: application/json' \
  -d '{"alias": "alice-9b-64k", "model": "Qwen/Qwen3.5-9B", "context_length": 65536}'
curl -X POST http://140.109.21.51:11432/aliases/alice-9b-64k/load
curl http://140.109.21.51:11432/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "alice-9b-64k", "messages": [{"role": "user", "content": "Hello"}]}'
```

| Endpoint | Purpose |
| --- | --- |
| `GET /models/local` | Loadable models in the cache. Repos that are still downloading or incomplete are hidden |
| `POST /models/download` | Body `{repo_id, revision?}`. Returns 202 and a job. Only safetensors, JSON, tokenizer and template files are fetched. Rejected with 507 when less than 20 GiB of disk would remain |
| `GET /models/downloads`, `GET /models/downloads/{id}` | Job state: `queued`, `running`, `succeeded`, `failed`. Jobs run one at a time and are kept in memory |
| `GET /aliases` | Aliases with their backend state |
| `POST /aliases` | Body `{alias, model, context_length}`. `alias` matches `[A-Za-z0-9._-]{1,64}`. `context_length` is between 1024 and the model's maximum. Returns 400 when the model could never fit in the GPU |
| `DELETE /aliases/{alias}` | Stops the backend and removes the alias. Returns 409 while it has active requests |
| `POST /aliases/{alias}/load`, `POST /aliases/{alias}/unload` | Load now, or unload. Unload returns 409 while requests are active |
| `GET /v1/models`, `GET /v1/models/{alias}` | OpenAI model list with a `loaded` flag |
| `GET /health`, `GET /ready` | Gateway state, backend states and free GPU memory. `/ready` returns 503 until a backend is ready |
| any other path | Proxied to the alias named in the JSON `model` field |

Requests without `model` return 400, and only exact alias names route (a repo id returns 404). To change an alias's context length, delete it and create it again.

When a model does not fit, the request fails with 503 and `code: insufficient_gpu_memory`, which states how many GiB were needed and free. Other load failures return a safe `code` and a `diagnostic_id` to look up in `docker compose logs`.

## Development

```bash
uv sync --extra test --extra dev
uv run --extra test python -m pytest -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
```

Unit tests need no GPU. See [architecture](docs/architecture.md), [configuration](docs/configuration.md) and [development](docs/development.md).
