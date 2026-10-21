# Using the gateway

This guide is for people who send requests to the gateway. Running and deploying it is covered in the [README](../README.md).

The gateway runs on idlab and serves an OpenAI-compatible API at `http://140.109.21.51:11432`. Use the IP address: the host name `idlab` does not resolve on the other lab machines. The gateway has no authentication and is reachable only from the internal network. It shares one GPU with everyone who uses it, and with other processes on that GPU.

## Concepts

- A **model** is a Hugging Face repo in the shared cache, such as `Qwen/Qwen3.5-9B`.
- An **alias** is a name you choose for a model plus a context length, such as `alice-9b-64k`. Requests name an alias, never a repo id.
- A **backend** is the vLLM server that runs one alias. It starts on demand and its states are `stopped`, `loading`, `ready`, `draining` and `failed`.

Only the model and the context length are yours to pick. Everything else, such as KV cache precision, batching and speculative decoding, is fixed by the gateway.

## Quick start

```bash
# 1. See which models are available and their maximum context
curl http://140.109.21.51:11432/models/local

# 2. Create your alias
curl -X POST http://140.109.21.51:11432/aliases -H 'content-type: application/json' \
  -d '{"alias": "alice-9b-64k", "model": "Qwen/Qwen3.5-9B", "context_length": 65536}'

# 3. Chat. The first request loads the backend, so it can take a few minutes
curl http://140.109.21.51:11432/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "alice-9b-64k", "messages": [{"role": "user", "content": "Hello"}]}'

# 4. Unload when you are done, so others can use the memory
curl -X POST http://140.109.21.51:11432/aliases/alice-9b-64k/unload
```

With the OpenAI Python SDK, point `base_url` at `/v1` and pass any API key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://140.109.21.51:11432/v1", api_key="unused", timeout=900)
reply = client.chat.completions.create(
    model="alice-9b-64k",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

## Aliases

- `alias` matches `[A-Za-z0-9._-]{1,64}`. Prefix it with your name so others can tell whose it is.
- `context_length` is between 1024 and the model's `max_context` from `/models/local`.
- Creation fails with 400 when the name is taken, the context is out of range, or the alias could not fit even on an empty GPU. An unknown model returns 404.
- A context length cannot be changed. Delete the alias and create it again.
- Aliases survive gateway restarts. Anyone can load, unload or delete any alias, so only touch your own.

Smaller context lengths need less GPU memory, so pick the smallest one your work needs.

## Loading

A backend loads in one of two ways:

- **Implicitly**, by sending a request to a stopped alias. The request waits until the backend is ready and then gets its answer.
- **Explicitly**, with `POST /aliases/{alias}/load`. The call returns when the backend is ready, with `"state": "ready"`.

Measured load times on idlab:

| Model | First load | Later loads |
| --- | --- | --- |
| `Qwen/Qwen3.5-9B` | 170 to 220 s | about 80 s |
| `Qwen/Qwen3-0.6B` | about 60 s | about 55 s |

Only one backend loads at a time. If another alias is loading, your load waits for it to finish first. Several requests to the same stopped alias share one load.

Set your client timeout to at least 15 minutes for the first request, or load explicitly before you start sending work.

## Memory and unloading

The gateway never stops a loaded backend to make room for another one. When the GPU does not have enough free memory for your alias, the load fails at once with 503:

```json
{"error": {"type": "backend_unavailable", "code": "insufficient_gpu_memory",
  "message": "insufficient GPU memory: 25.9 GiB needed, 5.3 GiB free",
  "diagnostic_id": "e022d0b3e61c43ad887be866d4a9bff6"}}
```

Memory is freed only in these cases:

- a backend has received no requests for one hour and is unloaded automatically
- someone calls `POST /aliases/{alias}/unload`
- someone deletes the alias

When you get `insufficient_gpu_memory`, retry later or use an alias with a smaller model or context length. `GET /health` shows the current free memory in `capacity.gpu.free_bytes` and which backends are loaded.

Unloading returns 409 while the backend still has requests in flight. The same applies to deleting an alias.

## Requests

Any path other than the gateway's own endpoints is forwarded to the alias named in the JSON `model` field. This covers `/v1/chat/completions`, `/v1/completions` and the other vLLM OpenAI endpoints. A request without `model` returns 400, and a repo id or an unknown name returns 404.

### Concurrency

Each alias runs in its own vLLM process. Requests to different aliases run in parallel. Requests to the same alias are batched, up to 16 sequences at a time, and the rest wait in vLLM's queue.

### Streaming

Set `"stream": true`. The response is Server-Sent Events and ends with `data: [DONE]`.

### Reasoning

For models with a reasoning parser, such as the Qwen3 family, the model's thinking is returned in `message.reasoning` and the answer in `message.content`. Thinking uses up `max_tokens`. If the limit is small, `content` can be empty because the model never finished thinking.

To turn thinking off on Qwen3 models:

```json
{"model": "alice-9b-64k", "messages": [...], "chat_template_kwargs": {"enable_thinking": false}}
```

### Tool calls

Tool calling is enabled with automatic tool choice when the model has a tool-call parser. Pass `tools` in the OpenAI format. `/models/local` lists each model's `reasoning_parser` and `tool_call_parser`.

## Downloading a model

If the model you need is not in `/models/local`, download it:

```bash
curl -X POST http://140.109.21.51:11432/models/download -H 'content-type: application/json' \
  -d '{"repo_id": "Qwen/Qwen3-0.6B"}'
curl http://140.109.21.51:11432/models/downloads/<job id>
```

The job state goes `queued`, `running`, then `succeeded` or `failed`. Downloads run one at a time. Only safetensors, JSON, tokenizer and template files are fetched. The request is rejected with 507 when less than 20 GiB of disk would remain. Downloaded models are in the shared cache and visible to everyone.

## Endpoint reference

| Endpoint | Purpose |
| --- | --- |
| `GET /models/local` | Models in the cache with size, `max_context`, parsers and the aliases that use them |
| `POST /models/download` | Body `{repo_id, revision?}`. Returns 202 and a job |
| `GET /models/downloads`, `GET /models/downloads/{id}` | Download jobs |
| `GET /aliases` | All aliases with `state`, `active_requests` and `last_used` |
| `POST /aliases` | Body `{alias, model, context_length}`. Returns 201 |
| `DELETE /aliases/{alias}` | Stops the backend and removes the alias |
| `POST /aliases/{alias}/load` | Loads the backend and waits until it is ready |
| `POST /aliases/{alias}/unload` | Unloads the backend |
| `GET /v1/models`, `GET /v1/models/{alias}` | OpenAI model list, with a `loaded` flag |
| `GET /health` | Backend states and free GPU memory |
| any other path | Forwarded to the alias in `model` |

## Errors

Gateway errors have the shape `{"error": {"type": ..., "message": ...}}`. Load failures also carry `code` and `diagnostic_id`.

| Status | Cause |
| --- | --- |
| 400 | Invalid JSON, missing `model`, bad alias name, context out of range, or alias too large for the GPU |
| 404 | Unknown alias or model |
| 409 | Unload or delete while the alias has active requests |
| 503 | The backend could not be loaded. See `error.code` |
| 507 | Not enough disk for a download |

| `error.code` | Meaning | What to do |
| --- | --- | --- |
| `insufficient_gpu_memory` | The GPU does not have enough free memory | Retry later, or use a smaller model or context |
| `kv_cache_memory_insufficient`, `cuda_out_of_memory` | vLLM ran out of memory while starting | Retry later, or use a smaller context |
| `backend_startup_timeout` | The backend did not become ready in time | Retry. If it repeats, report it |
| `backend_exited` | The backend crashed | Retry. If it repeats, report it |
| other codes | Startup or configuration failure | Report it |

When you report a failure, include the `diagnostic_id`. The operators use it to find the backend log. After a failed load, the alias cannot be loaded again for about 2 seconds.
