# Configuration

## Gateway settings

| CLI flag | Environment variable | Default |
| --- | --- | --- |
| `--host` | `VLLM_SERVICE_HOST` | `0.0.0.0` |
| `--port` | `VLLM_SERVICE_PORT` | `11432` |
| `--hub-cache` | `HF_HUB_CACHE` | Hugging Face default (`~/.cache/huggingface/hub`) |
| `--state-dir` | `VLLM_STATE_DIR` | `~/.local/state/vllm-gateway` |
| `--backend-port`, `--backend-port-count` | `VLLM_BACKEND_PORT`, `VLLM_BACKEND_PORT_COUNT` | `30000`, `64` |
| `--startup-timeout` | `VLLM_STARTUP_TIMEOUT` | `900` seconds |
| `--backend-shutdown-timeout` | `VLLM_BACKEND_SHUTDOWN_TIMEOUT` | `30` seconds per backend |
| `--graceful-shutdown-timeout` | `VLLM_GRACEFUL_SHUTDOWN_TIMEOUT` | `30` seconds for Uvicorn |
| `--log-level` | `VLLM_SERVICE_LOG_LEVEL` | `info` |

The image sets `HF_HUB_CACHE=/data/huggingface/hub` and `VLLM_STATE_DIR=/state`. `HF_TOKEN` is read by `huggingface_hub` and is needed only for gated repositories. Keep it in `.env`, which is not committed.

## Fixed vLLM settings

These are constants in `vllm_gateway/manager.py`. Every backend gets them.

| Setting | Value |
| --- | --- |
| Idle unload | 3600 seconds |
| `--max-num-seqs`, `--max-num-batched-tokens` | 16, 2048 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--attention-backend` | `flashinfer` |
| `--async-scheduling`, `--enable-flashinfer-autotune`, `--enable-prefix-caching` | on |
| Tensor and data parallel | 1 (the container sees one GPU) |
| Speculative decoding | `{"method": "mtp", "num_speculative_tokens": 2}` when the model has MTP weights |

## Family profile

The parsers come from `model_type` in `config.json` (the top-level type, then `text_config`). The table is `profile_for()` in `vllm_gateway/catalog.py`.

| `model_type` | Reasoning parser | Tool-call parser |
| --- | --- | --- |
| `qwen3_<digit>…` (for example `qwen3_5`), `qwen3_next` | `qwen3` | `qwen3_coder` |
| `qwen3`, `qwen3_moe` | `qwen3` | `hermes` |
| anything else | none | none |

MTP is enabled only when `config.json` declares MTP layers (`mtp_num_hidden_layers` or `num_nextn_predict_layers`) and the safetensors tensor names contain `mtp.`. Some quantized repos keep the config field but drop the weights.

A repo is listed as a model only when its `config.json` has an architecture ending in `ForCausalLM` or `ForConditionalGeneration`, it has `max_position_embeddings`, the layer and head fields needed for the KV size, and all of its safetensors files. A cache with a `.incomplete` blob is hidden.

## Memory estimate

`vllm_gateway/memory.py` computes, for a model and a context length:

```text
kv_per_token = 2 × full_attention_layers × kv_heads × head_dim × 1 byte (fp8)
per_sequence = linear_state × LINEAR_STATE_FACTOR (5) + kv_per_token × context_length
kv_pool      = per_sequence × KV_FULL_SEQS (2) × KV_MARGIN (1.2)
estimate     = weights + kv_pool + OVERHEAD (4 GiB)
required     = estimate + SAFETY (2 GiB)
```

- `full_attention_layers` counts `full_attention` entries in `layer_types`. Without `layer_types` it is `num_hidden_layers / full_attention_interval`, and otherwise `num_hidden_layers`. With MTP enabled the draft layers are added.
- The backend is started with `--kv-cache-memory-bytes <kv_pool>` and `--gpu-memory-utilization <ceil(estimate / GPU total, 0.01)>`. With `--kv-cache-memory-bytes` set, vLLM skips memory profiling and reserves exactly that KV pool.
- `linear_state` is the recurrent state of the linear-attention layers of a hybrid model (SSM state plus short convolution, computed from the `linear_*` fields and `mamba_ssm_dtype`). It is 0 for dense models. vLLM refuses to start when the KV pool cannot hold one request of the full context length, and for hybrid models it reserves several times the raw state per sequence, hence the factor.
- A model loads only when the free memory of the whole GPU is at least `required`.

Calibration on an RTX PRO 6000 (96 GiB) with Qwen3.5-9B:

- At 32768 tokens the weights took 18.1 GiB and the measured total was about 21 GiB, with no growth under four concurrent 14k-token prompts. `OVERHEAD_BYTES` is set from that.
- An early formula without the recurrent state reserved 0.34 GiB at 8192 tokens, and vLLM refused to start because one request needs 0.37 GiB. Fitting the 8192 and 32768 token logs gives about 18.5 KB per token and 245 MB per sequence, against 51.5 MB of raw state. `LINEAR_STATE_FACTOR` is set from that.

Recalibrate the constants in `memory.py` when you change the fixed vLLM settings, the GPU, or add a model family. A pool that is too small shows up as `kv_cache_memory_insufficient`.

## Files and users

| Content | Location |
| --- | --- |
| Models | Shared Hugging Face cache, mounted at `/data/huggingface` |
| Aliases | `$VLLM_STATE_DIR/aliases.json` (the `gateway-state` volume) |
| Backend output | Gateway log (`docker compose logs gateway`) |
| Deployment settings | `.env` (not committed; template in `.env.example`) |

The container runs as `GATEWAY_UID:GATEWAY_GID` with `umask 002`, so files it downloads into the shared cache belong to the cache's group and stay writable for its members.

## Security

The gateway has no authentication and listens on `0.0.0.0`. Anyone who can reach the port can create and delete aliases, trigger downloads and use every model. Keep it on an internal network. The Hugging Face token must not be committed or put in logs or command-line arguments.
