# vLLM multi-model gateway

這個專案提供單一 OpenAI-compatible endpoint（預設 `:11432`），並依 repo 內的 `models.toml` 按需啟動多個獨立 vLLM backend。每個 resident model 有自己的 private port、process、request lease 與 idle policy；目前預設最多同時常駐兩個模型。

目前 catalog 包含：

- `qwen3.8-27b`（預設，`Qwen/Qwen3.8-27B-FP8`，GPU 0）
- `qwen3.5-9b`（`Qwen/Qwen3.5-9B`，GPU 1）
- `qwen3.6-27b`（`Qwen/Qwen3.6-27B-FP8`，GPU 0）

Gateway 會把 alias canonicalize，再將 request 的 `model` 改成 canonical ID 後送往對應 backend。同模型請求共用同一個 vLLM scheduler；不同模型可同時載入並平行服務。

詳細設計與維護方式：

- [Gateway 架構](docs/architecture.md)
- [設定與檔案放置](docs/configuration.md)
- [開發與驗證](docs/development.md)

## 安裝

```bash
uv sync --extra runtime --extra test
```

目前 runtime contract 與 lock file 使用 vLLM 0.27.1。Gateway 本身可在沒有 GPU runtime 的環境執行單元測試；實際載入模型仍需要相容的 NVIDIA/CUDA 環境。

## Model catalog

`models.toml` 支援全域 default 與 per-model override。Repository 目前的核心設定如下；完整 GPU placement 與 `vllm_args` 以 [`models.toml`](models.toml) 為準：

```toml
[default]
context_length = 262144
kv_cache_dtype = "fp8_e4m3"
tensor_parallel_size = 1
data_parallel_size = 1
mem_fraction_static = 0.45
max_num_seqs = 16
max_num_batched_tokens = 2048
idle_seconds = 1800
reasoning_parser = "qwen3"
tool_call_parser = "qwen3_coder"
attention_backend = "flashinfer"

["qwen3.8-27b"]
model_path = "Qwen/Qwen3.8-27B-FP8"
idle_seconds = 0.0
default = true
aliases = ["qwen3.8-27b-fp8"]

["qwen3.5-9b"]
model_path = "Qwen/Qwen3.5-9B"
context_length = 32768
mem_fraction_static = 0.35
max_num_seqs = 128
idle_seconds = 300
aliases = ["qwen3.5-9b"]

["qwen3.6-27b"]
model_path = "Qwen/Qwen3.6-27B-FP8"
idle_seconds = 0.0
aliases = ["qwen3.6-27b-fp8"]
```

`model_path` 可以是 Hugging Face repo ID，或包含 Transformers/Safetensors 權重的本機目錄。Canonical name、normalized name、alias、model path 與 HF source 若跨模型衝突，gateway 會在啟動時明確報錯，不會靜默 routing 到錯誤模型。

`attention_backend` 是 catalog 的正式欄位；不要把 `--attention-backend` 放入 `vllm_args`，因為 gateway 會拒絕其中的受管理參數。

`reasoning = "off"` 會在 chat request 未自行指定時加入：

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

## 啟動 gateway

直接執行：

```bash
uv run python -m sglang_service
```

預載預設模型：

```bash
uv run python -m sglang_service --preload
```

同時預載多個模型：

```bash
uv run python -m sglang_service \
  --preload-model qwen3.8-27b \
  --preload-model qwen3.6-27b
```

重要選項：

```text
--backend-port 30000          private port range 起點
--backend-port-count 64       allocator 可使用的 port 數
--max-loaded-models 2         resident/loading backend 硬上限
--idle-seconds 0              global idle unload；0 表示保持載入
--startup-timeout 900         backend model load timeout
--backend-shutdown-timeout 30 request drain/process stop timeout
--tensor-parallel-size N      明確覆寫所有 model 的 TP
--data-parallel-size N        明確覆寫所有 model 的 DP
--vllm-arg ARG                額外安全的 vLLM argument，可重複
```

Identity、network 與 catalog 管理的 resource flags 不能透過 `--vllm-arg` 覆寫；例如 `--port`、`--model`、`--served-model-name`、TP/DP、memory、context 與 KV cache。這類設定應使用 catalog 或 gateway 的正式 CLI option。

## `~/.local/bin/vllm` 管理工具

Companion script 預設使用 `/home/sam/Projects/vllm`，也可設定 `VLLM_PROJECT_DIR`。它只管理 gateway PID；backend children 全由 `ModelManager` 在 graceful shutdown 中停止，不會使用 `fuser`、`pgrep` 或 port regex 猜測 process ownership。

```bash
vllm start
vllm status
vllm models
vllm load qwen3.8-27b
vllm load qwen3.6-27b
vllm unload qwen3.8-27b
vllm restart
vllm stop
vllm logs 200
vllm download
```

`start` 會等待 `/health`；設定 preload 時會進一步等待 `/ready`。`unload` 若模型仍有 active request 會回 409，不會強制中斷請求。

常用環境變數：

```text
VLLM_PROJECT_DIR
VLLM_CONFIG_FILE
VLLM_SERVICE_HOST / VLLM_SERVICE_PORT
VLLM_BACKEND_PORT / VLLM_BACKEND_PORT_COUNT
VLLM_MAX_LOADED_MODELS
VLLM_PRELOAD / VLLM_PRELOAD_MODELS（逗號分隔）
VLLM_MODEL_IDLE_SECONDS
VLLM_TP_SIZE / VLLM_DP_SIZE
VLLM_GPU_MEMORY_UTILIZATION
VLLM_PID_FILE / VLLM_LOG_FILE
```

舊的 `SGLANG_CONFIG_FILE`、`SGLANG_MODELS_FILE`、`SGLANG_DIR` 等同義環境變數仍保留為低優先級相容 fallback；新設定應使用 `VLLM_*`。

## API

Liveness 與 resident 狀態：

```bash
curl http://127.0.0.1:11432/health
curl http://127.0.0.1:11432/ready
curl http://127.0.0.1:11432/v1/models
```

- `/health`：gateway liveness、`loaded_models`、每個 backend state/port/PID/active requests 與 capacity。
- `/ready`：至少一個 backend 已完成嚴格 readiness 時回 200，否則回 503。
- `/v1/models`：完整 catalog；每個 model 的 `loaded` 代表該 backend 正處於 ready。

Inference：

```bash
curl http://127.0.0.1:11432/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "用一句話介紹 vLLM"}],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }'
```

未提供 `model` 的 JSON request 使用 catalog default。OpenAI SDK 可將 `base_url` 設為 `http://127.0.0.1:11432/v1`。

本機管理 API（僅接受 loopback client）：

```bash
curl http://127.0.0.1:11432/admin/models
curl -X POST http://127.0.0.1:11432/admin/models/qwen3.8-27b/load
curl -X POST http://127.0.0.1:11432/admin/models/qwen3.8-27b/unload
```

熱重載設定檔：

```bash
curl -X POST http://127.0.0.1:11432/admin/reload
```

重新解析 `models.toml` 並原子替換 catalog，不需重啟服務：新增的模型在首次請求時懶載入；spec 變更或被移除的模型其 backend 會 drain 後停止；完全相同且載入中的模型不受影響（process 不重啟）。解析失敗回 400，舊設定繼續服務。回應含 `added` / `removed` / `updated` / `kept_loaded` diff 摘要。若啟動時有指定 `--default-model`，reload 不會改變 default model。

## Backend lifecycle 與安全性

- Backend port 由 canonical model name 穩定映射到 bounded private range；collision 時 deterministic probing。
- 外部程序占用 port 時 gateway 只跳過或報錯，絕不 kill 該程序。
- Backend 只有在 `/health == 200`，且 `/v1/models == 200` 並包含預期 canonical model 時才進入 ready。
- 同模型 concurrent first requests 共用一個 startup task；不同模型在 capacity 允許時可並行載入。
- Capacity 滿時只會 LRU 卸載沒有 active request 的 backend；若所有 backend 都 active/loading，新的 model load 回 503。
- Shutdown 與 idle unload 先進入 draining；等待 active leases 完成後才停止 gateway-owned process group。

## 測試

```bash
uv run --extra test python -m pytest -q
```

GPU smoke test 應在確認硬體容量後執行。若模型載入因 VRAM 或 runtime flag 失敗，先查看 `vllm logs` 與 `/health` 的 backend error diagnostics，不要在未驗證前任意改動 context、KV dtype 或 memory fraction。
