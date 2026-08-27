# 設定與檔案放置

## 設定來源與優先級

Gateway 預設讀取 repository root 的 `models.toml`。`--config` 可指定其他 TOML；`.ini` 路徑只保留 legacy preset 相容性。

單一 backend command 的設定優先級為：

1. Gateway CLI 或對應 `VLLM_*` 環境變數提供的全域 override。
2. `models.toml` 的 per-model 欄位。
3. `[default]` 合併到 model entry 的值。
4. `ModelManager` 的內建 fallback。

例如 TP、DP、GPU memory、sequence/token budget、attention backend 與 context 都依此規則解析。全域 override 會影響每個 model，部署前需確認沒有破壞各 GPU 的隔離配置。

## `models.toml`

`[default]` 放所有 model 共用的值；`["canonical-name"]` 放 model source 與 override。現有 parser 也接受 `[models.<name>]` map 和 `[[models]]` array-of-tables，但 repository 本身使用 root named tables。

常用 typed fields：

| 類別 | 欄位 |
|---|---|
| Identity | `model_path`, `hf_repo`, `hf_file`, `aliases`, `default` |
| Capacity | `tensor_parallel_size`, `data_parallel_size`, `mem_fraction_static`, `max_num_seqs`, `max_num_batched_tokens` |
| Context/KV | `context_length`, `kv_cache_dtype`, `rope_scaling`, `rope_scale`, `yarn_orig_ctx`, `rope_theta` |
| Request behavior | `reasoning`, `reasoning_parser`, `tool_call_parser`, `chat_template` |
| Lifecycle | `idle_seconds` |
| Runtime | `attention_backend`, `vllm_args` |

`model_path` 可指向本機 Transformers/Safetensors 目錄或 Hugging Face repo ID。若它看起來是明確的本機路徑但不存在，gateway 會報錯。指定 `hf_repo` 和 `hf_file` 時，catalog 才會用 `huggingface_hub.hf_hub_download()` 解析單一檔案。

Canonical name、normalized name、alias、model path、source 與 Hugging Face repo 必須在整份 catalog 中唯一。名稱衝突會使 gateway 在 bind public port 前停止。

### `vllm_args` 邊界

`vllm_args` 只放 gateway 沒有 typed field 的安全上游參數。以下類別由 gateway 管理，不能在 `vllm_args` 或 `--vllm-arg` 覆寫：

- model identity、host、port、served model name
- TP/DP、GPU memory、context、sequence/token budget、KV dtype
- load format、runner、reasoning/tool parser、chat template
- attention backend 與 Hugging Face overrides

需要新增受管理參數時，修改 `ModelSpec`、catalog parser、CLI/manager precedence 和 command-generation tests。不要從 protected set 移除 flag 來規避 validation。

## 環境變數

Gateway source 直接支援主要的 `VLLM_*` 設定，包括：

```text
VLLM_CONFIG_FILE
VLLM_DEFAULT_MODEL
VLLM_SERVICE_HOST / VLLM_SERVICE_PORT
VLLM_BACKEND_PORT / VLLM_BACKEND_PORT_COUNT
VLLM_MAX_LOADED_MODELS
VLLM_TP_SIZE / VLLM_DP_SIZE
VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS / VLLM_MAX_NUM_BATCHED_TOKENS
VLLM_ATTENTION_BACKEND
VLLM_CONTEXT_LENGTH
VLLM_STARTUP_TIMEOUT / VLLM_BACKEND_SHUTDOWN_TIMEOUT
VLLM_MODEL_IDLE_SECONDS
VLLM_SERVICE_LOG_LEVEL
```

`cli.py` 還接受部分舊 `SGLANG_*` config path 名稱作低優先級 fallback。部署用 companion script 支援更多 lifecycle 變數，例如 preload、PID/log path 與 stop timeout；該 script 位於 `~/.local/bin/vllm`，不由此 repository 安裝。

## 檔案放置

| 內容 | 位置 | 誰讀取 |
|---|---|---|
| 可版本控制的 model catalog | `models.toml` 或 `VLLM_CONFIG_FILE` 指定的檔案 | `ModelCatalog`、companion script |
| Python package metadata 與 locked dependencies | `pyproject.toml`, `uv.lock` | uv、build backend |
| 架構與開發規範 | `docs/`, 各層 `AGENTS.md` | 人員與 coding agents |
| Model 權重與 Hugging Face cache | repository 外的本機 model path 或 Hugging Face cache | vLLM、huggingface-hub |
| Gateway PID/log | 預設 `~/.cache/vllm-gateway.pid`、`~/.cache/vllm-gateway.log` | companion script |
| Python/test/build artifacts | `.venv/`, `__pycache__/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info` | 開發工具；不要提交 |
| Agent harness state | `.agents/`, `.pi/` | agent tooling；已由 `.gitignore` 排除 |

## Secrets

Repository 沒有自建 API key 或 secret-file loader。Hugging Face token、代理憑證或其他部署 secret 應透過 process environment、Hugging Face credential store 或外部 secret manager 提供。不要把 token 寫進 `models.toml`、README、docs、test fixture 或 log 範例。

## 安全更新流程

1. 編輯 catalog 後先執行 catalog 與 command-generation tests。
2. 呼叫 `POST /admin/reload`；先檢查回應中的 `added`、`removed`、`updated`、`kept_loaded`。
3. 檢查 `/health` 和 `/v1/models`。需要載入權重時，再執行目標 model 的 GPU smoke test。
4. Reload 失敗時不要重啟來掩蓋 parser 問題；服務會保留舊 catalog，先修正回應與 log 指出的設定錯誤。
