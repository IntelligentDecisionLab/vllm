# 設定與檔案放置

## 設定來源與優先級

Gateway 預設讀取 repository root 的 `models.toml`。`--config` 可指定其他 TOML。

單一 backend command 的設定優先級為：

1. Gateway CLI 或對應 `VLLM_*` 環境變數提供的全域 override。
2. `models.toml` 的 per-model 欄位。
3. `[default]` 合併到 model entry 的值。
4. `ModelManager` 的內建 fallback。

例如 TP、DP、GPU memory、sequence/token budget、attention backend 與 context 都依此規則解析。全域 override 會影響每個 model，部署前需確認沒有破壞各 GPU 的隔離配置。

## `models.toml`

`[default]` 放所有 model 共用的值；`["canonical-name"]` 放 model source 與 override。現有 parser 也接受 `[models.<name>]` map 和 `[[models]]` array-of-tables，但 repository 本身使用 root named tables。

常用 typed fields：

| 類別             | 欄位                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| Identity         | `model_path`, `hf_repo`, `aliases`, `default`                                                               |
| Capacity         | `tensor_parallel_size`, `data_parallel_size`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens` |
| Context/KV       | `context_length`, `kv_cache_dtype`, `rope_scaling`, `rope_scale`, `yarn_orig_ctx`, `rope_theta`               |
| Request behavior | `reasoning`, `reasoning_parser`, `tool_call_parser`, `chat_template`                                          |
| Lifecycle        | `idle_seconds`                                                                                                |
| Runtime          | `attention_backend`, `device_ids`, `async_scheduling`, `flashinfer_autotune`, `enable_prefix_caching`, `vllm_args`                                                                              |
| Speculative decoding | `speculative_method`, `num_speculative_tokens` |

`device_ids` 使用 TOML 整數陣列（例如 `[0, 1]`）；`async_scheduling`、`flashinfer_autotune`、`enable_prefix_caching` 使用 boolean。未設定時不輸出 flag，`false` 會輸出對應 `--no-` flag，以覆寫 default 的 `true`。`speculative_method` 使用字串，`num_speculative_tokens` 使用整數；只有設定 method 時才產生 `--speculative-config`，tokens 可省略。這些新欄位支援 `[default]` 與 per-model，沒有新增 CLI/env override。

`model_path` 可指向本機 Transformers/Safetensors 目錄或 Hugging Face repo ID。若它看起來是明確的本機路徑但不存在，gateway 會報錯。

Canonical name、normalized name、alias、model path、source 與 Hugging Face repo 必須在整份 catalog 中唯一。名稱衝突會使 gateway 在 bind public port 前停止。

### 共用 GPU 的記憶體 budget

`gpu_memory_utilization` 會轉成 vLLM 的 `--gpu-memory-utilization`，其值是 GPU **總記憶體**的比例，不是啟動當下 free memory 的比例。vLLM 啟動時要求 free memory 至少能覆蓋 `total memory × gpu_memory_utilization`；同卡的外部程序若先占用太多記憶體，backend 會在載入權重前退出。

2026-08-27 的 `qwen3.5-9b` incident 中，GPU 1 剩餘 30.64 GiB，但 `0.35` 要求 33.24 GiB，因此 `vllm load` 回 503。將該 model 的值降為 `0.30`（28.49 GiB budget）並 hot reload 後，backend 完成權重載入、CUDA graph capture 與 readiness。

遇到同類錯誤時：

1. 用 `nvidia-smi` 確認目標 GPU 的 free memory 與其他 process。
2. 從 backend stderr 找出 vLLM 回報的 free/desired GiB；gateway 的 503 只摘要 subprocess exit。
3. 移除不應共存的 process，或降低該 model 的 `gpu_memory_utilization`，並保留權重、activation、CUDA graph 和 KV cache 所需空間。
4. 執行 `POST /admin/reload` 後重新 load；確認 `/health` 進入 `ready`，再送 completion smoke test。

不要把目前 free memory 全部換算成 fraction。共用 GPU 的使用量會變動，貼近臨界值會讓下次啟動再次失敗。

### `vllm_args` 邊界

`vllm_args` 只放 gateway 沒有 typed field 的安全上游參數。以下類別由 gateway 管理，不能在 `vllm_args` 或 `--vllm-arg` 覆寫：

- model identity、host、port、served model name
- TP/DP、GPU memory、context、sequence/token budget、KV dtype
- load format、runner、reasoning/tool parser、chat template
- attention backend 與 Hugging Face overrides
- `--device-ids`、`--async-scheduling`、`--enable-flashinfer-autotune`、`--speculative-config`
- `--enable-chunked-prefill`、`--enable-prefix-caching`、`--enable-auto-tool-choice`

Protected flags 的 underscore 拼法、`--flag=value` 與 `--no-<flag>` 形態同樣被拒絕。

目前 catalog 的生成模型使用 V1 chunked prefill default，已移除重複的 `--enable-chunked-prefill`。Prefix caching 的 default 依模型能力與 hybrid 架構決定；Qwen hybrid 模型保留 `enable_prefix_caching = true`，維持遷移前明確啟用的行為。

Gateway 保留自己的 fallback：`gpu_memory_utilization = 0.90`、`max_num_seqs = 256`、`max_num_batched_tokens = 4096`。vLLM 0.27.1 的 GPU memory default 為 `0.92`；`SchedulerConfig` 的 seq/token dataclass default 為 `128` / `2048`，但 `EngineArgs` 啟動時會依裝置與 usage 自動調整。Catalog 或全域 override 仍優先於 gateway fallback。

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
VLLM_GRACEFUL_SHUTDOWN_TIMEOUT
VLLM_MODEL_IDLE_SECONDS
VLLM_SERVICE_LOG_LEVEL
```

部署用 companion script 支援更多 lifecycle 變數，例如 preload、PID/log path 與 stop timeout；該 script 位於 `~/.local/bin/vllm`，不由此 repository 安裝。

## 檔案放置

| 內容                                           | 位置                                                                        | 誰讀取                                |
| ---------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------- |
| 可版本控制的 model catalog                     | `models.toml` 或 `VLLM_CONFIG_FILE` 指定的檔案                              | `ModelCatalog`、companion script      |
| Python package metadata 與 locked dependencies | `pyproject.toml`, `uv.lock`                                                 | uv、build backend                     |
| 架構與開發規範                                 | `docs/`, 各層 `AGENTS.md`                                                   | 人員與 coding agents                  |
| Model 權重與 Hugging Face cache                | repository 外的本機 model path 或 Hugging Face cache                        | vLLM、huggingface-hub                 |
| Gateway PID/log                                | 預設 `~/.cache/vllm-gateway.pid`、`~/.cache/vllm-gateway.log`               | companion script                      |
| Python/test/build artifacts                    | `.venv/`, `__pycache__/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info` | 開發工具；不要提交                    |
| Agent harness state                            | `.agents/`, `.pi/`                                                          | agent tooling；已由 `.gitignore` 排除 |

## Secrets

Repository 沒有自建 API key 或 secret-file loader。Hugging Face token、代理憑證或其他部署 secret 應透過 process environment、Hugging Face credential store 或外部 secret manager 提供。不要把 token 寫進 `models.toml`、README、docs、test fixture 或 log 範例。

## 安全更新流程

1. 編輯 catalog 後先執行 catalog 與 command-generation tests。
2. 呼叫 `POST /admin/reload`；先檢查回應中的 `added`、`removed`、`updated`、`kept_loaded`。
3. 檢查 `/health` 和 `/v1/models`。需要載入權重時，再執行目標 model 的 GPU smoke test。
4. Reload 失敗時不要重啟來掩蓋 parser 問題；服務會保留舊 catalog，先修正回應與 log 指出的設定錯誤。
