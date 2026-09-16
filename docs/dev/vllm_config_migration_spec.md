# vLLM 設定欄位遷移 — 開發 Spec

狀態：已完成（2026-09-15 批准，2026-09-15 完成）

## 1. 背景與目標

Gateway 最初為 SGLang backend 設計，現已全面改用 vLLM（`pyproject.toml` 固定 `vllm==0.27.1`，已安裝於 `.venv`）。`models.toml` 與 `ModelSpec` 仍保留 SGLang 時代的欄位名稱，且多個 vLLM 原生參數沒有 typed field，只能塞進 `vllm_args` 字串陣列（使用者描述的 "temporary vllm_kwargs 寫法"）。

目標：

1. 每個 gateway 管理的 vLLM 參數有單一 owner（`ModelSpec` typed field + 明確 precedence），符合 `sglang_service/AGENTS.md` 的受管理參數規範。
2. `vllm_args` 只保留 gateway 確實不管理的上游參數。
3. 欄位名稱與 vLLM 語彙對齊，不再以 SGLang 名稱誤導語義。

## 2. 審查結果（vLLM 0.27.1 ground truth）

方法：直接 introspect 本機 `.venv` 安裝的 vLLM 0.27.1（argument parser、`SchedulerConfig`、`CacheConfig`、`SpeculativeConfig`、attention backend registry、tool/reasoning parser registry），再以 vLLM 官方 Qwen3.5/3.6 recipe 與 Qwen3-Next recipe 交叉驗證。本地安裝版是 primary authority（比 web 文件更精確對應部署版本）。

### 2.1 `models.toml` 現行欄位對照

| models.toml 現況 | vLLM 0.27.1 狀態 | 結論 |
| --- | --- | --- |
| `context_length` → `--max-model-len` | ✓ 存在 | OK，保留 |
| `kv_cache_dtype` → `--kv-cache-dtype`（choices 含 `fp8_e4m3`） | ✓ 存在，default `auto` | OK，保留 |
| `tensor_parallel_size` / `data_parallel_size` → 同名 flag | ✓ 存在（dp default 1；dp>1 時 world_size 倍增） | OK，保留 |
| `mem_fraction_static` → `--gpu-memory-utilization`（default **0.92**，GPU 總記憶體比例） | ✓ flag 存在，✗ 名稱與語義是 SGLang 的 | **改名**。SGLang `mem-fraction-static`＝權重+KV 的 static 分配比例；vLLM `gpu-memory-utilization`＝整個 process 可動用 HBM 的比例。同名不同義 |
| `max_num_seqs` → `--max-num-seqs`（vLLM default 128；gateway fallback 256） | ✓ | OK，保留 |
| `max_num_batched_tokens` → `--max-num-batched-tokens`（vLLM default 2048；gateway fallback 4096） | ✓ | OK，保留（fallback 差異記錄於 2.3） |
| `idle_seconds` | vLLM 無對應；gateway 內部 idle reaper | OK（gateway 功能，非 vLLM 參數） |
| `reasoning_parser = "qwen3"` → `--reasoning-parser` | ✓ `qwen3` 已在 registry | OK，保留 |
| `tool_call_parser = "qwen3_coder"` → `--enable-auto-tool-choice --tool-call-parser` | ✓ `qwen3_coder` 已在 registry | OK，保留 |
| `attention_backend = "flashinfer"` → `--attention-backend` | ✓ 值會先 upper 再查 enum（`AttentionBackendEnum[value.upper()]`），小寫可用 | OK，保留 |
| `chat_template` → `--chat-template` | ✓ | OK（目前未使用） |
| `rope_*` → `--hf-overrides` | ✓ flag 可用 | OK（目前未使用；注意 Qwen3.5 長 context 官方範例用 `text_config.rope_parameters` 形態，與現有 top-level `rope_scaling` 映射不同，本次不處理） |
| `vllm_args` 內 `--enable-chunked-prefill` | V1 engine `enable_chunked_prefill` **default True** → no-op | **移除**（V1 恒為 chunked prefill） |
| `vllm_args` 內 `--enable-prefix-caching` | V1 engine `enable_prefix_caching` **default True** → no-op | **移除** |
| `vllm_args` 內 `--async-scheduling` | ✓ 真實（`SchedulerConfig.async_scheduling`，default None=auto） | **提升為 typed field** |
| `vllm_args` 內 `--enable-flashinfer-autotune` | ✓ 真實（`KernelConfig.enable_flashinfer-autotune`，kernel warmup） | **提升為 typed field** |
| `vllm_args` 內 `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` | ✓ 真實；`mtp` 是合法 method（`SpeculativeConfig.MTPModelTypes` 含 `mtp`、`qwen3_5_mtp`、`qwen3_next_mtp`）；官方 recipe 建議 `num_speculative_tokens` 1–5 | **提升為 typed fields**（`speculative_method` + `num_speculative_tokens`） |
| `vllm_args` 內 `--device-ids "1"`（9B entry） | ✓ **0.27.1 存在**（較新版本加入；comma-separated 實體 GPU id/UUID；保留 GPU topology visibility 供 GPU-NIC affinity 與 DeepGEMM；Ray executor 下無效） | **提升為 typed field**（`device_ids`） |

### 2.2 死欄位 / legacy（本次不處理，記錄備查）

| 欄位 | 狀態 |
| --- | --- |
| `reasoning`（bool）、`spec_type` | 無 vLLM 對應；僅出現在 `public_dict` metadata。SGLang 殘留，保留 |
| `embeddings` / `pooling` / `is_gguf` / `--load-format gguf` / `hf_file` 單檔下載 / `.ini` preset dialect | SGLang/llama.cpp 時代路徑；vLLM 仍支援 `--runner pooling` 與 `--load-format gguf`，目前 catalog 未使用。保留 |
| `cli.py` 與 companion script 的 `SGLANG_*` env fallback | 既有相容性，保留 |

### 2.3 觀察（非 code 變更，建議使用者複查）

1. **9B 高並發 entry 的 MTP**：官方 Qwen3.5 recipe 明言 MTP 以 throughput 換 latency，高並發下 speculative tokens 吃掉 KV cache 會降低有效 batch size。9B 批次實驗 entry 用 `num_speculative_tokens = 2` 與「高並發/高吞吐」目標相悖，建議實測比較後決定是否降為 1 或移除。本 spec 只改寫法、不改值。
2. **Gateway fallback 與 vLLM 0.27.1 內建 default 不同**：`gpu-memory-utilization` 0.90 vs 0.92、`max-num-seqs` 256 vs 128、`max-num-batched-tokens` 4096 vs 2048。保留 gateway 的顯式 fallback（單一 owner），僅在 docs 記錄差異。
3. **`_validate_extra_args` 不認 `--no-X` 形態**：protected set 以 `--flag` 字面比對，`--no-enable-prefix-caching` 這種 negation 形態會繞過保護（`argparse` FlexibleArgumentParser 會產生 `--no-` 變體）。U2 順帶修正。

## 3. 範圍

### In

- 新增 typed fields：`device_ids`、`async_scheduling`、`flashinfer_autotune`、`speculative_method`、`num_speculative_tokens`
- 改名：`mem_fraction_static` → `gpu_memory_utilization`（parser 保留舊名為 deprecated alias）
- `models.toml` 移除兩個 V1 no-op flag；`vllm_args` 內容全部改為 typed fields（值不變）
- protected set 擴充 + `--no-X` 形態識別
- `models.toml`、`README.md`、`docs/configuration.md` 同步
- 測試對齊 working-tree `models.toml`（baseline 11 red → green）

### Not-Doing

- 不為新欄位新增 CLI flag / 全域 env override（只作 per-model catalog field；日後需要再加）
- 不移除 legacy 路徑（INI preset、GGUF、`embeddings`/`pooling`、`hf_file` 單檔）
- 不改 `rope_*` 映射語義（未使用；Qwen3.5 `rope_parameters` 形態另行處理）
- 不改 `reasoning` / `spec_type`（保留 metadata）
- 不改任何效能調校值（MTP tokens、mem fraction、TP、seqs…），只改寫法
- 不跑 GPU smoke test（command-generation 層變更；真實 model 載入驗收由使用者依 `docs/development.md` 執行）

## 4. Units

依序執行：U1 → U2。每個 unit 完成時整個 test suite 必須 green。

### U1 — 測試對齊 working-tree `models.toml`（解鎖 baseline）

現狀：`uv run --extra test python -m pytest -q` 為 **11 failed / 58 passed**。全部失敗來自使用者未提交的 `models.toml` 變更（移除 `qwen3.6-27b`；27B TP 1→2；9B `context_length` 32768→131072、`idle_seconds` 300→0.0、`mem_fraction_static` 0.35）未同步到測試。

- **檔案邊界**：只改 `tests/test_catalog.py`（11 個失敗測試全部在此檔）。不動 `sglang_service/` 任何 code、不動 `models.toml`。
- **驗收標準**：
  1. 11 個失敗測試全部更新為反映 working-tree catalog（模型清單只剩 `qwen3.8-27b`、`qwen3.5-9b`；27B `tensor_parallel_size == 2`；9B `context_length == 131072`、`idle_seconds == 0.0`；相關 default/alias/capacity 斷言同步）。
  2. 測試只修斷言，不刪測試、不弱化既有 coverage（每個原斷言都要有新對應）。
- **驗證**：`uv run --extra test python -m pytest -q` → 全綠。

### U2 — vLLM typed-field 遷移

- **檔案邊界**：`sglang_service/catalog.py`、`sglang_service/manager.py`、`models.toml`、`README.md`、`docs/configuration.md`、`tests/test_catalog.py`、`tests/test_manager.py`。
- **驗收標準**（逐條可驗證）：
  1. `ModelSpec` 新增 frozen dataclass 欄位：`device_ids: tuple[int, ...] | None = None`、`async_scheduling: bool | None = None`、`flashinfer_autotune: bool | None = None`、`speculative_method: str | None = None`、`num_speculative_tokens: int | None = None`；`mem_fraction_static` 改名為 `gpu_memory_utilization`（parser 仍接受舊 key 作 alias，寫 `gpu_memory_utilization` 時以新名為 primary）。
  2. `from_toml_file` 解析上述新 key（`device_ids` 接受 TOML int 陣列；bool/int 直接型別化，不經字串）；`[default]` 合併路徑同樣生效。
  3. `manager._command` 輸出（只在欄位設定時輸出）：
     - `device_ids` → `--device-ids "0,1"`（comma-joined）
     - `async_scheduling = true` → `--async-scheduling`
     - `flashinfer_autotune = true` → `--enable-flashinfer-autotune`
     - `speculative_method`（＋ `num_speculative_tokens` 若設定）→ `--speculative-config '{"method": "...", "num_speculative_tokens": N}'`
  4. `mem_fraction_static` → `gpu_memory_utilization` 改名貫穿：`ModelManager._command` 的 fallback 鏈改讀新欄位（fallback 值 0.90 不變）；`public_dict` 與 API contract 不受影響（目前不暴露該欄位）。
  5. `_PROTECTED_VLLM_FLAGS` 新增：`--device-ids`、`--async-scheduling`、`--enable-flashinfer-autotune`、`--speculative-config`、`--enable-chunked-prefill`、`--enable-prefix-caching`、`--enable-auto-tool-choice`；`_validate_extra_args` 額外識別 `--no-<flag>` 形態（normalization 時 strip 前導 `no-`），使 negation 形態同樣被拒。
  6. `models.toml`（以 working-tree 版本為基底，保留使用者未提交變更）：
     - 移除全部 `--enable-chunked-prefill`、`--enable-prefix-caching`
     - 原 `vllm_args` 其餘內容改寫為 typed fields，**值不變**（`[default]`：`async_scheduling = true`、`flashinfer_autotune = true`、`speculative_method = "mtp"`、`num_speculative_tokens = 2`；27B：tokens 3；9B：`device_ids = [1]`）
     - `mem_fraction_static` 全改 `gpu_memory_utilization`（值不變）
     - 遷移後 `vllm_args` 應為空或整段移除
  7. `docs/configuration.md`：typed field 表格更新（新欄位類別、`gpu_memory_utilization` 取代 `mem_fraction_static` 並保留舊名 alias 說明）、protected 清單同步、V1 no-op 說明與 2.3 的 fallback 差異記錄。
  8. `README.md`：範例設定同步新欄位名稱（保留 README 既有值）。
  9. 測試（依 `tests/AGENTS.md` 慣例：config field 變更至少加 parsing + generated command assertion；protected option 測 hyphen/underscore spelling）：
     - 每個新欄位：TOML parsing 斷言 + 生成命令斷言
     - `mem_fraction_static` 舊 key 仍被接受（compat 回歸測試）
     - protected rejection：新 flag 的 hyphen 形態、underscore 形態、`--no-` 形態各一
- **驗證**：
  - `uv run --extra test python -m pytest -q` → 全綠
  - `uv run python -m sglang_service --help` → 正常（CLI 面不變）

## 5. 執行順序與約束

1. U1 → U2 依序；U2 以 U1 的 green baseline 為起點。
2. **working-tree 未提交變更（`models.toml`、`README.md`、`docs/configuration.md`）是使用者目前 intent**：U1 以它為測試目標，U2 以它為遷移基底。任何 unit 不得 revert 這些 hunk；其餘 dirty 區域（如 README 的 9B 範例值 `0.30`、configuration.md 的 incident 記錄）原樣保留。
3. 不觸碰 `app.py`、`cli.py`（本次無 API/CLI 變更）；`cli.py` 的 `VLLM_GPU_MEMORY_UTILIZATION` env 已對應 `ModelManager.gpu_memory_utilization` kwarg，改名不涉及。
4. 每個 unit 由獨立 general subagent 實作，brief 引用本 spec 檔路徑 + unit 編號 + 驗收標準條款。

## 6. 驗證指令

```bash
# 完整單元測試（每個 unit 完成後必跑）
uv run --extra test python -m pytest -q

# CLI 與 package（U2 完成後）
uv run python -m sglang_service --help
uv build
```

## 7. 風險與備註

- `vllm_args` protected set 擴充會拒絕既有部署把 no-op flag 放進 `vllm_args` 的寫法；這些 flag 在 V1 本就是 no-op，拒絕是安全行為（與 `--attention-backend` incident 的處理原則一致）。
- `device_ids` + `tensor_parallel_size` 的真實 GPU 組合（如 27B TP=2 要配 `device_ids = [0,1]`）只能在 GPU smoke test 驗證，屬使用者驗收範圍。
- 若 0.27.1 的 `--device-ids` 在部署環境行為與 introspect 結果不符（例如 driver 層限制），回退方案：改以 `CUDA_VISIBLE_DEVICES` 環境變數注入（`_start_backend` env 建構處），欄位名稱不變。
