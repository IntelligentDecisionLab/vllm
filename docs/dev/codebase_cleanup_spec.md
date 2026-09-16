# Codebase 清理 — 開發 Spec

狀態：已批准（2026-09-16），執行中（draft 0.2）

## 1. 背景與目標

Gateway 已全面改用 vLLM backend，但 repo 仍保留 SGLang/llama.cpp 時代的殘留：package 名稱 `sglang_service`、`SGLANG_*` env fallback、legacy INI parser、無 vLLM 對應的死欄位、多組未文件的 TOML/CLI 別名。另缺 CI 與 lint gate，文檔與 `models.toml` 實際內容漂移，companion script 不在 repo 內但測試依賴工作站固定路徑。

目標（依方向 A 切片執行，每片獨立可合併、合併時全綠）：

1. 建立自動化 gate：GitHub Actions（CPU-only）+ ruff（lint + format）+ pre-commit。
2. 文檔對齊程式碼事實。
3. 移除 SGLang/llama.cpp legacy：死欄位、INI parser、`SGLANG_*` env、未文件別名。
4. package 改名 `sglang_service` → `vllm_gateway`。
5. companion script 納入 repo 與 packaging，測試不再依賴工作站固定路徑。

執行 constraint：目前服務在 main checkout 執行中（editable install），所有開發在 worktree 進行，cutover 步驟見 §10。

## 2. 基線（2026-09-16 驗證）

- `uv run --extra test python -m pytest -q`：136 passed（3.94s，CPU-only）。
- `~/.local/bin/vllm` 存在（346 行 bash），`~/.local/bin/uv` 存在。
- repo 無 git remote、無 `.github/`、無 `gh` CLI；推 GitHub 是使用者手動步驟。
- 主 checkout 的 package 是 editable install（`_editable_impl_vllm_inference_service.pth` 指向 repo root）；merge 後需 `uv sync` 更新 console script 指向。
- 程式碼最長行 183；ruff `line-length` 訂 120。
- `SglangNamespace` / `SglangArgumentParser`（`cli.py`）的 legacy properties（`config`、`models_config`、`models_file`、`preset_file`）只有 tests 引用，production code 只用 `args.config_file`。
- `spec.reasoning` 不是死欄位：`app.py:_apply_model_defaults` 用 `reasoning == "off"` 注入 `enable_thinking: false`，README 有文件。保留。
- `spec.embeddings` / `spec.is_gguf` 有實際 command-generation 效果（`--runner pooling` / `--load-format gguf`），但當前 catalog 未使用，屬 SGLang/llama.cpp 時代路徑。
- companion script 的 `download` 子命令 import `sglang_service.catalog` 並用 `spec.hf_file` 做單檔下載；`huggingface-hub` dependency 主要供它使用（gateway 移除 `hf_file` 分支後仍保留此 dependency）。
- script 的 `pid_matches_gateway` 用 cmdline 子串匹配 `sglang_service`、`vllm-service`、`vllm-gateway`、`$PROJECT_DIR/main.py`。

## 3. 全域驗收標準

全部 slice 完成後：

1. `uv run --extra test python -m pytest -q` 綠（CPU-only）。
2. `uv run --extra dev ruff check .` 與 `uv run --extra dev ruff format --check .` 綠。
3. GitHub Actions 綠（repo 推送後）。
4. `rg -in "sglang" sglang_service vllm_gateway tests docs README.md AGENTS.md pyproject.toml` 零命中，唯一例外是 `docs/dev/vllm_config_migration_spec.md` 與 `docs/dev/codebase_cleanup_spec.md` 本身（歷史紀錄）。
5. `rg "SGLANG_" .`（排除 .venv、uv.lock、dist）零命中。
6. `uv run python -m vllm_gateway --help` 正常；`vllm-service --help` console script 正常。
7. cutover 後 `vllm restart` + GPU smoke test（`docs/development.md#gpu-smoke-test`）通過。

## 4. Worktree 與 branch 設定

```bash
git worktree add ../vllm-cleanup -b cleanup/vllm-identity
cd ../vllm-cleanup
uv sync --extra test          # slice 0 加入 dev extra 後改為 --extra test --extra dev
```

- 每個 slice 一個 branch（`slice-N-...`）或同一 branch 上一個 commit 群，綠了再進下一片。
- 有 GitHub remote 後每片開 PR；remote 設定前直接 merge 本地 main。
- 使用者手動步驟（slice 0 之前）：建立 GitHub public repo、`git remote add origin ...`、`git push -u origin main`。

## 5. Slice 0：CI + ruff gate

### 需求

- R0.1 `pyproject.toml` 加 `dev = ["ruff>=0.14"]` optional extra（`uv.lock` 同步更新）。
- R0.2 ruff 設定放 `pyproject.toml` 的 `[tool.ruff]`：`line-length = 120`、`target-version = "py312"`、`select = ["E", "F", "W", "I", "UP", "B"]`。若個別規則噪音過大，加 `[tool.ruff.lint.per-file-ignores]` 或縮小 select，不得整檔 `# noqa` 迴避。
- R0.3 執行 `ruff check --fix` + `ruff format`，手動解決殘留 violation。format 造成的一次性 reformat churn 接受，集中在此 slice。
- R0.4 `.github/workflows/ci.yml`：`push`（main）與 `pull_request` 觸發；`ubuntu-latest`；Python 3.12；steps：`actions/checkout` → `astral-sh/setup-uv`（enable python）→ `uv sync --extra test --extra dev` → `uv run --extra dev ruff check .` → `uv run --extra dev ruff format --check .` → `uv run --extra test python -m pytest -q`。
- R0.5 `.pre-commit-config.yaml`：`astral-sh/ruff-pre-commit` 的 `ruff-check`（`--fix`）與 `ruff-format` hooks，pin 固定 tag。
- R0.6 文檔跟進：`docs/development.md` 的「Repository 目前沒有 CI workflow，也沒有 Ruff、Black、mypy...」段改寫為 gate 現況（CI 觸發、本機執行方式、pre-commit 安裝）；root `AGENTS.md` Verification 的「Repository 沒有配置 lint/type-check gate」同步修正（type-check gate 仍無，不得宣稱有）。

### 檔案邊界

`pyproject.toml`、`uv.lock`、`.github/workflows/ci.yml`、`.pre-commit-config.yaml`、`docs/development.md`、`AGENTS.md`、以及 ruff 自動修正波及的任意 `.py`（format 噪音）。不動任何行為邏輯。

### 驗證

- `uv run --extra dev ruff check .`、`uv run --extra dev ruff format --check .` 綠。
- `uv run --extra test python -m pytest -q` 綠（136 個測試數不減）。
- Actions 綠（推送後確認）。

## 6. Slice 1：文檔漂移修正

### 需求

- R1.1 `README.md`：catalog 清單移除 `qwen3.6-27b`（已退役）；「目前 catalog 包含」改兩個 model。
- R1.2 `README.md` 的 catalog 範例 block 對齊 `models.toml` 現況：`[default]` 補 `async_scheduling`、`flashinfer_autotune`、`enable_prefix_caching`、`speculative_method = "mtp"`、`num_speculative_tokens = 2`；`["qwen3.8-27b"]` 補 `tensor_parallel_size = 2`、`max_num_seqs = 64`、`num_speculative_tokens = 3`；`["qwen3.5-9b"]` 改 `context_length = 131072`、`gpu_memory_utilization = 0.35`、`device_ids = [1]`、`max_num_seqs = 128`。
- R1.3 `docs/dev/vllm_config_migration_spec.md` 狀態行改為「已完成（2026-09-15 批准，2026-09-15 完成）」；內文是歷史紀錄，不改寫。
- R1.4 全庫掃一遍 docs 與程式碼的其餘事實性漂移（README/docs 宣告但 code 不存在的行為，或反之），逐條修正。不處理 slice 2/3 將改變的內容（INI、`SGLANG_*`、`sglang_service` 路徑），那些由各自 slice 跟進。

### 檔案邊界

`README.md`、`docs/`。零程式碼變更。

### 驗證

- 人工比對 README 範例與 `models.toml` 逐欄一致。
- pytest 綠（不該有影響，作為回歸防護）。

## 7. Slice 2：legacy 語意清理

### 7.1 移除清單（對照 §2 基線驗證）

**ModelSpec 欄位（`catalog.py`）**

| 欄位 | 現行接線 | 處置 |
| --- | --- | --- |
| `embeddings` | manager `--runner pooling`、`ModelCatalog.default` 的 `not spec.embeddings` 過濾、`public_dict` metadata `embedding` | 欄位與三處接線全刪 |
| `pooling` | `public_dict` metadata `pooling` | 刪 |
| `reasoning_budget` | `public_dict` metadata `reasoning_budget` | 刪 |
| `spec_type` | `public_dict` metadata `speculative_type` | 刪 |
| `hf_file` | `source` property、`resolve_model_path` 的 `hf_hub_download` 分支、parser | 欄位與分支全刪；`huggingface-hub` dependency 保留（companion script 的 `download` 用） |
| `is_gguf`（property） | manager `--load-format gguf`、`public_dict` metadata `format` | property、manager 分支、metadata key 全刪 |
| `reasoning` | `app.py:_apply_model_defaults`、README 文件 | **保留**；`app.py` docstring 的「llama.cpp preset」措辭改成中性的「catalog 請求預設」 |

**`_PROTECTED_VLLM_FLAGS`**：`--load-format` 與 `--runner` 保留在 protected set。gateway 不再輸出它們，但移除保護會等於透過 `vllm_args` 把 GGUF/pooling 功能開後門，與本 slice 的移除決定矛盾。

**catalog parser 別名 key（`from_toml_file`）**：只保留 `docs/configuration.md` typed fields 表列的底線正規 key。刪除以下別名與其 `_first()`/`or` 接線：

| 正規 key | 刪除的別名 |
| --- | --- |
| `model_path` | `model`、`path` |
| `hf_repo` | `hf-repo`、`repo_id` |
| `tensor_parallel_size` | `tp_size`、`tp` |
| `data_parallel_size` | `dp_size`、`dp` |
| `max_num_seqs` | `max_running_requests` |
| `max_num_batched_tokens` | `chunked_prefill_size` |
| `context_length` | `ctx_size`、`c`、`ctx-size` |
| `kv_cache_dtype` | `kv-cache-dtype`、`cache-type-k`、`cache_type_k`、`cache-type-v`、`cache_type_v`（`_parse_kv_cache_dtype` 正規化函式保留，只收正規 key） |
| `gpu_memory_utilization` | `mem_fraction_static`（deprecated alias，遷移已完成） |
| `idle_seconds` | `idle-seconds`、`sleep_idle_seconds`、`sleep-idle-seconds` |
| `rope_scaling` | `rope-scaling`、`rope_type` |
| `rope_scale` | `rope-scale`、`factor` |
| `yarn_orig_ctx` | `yarn-orig-ctx`、`orig-ctx`、`original_max_position_embeddings` |
| `rope_theta` | `rope-theta` |
| `reasoning_parser` / `tool_call_parser` / `chat_template` / `attention_backend` | 各別名的 hyphen spelling |
| `aliases` | `alias` |
| `default` | `is_default` |
| `vllm_args` | `extra_args`、`args` |
| `hf_file`（欄位刪除） | `hf-file` |
| `embeddings`（欄位刪除） | `embedding`、`is_embedding` |

TOML 結構方言（root named tables、`[models.<name>]` map、`[[models]]` array-of-tables）保留，`docs/configuration.md` 有文件。

**INI / preset（`catalog.py` + `cli.py`）**

- `_parse_preset_file()`、`ModelCatalog.from_preset_file()`、`from_file()` 的 `.ini` 分支、`ModelCatalog.preset_file` 屬性：全刪。`from_file()` 保留為 TOML 入口（`cli.py`、`app.py` reload、companion script 都叫它）。
- `cli.py`：`--config-file`、`--models-config`、`--preset-file` 三個 alias flag 刪除，保留 `--config`。
- `cli.py` `_default_config_file()` env chain 收斂為只有 `VLLM_CONFIG_FILE`（刪除 `VLLM_MODELS_FILE`、`VLLM_MODELS_CONFIG`、`VLLM_CONFIG`、`VLLM_PRESET_FILE`、`VLLM_PRESETS_FILE`、`SGLANG_CONFIG_FILE`、`SGLANG_MODELS_FILE`、`SGLANG_PRESETS_FILE`）。
- `cli.py`：`SglangNamespace` 與 `SglangArgumentParser` 兩個 class 整個刪除（properties 只有 tests 引用），`build_parser()` 直接用 `argparse.ArgumentParser`。

**companion script（`~/.local/bin/vllm`，repo 外，in-place 編輯）**

- 移除全部 `SGLANG_*` fallback（19 行 env 定義）與 `CONFIG_FILE` chain 的 `VLLM_MODELS_FILE`。
- `download` 子命令：移除 `spec.hf_file` 的 `hf_hub_download` 分支，只留 `snapshot_download`。
- 此步驟是 cutover 子步驟：不需重啟服務（running process 不受影響；script 只在下次呼叫時生效）。
- script 改動後 `test_catalog.py:test_companion_script_defaults` 的 content assertions 與 `tests/test_script.py` 必須同 commit 更新並通過。

### 需求

- R2.1 依 §7.1 表移除 ModelSpec 欄位與接線。
- R2.2 依 §7.1 表收斂 parser 別名 key。
- R2.3 移除 INI/preset 路徑與 cli alias。
- R2.4 in-place 更新 `~/.local/bin/vllm`（§7.1 script 項）。
- R2.5 tests 跟進：刪除 `test_ini_preset_preserves_default`、`test_cli_namespace_models_file`、`test_gpu_memory_deprecated_alias_precedence`、`test_parse_kv_cache_dtype_hyphenated`；`test_from_file_and_preset_file_compat` 刪除 `from_preset_file` 段並改名 `test_from_file`；`test_sglang_download_single_table_and_aliases` 移除 legacy alias case（保留正規結構 case）；`test_model_spec_public_dict`、`test_loads_models_toml`（`is_gguf` 斷言）、`test_cli_defaults`（`args.config`/`args.preset_file` → `args.config_file`）、`test_cli_env_vars`（SGLANG_*/VLLM alias env case）同步更新；`test_companion_script_defaults` 的四個 content assertion（`PROJECT_DIR`/`TP_SIZE`/`DP_SIZE`/`PORT` 行的 `SGLANG_*` fallback 模板字串）改寫為無 `SGLANG_*` 的新 script 行。
- R2.6 docs 跟進：`docs/configuration.md` 移除 `.ini` 相容性句、`SGLANG_*` 段（line 94）、`mem_fraction_static` deprecated alias 段、Identity 表的 `hf_file`、`vllm_args` 邊界段不受影響；「常用 typed fields」表維持底線 key。`README.md` 移除 line 147 的 `SGLANG_*` fallback 說明。`sglang_service/AGENTS.md` 的「parser 處理支援的 key spelling」慣例改為「parser 只接受 `docs/configuration.md` 列出的底線正規 key」。

### 檔案邊界

`sglang_service/catalog.py`、`sglang_service/cli.py`、`sglang_service/manager.py`、`sglang_service/app.py`（docstring 一行）、`tests/test_catalog.py`、`tests/test_script.py`、`docs/configuration.md`、`README.md`、`sglang_service/AGENTS.md`、`~/.local/bin/vllm`（repo 外，cutover 子步驟）。

### 驗證

- `uv run --extra test python -m pytest -q` 綠。
- `uv run --extra dev ruff check .` 綠。
- `bash -n ~/.local/bin/vllm` 通過；`vllm status` 對 running service 仍回報正常。
- `rg -n "SGLANG_|from_preset_file|_parse_preset_file|mem_fraction_static|is_gguf|spec_type|reasoning_budget|hf_file" sglang_service tests` 零命中；`rg -n "embeddings" sglang_service tests` 唯一允許的命中是 `manager.py` 的 HF override key `original_max_position_embeddings`（Hugging Face 設定檔的正式欄位名，不是 gateway 欄位）。

## 8. Slice 3：package 改名 `sglang_service` → `vllm_gateway`

### 需求

- R3.1 `git mv sglang_service vllm_gateway`。
- R3.2 `pyproject.toml`：`packages = ["vllm_gateway"]`；`vllm-service` / `vllm-gateway` 兩支 console script 改指 `vllm_gateway.cli:main`。
- R3.3 移除 `main.py`（entry shim，與 `python -m` 及 console script 重覆）。
- R3.4 `vllm_gateway/__init__.py` docstring「SGLang backends」→「vLLM backends」。
- R3.5 logger 名稱：`app.py`、`manager.py` 的 `logging.getLogger("sglang_service")` → `"vllm_gateway"`。
- R3.6 tests imports：`tests/test_catalog.py`、`tests/test_app.py`、`tests/test_manager.py` 的 `from sglang_service...` → `from vllm_gateway...`（含 function body 內的 local import，全檔替換）；`tests/test_script.py` 的 fake server 檔名 `sglang_service_fake.py` → `vllm-gateway-fake.py`（檔名需含 `vllm-gateway` 子串，companion script 的 `pid_matches_gateway` 在 cutover 前後都能認得這個 fake process）；`test_sglang_download_single_table_and_aliases`（若 R2.5 後仍存在）改名 `test_single_table_and_aliases`。
- R3.7 docs：`docs/architecture.md`（元件清單 line 11-17 的路徑與 entry 清單）、`docs/development.md`（entry points 段）、`README.md`（`python -m sglang_service` 三處、`main.py` 提及）、root `AGENTS.md`（Instruction Index、Verification 命令）、`sglang_service/AGENTS.md` → `vllm_gateway/AGENTS.md`（`git mv`，內文 line 3 路徑）。歷史 spec（`docs/dev/vllm_config_migration_spec.md`、本檔）不改寫。
- R3.8 companion script（repo 外，cutover 時 in-place 編輯，不在本 slice 的 worktree commit 內）：`start()` 三處 `python -m sglang_service` → `python -m vllm_gateway`；`pid_matches_gateway` 的 `sglang_service` pattern → `vllm_gateway`、移除 `main.py` pattern；`download` 子命令內 `from sglang_service.catalog import ModelCatalog` → `from vllm_gateway.catalog import ModelCatalog`。cutover 順序見 §10。

### 檔案邊界

`vllm_gateway/`（整目錄）、`pyproject.toml`、`main.py`（刪）、`tests/`、`README.md`、`docs/architecture.md`、`docs/development.md`、`AGENTS.md`、`vllm_gateway/AGENTS.md`。

### 驗證

- `uv run --extra test python -m pytest -q` 綠。
- `uv run python -m vllm_gateway --help` 正常；worktree 內 `uv sync` 後 `uv run vllm-service --help` 正常。
- `rg -in "sglang" vllm_gateway tests README.md AGENTS.md docs pyproject.toml` 只命中兩份歷史 spec。

## 9. Slice 4：companion script 納入 repo + packaging 收尾

### 需求

- R4.1 repo 加 `scripts/vllm`：內容 = §7.1 + R3.8 清理後的 script（可執行位 755 入 git）。
- R4.2 `tests/test_script.py`：`SCRIPT` 改指 `Path(__file__).parent.parent / "scripts" / "vllm"`；三個測試不改邏輯，只跟隨路徑。
- R4.3 `tests/test_catalog.py:test_companion_script_defaults`：改讀 repo 路徑。
- R4.4 `pyproject.toml`：移除 hatch wheel 的 `force-include models.toml`。
- R4.5 文檔：`README.md` 的「`~/.local/bin/vllm` 管理工具」段改為「script 在 `scripts/vllm`，首次安裝 `install -m 755 scripts/vllm ~/.local/bin/vllm`」；`docs/development.md` 的 Companion script 段（「Script 不在 repository 內...」）改寫為 repo 內版本；`tests/AGENTS.md` 的 `test_script.py` 描述同步。
- R4.6 cutover：用 repo 版本覆寫 `~/.local/bin/vllm`（與 in-place 版本應已一致，覆寫作確認）。

### 檔案邊界

`scripts/vllm`（新）、`tests/test_script.py`、`tests/test_catalog.py`、`pyproject.toml`、`README.md`、`docs/development.md`、`tests/AGENTS.md`、`~/.local/bin/vllm`（cutover）。

### 驗證

- `uv run --extra test python -m pytest -q` 綠（含 `test_script.py` 對 repo 版本 script 的 lifecycle 測試）。
- `uv build` 產出的 wheel 不含 `models.toml`（`python -m zipfile -l dist/*.whl` 確認）。
- 最終 cutover 後：`vllm restart` + GPU smoke test（§3 第 7 項）。

## 10. Cutover 與操作步驟

服務在 main checkout 以 editable install 執行中。worktree 開發期間完全不受影響。cutover 只在 slice 3 merge 時發生一次（改名使舊入口消失）：

1. Slice 3 PR 綠後 merge 回 main。
2. 用**舊** `~/.local/bin/vllm` 執行 `vllm stop`（環境完全未動，舊 script 的 match pattern 認得 running process）。先停止再動程式碼，避免 stop 期間遇到 checkout/venv 半更新的狀態。
3. 主 checkout：`git pull`、`uv sync --extra runtime --extra test`（更新 editable mapping 與 console script 指向）。
4. In-place 更新 `~/.local/bin/vllm`（R3.8 的兩處改動）。
5. `vllm start`（新入口 `python -m vllm_gateway`）。
6. GPU smoke test（`docs/development.md#gpu-smoke-test`）確認服務正常。
7. 後續 slice 4 完成後，用 `scripts/vllm` 覆寫部署版（R4.6）。

停機窗：步驟 2 到步驟 5，預計 1-2 分鐘（`git pull` 與 `uv sync` 是增量操作）。風險：窗口內服務不可用。緩解：cutover 排在低使用時段，且步驟 2-5 一次做完。窗口結束後舊入口徹底消失，舊 script 已更新，不再有半狀態。slice 2 的 script in-place 編輯（§7.1）不需重啟，可獨立執行，不在停機窗內。

## 11. 明確不做

- 功能類：auth/API key、metrics/observability、GPU placement 自動化、多副本負載平衡。
- `vllm==0.27.1` pin 與 Python 3.12 不動。
- manager 的 lifecycle/lease/drain 邏輯不動（清理過程若發現既有 bug，停下來回報，不順手修）。
- type checker（mypy/pyright）：本輪不上，ruff 穩定後另評估。
- PyPI 發布設定；wheel 仍只是本機 build 驗證用途。
- 不改寫兩份歷史 spec 的內文。
- GGUF 與 pooling/embedding 模型支援（隨死欄位移除）；需要時另立專案重建 typed 支援。
