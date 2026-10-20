# AI 協作入口

本檔是 repository 的 AI 協作入口。先讀本檔；工作落在有 `AGENTS.md` 的子目錄時，再讀該目錄的文件。越靠近目標檔案的 `AGENTS.md` 優先級越高。每層文件只規範自己的範圍，並由該層的 Instruction Index 指向下一層入口。

## Repository Scope

本專案是 Python 3.12 的 OpenAI-compatible 共用 gateway，部署在內網 server 的 Docker container，只使用一張 GPU。它掃描共用 Hugging Face cache 中的模型，讓使用者建立 alias（模型加 context length），為每個 alias 按需啟動獨立的 vLLM API server subprocess，並以 FastAPI/httpx 轉送 request。它不是 vLLM inference engine、模型權重 repository 或 CUDA kernel 專案；上游 vLLM 負責 scheduler、模型執行與 GPU kernel。完整元件與 request lifecycle 見 [`docs/architecture.md`](docs/architecture.md)。

## 設定與檔案放置

- 使用者只能決定模型與 context length；其他 vLLM 參數寫死在 `vllm_gateway/manager.py`，parser 與 MTP 依模型 `config.json` 判斷，GPU 記憶體份額由 `vllm_gateway/memory.py` 估算。
- Alias 存在 `$VLLM_STATE_DIR/aliases.json`；模型來自 `HF_HUB_CACHE`；部署設定在 `.env`（不入版控，範本是 `.env.example`）。
- Python dependencies 與 lock state 分別維護在 `pyproject.toml` 和 `uv.lock`。
- 詳細設計與 workflow 放 `docs/`；`AGENTS.md` 只保留協作入口與局部規則。
- `.env`、HF token、model cache、log、virtualenv、test cache 和 agent state 不放入版本控制。
- 完整設定、記憶體公式與 secrets 規範見 [`docs/configuration.md`](docs/configuration.md)。

## Instruction Index

目前直接子目錄入口：

- [`vllm_gateway/AGENTS.md`](vllm_gateway/AGENTS.md)：gateway package 的 catalog、memory、proxy、backend lifecycle 與 command-generation 規範。
- [`tests/AGENTS.md`](tests/AGENTS.md)：測試分區與 async fake-process 慣例。

未來候選：

- `docs/`：文件出現獨立產生流程、專屬檢查或更深層結構後，再新增局部入口。`docs/dev/` 是早期工作站版本的歷史 spec，內容（`models.toml`、`scripts/vllm`、API key）已不適用，不要當作現況依據。

維護規則：目錄變複雜後，在該目錄新增 `AGENTS.md`，並回來更新最近一層祖先 `AGENTS.md` 的 Instruction Index。每層只列自己的直接子目錄，不枚舉整棵樹。

## Working Rules

- 修改前先讀 `git status` 和相關 diff。保留不相干的 dirty changes，不 reset、不覆蓋，也不把它們混入自己的修改。
- 維持 `app.py`、`catalog.py`、`memory.py`、`manager.py`、`downloads.py`、`cli.py` 的既有責任邊界與命名；不要順手重構相鄰程式。
- 開放給使用者的選項必須有單一 owner：`ModelSpec` 欄位、alias API、`ModelManager._command()` 和測試一起改。不要提供任意 vLLM 參數的 passthrough。
- 修改 lifecycle、admission 或 proxy 時，先對照 [`docs/architecture.md`](docs/architecture.md) 的不變條件 I1 到 I6，再補對應測試。Foreign process ownership 與 active request draining 是安全邊界。
- Gateway 沒有認證，只能放在內網。不要加入依賴 client IP 的判斷（Docker NAT 後看不到真實來源）。
- 部署到共用 server 時只 clone 到新目錄，不修改他人的 checkout；不對 GPU 上不屬於 gateway 的 process 發 signal。
- 公開文件（README、`docs/*.md`）使用英文；`AGENTS.md` 使用繁體中文。程式 identifier、CLI flag 與 protocol 名稱保留原文。
- 詳細開發流程與測試對應見 [`docs/development.md`](docs/development.md)。

## Verification

安裝與完整單元測試：

```bash
uv sync --extra test --extra dev
uv run --extra test python -m pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

CLI 與 package：

```bash
uv run python -m vllm_gateway --help
uv build
```

Repository 已配置 Ruff lint、format 與 pytest gate；type-check gate 仍未配置。變更 command generation、記憶體估算或 image 時，需在目標 GPU 做 smoke test，流程見 [`docs/development.md`](docs/development.md#deploying-and-the-gpu-smoke-test)。

手動在 container 內重現 backend command 時，`CUDA_HOME=/usr/local/cuda` 且 `PATH` 含 `/usr/local/cuda/bin`。Blackwell（SM 12.x）的 FlashInfer JIT 需要 CUDA 12.9 以上的 nvcc，image 內是 CUDA 13.0；若在 host 上直接跑，`/usr/bin/nvcc` 可能是舊版，表徵為 `FlashInfer requires GPUs with sm75 or higher`，錯誤訊息指向 GPU 架構，真正的問題是偵測到舊版 nvcc。
