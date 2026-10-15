# AI 協作入口

本檔是 repository 的 AI 協作入口。先讀本檔；工作落在有 `AGENTS.md` 的子目錄時，再讀該目錄的文件。越靠近目標檔案的 `AGENTS.md` 優先級越高。每層文件只規範自己的範圍，並由該層的 Instruction Index 指向下一層入口。

## Repository Scope

本專案是 Python 3.12 的 OpenAI-compatible multi-model gateway：它解析 `models.toml`，按需管理獨立 vLLM API server subprocess，並以 FastAPI/httpx 轉送 request。它不是 vLLM inference engine、模型權重 repository 或 CUDA kernel 專案；上游 vLLM 負責 scheduler、模型執行與 GPU kernel。完整元件與 request lifecycle 見 [`docs/architecture.md`](docs/architecture.md)。

## 設定與檔案放置

- 可版本控制的 model catalog 放在 root [`models.toml`](models.toml)；外部部署可用 `VLLM_CONFIG_FILE` 指向其他檔案。
- Python dependencies 與 lock state 分別維護在 `pyproject.toml` 和 `uv.lock`。
- 詳細設計與 workflow 放 `docs/`；`AGENTS.md` 只保留協作入口與局部規則。
- PID、log、model cache、credential、virtualenv、test cache 和 agent state 不放入版本控制。
- 完整 precedence、protected vLLM options、secrets 與 artifact placement 見 [`docs/configuration.md`](docs/configuration.md)。

## Instruction Index

目前直接子目錄入口：

- [`vllm_gateway/AGENTS.md`](vllm_gateway/AGENTS.md)：gateway package 的 catalog、proxy、backend lifecycle 與 command-generation 規範。
- [`tests/AGENTS.md`](tests/AGENTS.md)：測試分區、async fake-process 慣例與 companion script 驗證。

未來候選：

- `docs/`：文件出現獨立產生流程、專屬檢查或更深層結構後，再新增局部入口；目前三份手寫文件不需要額外規範。

維護規則：目錄變複雜後，在該目錄新增 `AGENTS.md`，並回來更新最近一層祖先 `AGENTS.md` 的 Instruction Index。每層只列自己的直接子目錄，不枚舉整棵樹。

## Working Rules

- 修改前先讀 `git status` 和相關 diff。保留不相干的 dirty changes，不 reset、不覆蓋，也不把它們混入自己的修改。
- 維持 `app.py`、`catalog.py`、`manager.py`、`cli.py` 的既有責任邊界與命名；不要順手重構相鄰程式。
- 設定或 command option 要有單一 owner。Gateway-managed vLLM option 使用 typed field 與明確 precedence，不得塞進 `vllm_args` 規避 validation。
- 修改 lifecycle 或 proxy 時，先列出 state/lease 不變條件，再補對應測試。Foreign process ownership 與 active request draining 是安全邊界。
- Repository 文件使用繁體中文；程式 identifier、CLI flag 與 protocol 名稱保留原文。
- 詳細開發流程與測試對應見 [`docs/development.md`](docs/development.md)。

## Verification

安裝：

```bash
uv sync --extra runtime --extra test
```

完整單元測試：

```bash
uv run --extra test python -m pytest -q
```

CLI 與 package：

```bash
uv run python -m vllm_gateway --help
uv build
```

Repository 已配置 Ruff lint、format 與 pytest gate；type-check gate 仍未配置。只有 runtime、catalog 或 backend command 變更需要另做 GPU smoke test；命令與驗收項目見 [`docs/development.md`](docs/development.md#gpu-smoke-test)。

手動重現 backend command 除錯時（直接跑 vLLM，不經 gateway），環境必須有 `CUDA_HOME=/usr/local/cuda` 且 `PATH` 含 `/usr/local/cuda/bin`：本機 `/usr/bin/nvcc` 是 CUDA 12.0，flashinfer JIT 偵測 CUDA arch 時會對 SM 12.x GPU 誤報 `SM 12.x requires CUDA >= 12.9`，表徵為 `FlashInfer requires GPUs with sm75 or higher`——錯誤訊息指向 GPU 架構，真正的問題是偵測到舊版 nvcc。Gateway 自建的 backend subprocess 不受影響（`manager.py` 已設定 `CUDA_HOME` 與 PATH）。
