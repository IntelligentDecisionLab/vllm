# Gateway package 協作規範

本檔只適用於 `vllm_gateway/`。先遵守 root [`AGENTS.md`](../AGENTS.md)，再套用以下局部規範。元件互動、request flow、state machine 與不變條件見 [`docs/architecture.md`](../docs/architecture.md)。

## Local Architecture

- `cli.py` 只負責解析設定、組裝 catalog/manager/downloads/app 與啟動 Uvicorn。
- `catalog.py` 負責 HF cache 掃描、`config.json` 解析（family profile、KV 大小、MTP）、`ModelSpec` 與 alias 持久化；不要讓它管理 process 或讀 GPU。
- `memory.py` 只做記憶體估算與 GPU 記憶體查詢；估算常數（`KV_FULL_SEQS`、`KV_MARGIN`、`OVERHEAD_BYTES`、`SAFETY_BYTES`）集中在此檔。
- `manager.py` 是 backend process、port、admission、eviction、lease 與 lifecycle 的唯一 owner，也是所有寫死 vLLM 參數的位置。
- `downloads.py` 只負責背景下載與磁碟檢查；不要讓它碰 backend。
- `app.py` 負責 HTTP contract、alias routing 與 response streaming；不要在 route handler 自行啟停 process。
- `__main__.py` 只保留 module entry point。

## Local Conventions

- `ModelSpec` 維持 frozen/slots dataclass。新增欄位時同步 `ModelCatalog` 的解析與持久化、alias API、`_command()`（如影響 backend 參數）、`plan_memory()`（如影響記憶體）與 tests。
- Alias 只接受精確比對；不要加入 repo id、正規化名稱或大小寫不分的 lookup。
- `ModelManager._lock` 保護 backend map、state、active request count 與 port ownership。不要在持有 lock 時等待 subprocess、HTTP、NVML 或長時間工作；使用 task 和 condition 協調。
- `_startup_gate` 序列化「量測 free memory → 驅逐 → 啟動」。它由 start task 的 done callback 釋放，不能依賴 task 內的 `finally`（task 可能在第一步前就被 cancel）。
- 驅逐只能選 `READY` 且 `active_requests == 0` 的 backend。
- 每個成功取得的 `BackendLease` 必須在普通 response、SSE、HTTP error、cancellation 與 exception 路徑釋放一次。
- Readiness 必須驗證 backend `/health` 和 `/v1/models` 中的 alias，不能只檢查 port 是否開啟。
- 停止程序時只操作 manager 建立的 process group。不要加入全機 process scan 或對 foreign process 發 signal。
- 錯誤回應不得包含 backend 原始輸出、路徑或 exception 文字；使用 `diagnostics.py` 的固定訊息與 diagnostic ID。
- 設定與記憶體公式見 [`docs/configuration.md`](../docs/configuration.md)。

## Instruction Index

本目錄目前沒有包含 `AGENTS.md` 的直接子目錄。新增子 package 且需要獨立架構或 workflow 時，在該子目錄建立入口，並更新本節。

## Local Verification

Catalog、memory 或下載：

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py tests/test_memory.py tests/test_downloads.py
```

Proxy 或 lifecycle：

```bash
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
```

跨模組改動完成後，再執行 root 的完整測試。需要啟動真實模型的變更依 [`docs/development.md`](../docs/development.md#deploying-and-the-gpu-smoke-test) 執行 GPU smoke test。
