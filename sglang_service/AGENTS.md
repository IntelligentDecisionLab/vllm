# Gateway package 協作規範

本檔只適用於 `sglang_service/`。先遵守 root [`AGENTS.md`](../AGENTS.md)，再套用以下局部規範。元件互動、request flow、state machine 與 reload semantics 見 [`docs/architecture.md`](../docs/architecture.md)。

## Local Architecture

- `cli.py` 只負責解析設定、組裝 catalog/manager/app 與啟動 Uvicorn。
- `catalog.py` 負責輸入格式、`ModelSpec`、identity lookup 和 model source resolution；不要讓它管理 process。
- `manager.py` 是 backend process、port、capacity、lease 與 lifecycle 的唯一 owner。
- `app.py` 負責 HTTP contract、localhost management boundary、canonical routing 與 response streaming；不要在 route handler 自行啟停 process。
- `__main__.py` 只保留 module entry point。

## Local Conventions

- `ModelSpec` 維持 frozen/slots dataclass。新增 catalog 欄位時，同步 parser、manager precedence、public metadata（若屬 API contract）和 tests。
- Gateway-managed vLLM flags 必須留在 `_PROTECTED_VLLM_FLAGS`。曾因 `--attention-backend` 同時出現在 `vllm_args` 和 manager contract，導致真實 model load 被 validation 拒絕；正式欄位是 `ModelSpec.attention_backend`。
- `ModelManager._lock` 保護 backend map、state、active request count 與 port ownership。不要在持有 lock 時等待 subprocess、HTTP 或長時間工作；使用 task 和 condition 協調。
- 每個成功取得的 `BackendLease` 必須在普通 response、SSE、HTTP error、cancellation 與 exception 路徑釋放一次。
- Readiness 必須驗證 backend `/health` 和 `/v1/models` 中的 canonical model identity，不能只檢查 port 是否開啟。
- 停止程序時只操作 manager 建立的 process group。不要加入全機 process scan 或對 foreign listener 發 signal。
- Config precedence 與受管理欄位流程見 [`docs/configuration.md`](../docs/configuration.md)。

## Instruction Index

本目錄目前沒有包含 `AGENTS.md` 的直接子目錄。新增子package 且需要獨立架構或 workflow 時，在該子目錄建立入口，並更新本節。

## Local Verification

Catalog、CLI 或 command generation：

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py
```

Proxy 或 lifecycle：

```bash
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
```

跨模組改動完成後，再執行 root 的完整測試。需要啟動真實模型的變更依 [`docs/development.md`](../docs/development.md#gpu-smoke-test) 執行 GPU smoke test。
