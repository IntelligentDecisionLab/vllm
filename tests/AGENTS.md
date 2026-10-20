# Tests 協作規範

本檔只適用於 `tests/`。先遵守 root [`AGENTS.md`](../AGENTS.md)。完整測試分區與開發 workflow 見 [`docs/development.md`](../docs/development.md)。

## Test Structure

- `test_catalog.py` 以假的 HF cache（暫存目錄加最小 safetensors header）覆蓋掃描、`config.json` 解析、alias 驗證與持久化。
- `test_memory.py` 覆蓋記憶體估算與 `nvidia-smi` 輸出解析。
- `test_downloads.py` 以注入的假 Hub 函式覆蓋下載 job。
- `test_app.py` 以 `TestClient` 和 `FakeManager` 驗證 HTTP routing、alias/model/download 端點與 stream cleanup。
- `test_manager.py` 全檔使用 AnyIO；以 `FakeProcess`、`FakeGpu`、`asyncio.Event` 和 monkeypatch 控制並行 lifecycle 與 admission。
- `test_diagnostics.py` 覆蓋失敗分類。

## Local Conventions

- 單元測試不得載入模型或要求 GPU。以 fake subprocess、假 cache 目錄、`FakeGpu` 和 monkeypatch 隔離外部 runtime。
- Lifecycle test 要明確控制 gate 和 release；不要用任意 sleep 猜測 task 是否完成。短暫 `await asyncio.sleep(0)` 只用來讓 event loop 推進到可觀察狀態。
- 每個 lease test 在結束前釋放 lease 並 shutdown manager，避免遺留 watcher/client task。
- Proxy test 同時檢查 status/body、alias、forwarded payload 和 active request cleanup。
- 記憶體相關測試的數字要從 `memory.py` 的常數推導，不要寫死，以免校正常數時大量失敗。
- Foreign port/PID 測試不能向測試不擁有的 process 發 signal。這是 production ownership policy 的 regression coverage。
- 保持 fixture 與 helper 靠近使用它們的 test module；目前沒有共用 `conftest.py`，不要為單次使用建立抽象層。

## Instruction Index

本目錄目前沒有包含 `AGENTS.md` 的直接子目錄。未來若整合測試或 GPU 測試形成獨立目錄，再在該目錄新增入口並更新本節。

## Local Verification

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py tests/test_memory.py tests/test_downloads.py
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
```

修改 shared helper 或跨測試區行為後，執行完整 suite：

```bash
uv run --extra test python -m pytest -q
```
