# Tests 協作規範

本檔只適用於 `tests/`。先遵守 root [`AGENTS.md`](../AGENTS.md)。完整測試分區與開發 workflow 見 [`docs/development.md`](../docs/development.md)。

## Test Structure

- `test_catalog.py` 覆蓋 catalog shape、legacy compatibility、CLI/env parsing、command generation 與 request defaults。
- `test_app.py` 以 `TestClient` 和 `FakeManager` 驗證 HTTP routing、stream cleanup、management API 與 reload。
- `test_manager.py` 全檔使用 AnyIO；以 `FakeProcess`、`asyncio.Event` 和 monkeypatch 控制並行 lifecycle。
- `test_script.py` 驗證 repository 內的 `scripts/vllm`，涵蓋 start/status/load/unload/restart/stop 行為和 process ownership 防護。部署時再將該檔案安裝到 `~/.local/bin/vllm`。

## Local Conventions

- 單元測試不得載入模型或要求 GPU。以 fake subprocess、temporary TOML、local HTTP fake 和 monkeypatch 隔離外部 runtime。
- Lifecycle test 要明確控制 gate 和 release；不要用任意 sleep 猜測 task 是否完成。短暫 `await asyncio.sleep(0)` 只用來讓 event loop 推進到可觀察狀態。
- 每個 lease test 在結束前釋放 lease 並 shutdown manager，避免遺留 watcher/client task。
- Proxy test 同時檢查 status/body、canonical model、forwarded payload 和 active request cleanup。
- Config field 變更至少加入 parsing 與 generated command assertion；protected option 需測 hyphen/underscore spelling。
- Foreign port/PID 測試不能向測試不擁有的 process 發 signal。這是 production ownership policy 的 regression coverage。
- 保持 fixture 與 helper 靠近使用它們的 test module；目前沒有共用 `conftest.py`，不要為單次使用建立抽象層。

## Instruction Index

本目錄目前沒有包含 `AGENTS.md` 的直接子目錄。未來若整合測試或 GPU 測試形成獨立目錄，再在該目錄新增入口並更新本節。

## Local Verification

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
uv run --extra test python -m pytest -q tests/test_script.py
```

修改 shared helper 或跨測試區行為後，執行完整 suite：

```bash
uv run --extra test python -m pytest -q
```
