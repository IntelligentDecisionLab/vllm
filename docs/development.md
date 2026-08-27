# 開發與驗證

## 環境

專案要求 Python 3.12 以上，使用 uv 管理 lock file。Gateway 的 FastAPI/httpx 單元測試不需要 GPU；vLLM 是 optional runtime dependency。

```bash
uv sync --extra runtime --extra test
```

只開發 gateway 且不載入模型時，可省略 `runtime` extra：

```bash
uv sync --extra test
```

## Entry points

```bash
uv run python -m sglang_service --help
uv run python -m sglang_service
uv run python -m sglang_service --preload
```

`main.py` 和 `pyproject.toml` 的 `vllm-service`、`vllm-gateway` scripts 都呼叫相同的 CLI main。

## 測試分區

- `tests/test_catalog.py`：TOML/legacy INI parsing、CLI/env precedence、command generation、model defaults，以及部分 compatibility tests。
- `tests/test_app.py`：FastAPI routing、stream lease cleanup、management API 與 hot reload。
- `tests/test_manager.py`：async lifecycle、concurrent startup、capacity、drain、crash、port ownership 與 reload invariants。
- `tests/test_script.py`：以 temporary fake service 測試 `~/.local/bin/vllm` 的 start/status/load/unload/restart/stop 行為和 process ownership 防護。

完整驗證：

```bash
uv run --extra test python -m pytest -q
```

依修改範圍縮小迴圈：

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
uv run --extra test python -m pytest -q tests/test_script.py
```

Package build：

```bash
uv build
```

Repository 目前沒有 CI workflow，也沒有 Ruff、Black、mypy 或其他 lint/type-check 設定。不要在文件中宣稱存在未配置的 lint gate。

## 修改工作流

### Catalog field 或 command option

1. 在 `ModelSpec` 加 typed field，並在 TOML parser 處理支援的 key spelling。
2. 在 `ModelManager._command()` 明確定義 CLI override、model spec 與 fallback 的 precedence。
3. 若 vLLM option 由 gateway 管理，把它列入 protected flags，避免 `vllm_args` 產生重複或互相衝突的參數。
4. 更新 catalog parsing 與 command-generation tests，再同步 `models.toml` 和使用者文件。

### Proxy 或 lifecycle

- Proxy 每條成功 acquire 的路徑都必須 release lease，包括 HTTP error、SSE close、client cancellation 與未預期 exception。
- Manager concurrency tests 使用 `FakeProcess`、`asyncio.Event` 和 monkeypatch 控制 startup/stop，不啟動真實 vLLM。
- 修改 state transition 時需涵蓋 concurrent first load、等待者 cancellation、active unload、drain timeout、process crash 和 shutdown。
- 不得以 `fuser`、`pgrep`、port regex 或全機 process scan 猜測 ownership。

### Companion script

Script 不在 repository 內，但 `tests/test_script.py` 固定讀取 `~/.local/bin/vllm`。變更部署 script 時要在同一環境執行該測試。若要讓專案可移植地安裝 script，應先把 script 納入 repository 與 packaging，再調整測試路徑；不要只修改測試繞過現況。

## GPU smoke test

單元測試通過後，只有 runtime、catalog 或 backend command 改動需要 GPU smoke test：

```bash
vllm restart
vllm load <model>
vllm status
curl http://127.0.0.1:11432/ready
curl http://127.0.0.1:11432/v1/models
```

再送出一個非串流與一個串流 completion，確認 HTTP 200、canonical model routing、stream 結束後 active request 回到 0。模型載入耗時和 VRAM 取決於硬體，不能把 GPU smoke test 當成一般單元測試的一部分。
