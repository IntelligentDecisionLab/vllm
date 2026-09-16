# Gateway 架構

## 系統邊界

本 repository 實作一個 OpenAI-compatible gateway。Gateway 解析 catalog、啟動與停止獨立的 vLLM API server subprocess、轉送 HTTP request，並追蹤每個 backend 的 request lease。模型執行、scheduler、CUDA kernel 與權重格式仍由上游 vLLM 負責。

companion script 在 repo 的 `scripts/vllm`，部署時安裝到 `~/.local/bin/vllm`。它管理 gateway PID 與 log；gateway 的 `ModelManager` 才擁有 backend process group。測試執行 repo 內的 script 版本，確認 script 不會掃描或終止不屬於 gateway 的程序。

## 元件

- `vllm_gateway/cli.py`：解析 CLI 與環境變數，載入 catalog，建立 `ModelManager` 和 FastAPI app，再啟動 Uvicorn。
- `vllm_gateway/catalog.py`：把 TOML 解析成 immutable `ModelSpec`，處理 default、alias、名稱正規化與 model path。
- `vllm_gateway/manager.py`：分配 private port、建立 vLLM subprocess、執行 readiness probe、管理 capacity、lease、LRU/idle eviction、reload 與 shutdown。
- `vllm_gateway/app.py`：提供 health、model listing、localhost-only management API 和 catch-all proxy；在轉送前把 model identity canonicalize。
- `models.toml`：repository 目前使用的 model catalog 與部署參數。

`python -m vllm_gateway`、`vllm-service` 與 `vllm-gateway` 最後都進入 `vllm_gateway.cli:main`。

## Request flow

1. `app.py` 從 JSON body 讀取 `model`。未指定時使用目前 default model。
2. `ModelCatalog.get()` 接受 canonical name、normalized name、alias、model path 或 Hugging Face source，並回傳唯一的 `ModelSpec`。
3. App 將 outgoing `model` 改成 canonical name，套用 request-level model defaults，再向 manager 取得 `BackendLease`。
4. `ModelManager.acquire()` 共用同模型正在執行的 startup task。若 capacity 已滿，它只會卸載 `READY` 且沒有 active request 的 least-recently-used backend。
5. Manager 把 request 串流轉送到 `127.0.0.1:<private-port>`。一般 response 讀完、SSE stream 關閉或 proxy 發生例外時，app 都必須釋放 lease。

Lease 讓 shutdown、unload、reload 和 idle eviction 知道何時可以安全停止 backend。修改 proxy error path 時，需保留「每次成功 acquire 恰好 release 一次」的不變條件。

## Backend lifecycle

Backend state 為：

```text
STOPPED -> LOADING -> READY -> DRAINING -> STOPPED
                    \-> FAILED
```

重要不變條件：

- Private port 依 canonical model name 做 deterministic mapping；collision 時在 bounded range 內 probing。
- Port 被外部 process 占用時，gateway 跳過或回報錯誤，不取得該 process 的 ownership。
- Backend 必須同時通過 `/health`，且 `/v1/models` 包含預期 canonical model，才能進入 `READY`。
- 同模型 concurrent first requests 共用一個 startup task；等待者取消時不能取消共享 startup。
- `DRAINING` backend 不接受新 lease。Manager 等待 active lease 歸零，逾時後才停止 process。
- Gateway 使用新的 process session 啟動 backend，停止時只向自己建立的 process group 發送 SIGTERM，逾時後才 SIGKILL。

## Catalog reload

`POST /admin/reload` 重新解析啟動時的 config path。解析失敗時，app 保留舊 catalog 並回 400。解析成功後：

- 新增 model：加入 catalog，首次 request 時才載入。
- 移除 model：取消 startup 或 drain 後停止 resident backend。
- `ModelSpec` 改變：停止舊 backend；下次 request 使用新 spec 啟動。
- Spec 完全相同：保留目前 backend process。

若 CLI 指定 `--default-model`，reload 不更改 request default；否則 app 跟隨新 catalog 的 default。

## Gateway-managed vLLM options

`ModelManager._command()` 統一產生 identity、network、parallelism、memory、context、KV cache、parser、chat template、attention backend 和 Hugging Face override 參數。這些 option 不能藏在 `vllm_args` 或全域 `--vllm-arg` 中。

這項限制曾直接造成 model load 失敗：catalog 把 `--attention-backend` 放在 `vllm_args`，manager 因它是 protected option 而拒絕啟動。`attention_backend` 現在是 `ModelSpec` 的正式欄位；新增其他受管理 option 時，也應建立 typed field、CLI override 與 command-generation test，而不是繞過 protected flag validation。
