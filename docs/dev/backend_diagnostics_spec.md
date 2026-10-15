# Backend 診斷只對管理端提供原因

**Goal:** 管理端 load／CLI 可辨識 backend 失敗原因；推論 API 只回摘要與 diagnostic ID。
**Architecture:** manager 擁有 subprocess 輸出與每次 generation 的診斷；app 擁有 HTTP contract；新增小型 `diagnostics.py` 提供有界 buffer、安全摘要與 typed exception。Python 3.12、asyncio，不增加 dependency。
**Grounding:** `manager.py:Backend` 已有 error/failed_at，`_start_generation` 做失敗清理，`_watch_process` 記錄 crash，`_start_backend` 使用 stdout/stderr=None；`app.py:create_app` 的 admin load 與 acquire error 都回 503，public health/ready 也使用 manager.health；`scripts/vllm:load_model` 在 curl 後 echo，會蓋掉 curl 非零 exit status。【已驗證】

## 1. 只增加診斷，不改模型或 ownership

- In-Scope：backend startup/crash 診斷、API 權限分層、CLI 保留錯誤 body 與 exit status、測試／使用文件。
- Non-Goals：修改 models.toml、catalog、CLI options、dependencies；自動重試或調 VRAM；辨識 GPU contention／unload 殘留；掃全機 process；讀共用 log 猜 backend 歸屬；把 traceback／原始 log／路徑／credentials 放入 HTTP。
- 保留所有既有 dirty changes；不 commit，不安裝 script，不重啟實際 gateway 或載入 GPU 模型。

## 2. 診斷合約必須可測且不洩漏原始文字

以下新增合約均為【新設計】：

- **REQ-1:** `BackendDiagnostic` 是不可變資料，包含 `code: str`、`message: str`、`diagnostic_id: str`（UUID hex）。`BackendStartupError(RuntimeError)` 有 `.diagnostic`，internal str 可保留 exit code 等現有訊息以相容測試，但 HTTP 不直接回 str(exc)。每 generation 失敗只產生一筆診斷；同一 startup 的等待者與 cooldown 重用同一 ID；restart 清除旧診斷；runtime crash 亦可產生診斷。
- **REQ-2:** 每 backend 使用單一合併 PIPE（stdout=PIPE、stderr=STDOUT）且立即啟動 reader。固定大小 `read(4096)`，不可使用無界 readline；持續 drain，保存最多 64 KiB tail。每段仍寫 LOGGER（model/generation 標記），不因成功 ready 停止 drain。跨 chunk／ANSI／invalid UTF-8／超長行可處理；已匹配到的明確失敗訊號不因後續 tail 覆蓋而遺失。無 stream 的 FakeProcess 仍可運作。
- **REQ-3:** 僅憑明確文字分類：`No available memory for the cache blocks` → `kv_cache_memory_insufficient`；`CUDA out of memory` 或 `torch.OutOfMemoryError` → `cuda_out_of_memory`；啟動 deadline → `backend_startup_timeout`；未識別 exit／crash → `backend_startup_failed`（runtime 可用 `backend_exited`）。輸出使用固定安全英文摘要，不引用原始 exception／log。OOM 與 cache 訊號優先於 fallback；健康檢查未 ready 時不能僅因警告宣告失敗。另對 CapacityError／PortUnavailable／FileNotFoundError／ValueError 提供固定 code `backend_capacity_exceeded`／`backend_port_unavailable`／`model_files_unavailable`／`invalid_backend_configuration`，其餘 RuntimeError fallback `backend_startup_failed`。
- **REQ-4:** 退出／timeout 的最終診斷需在停止程序與有限輸出收尾後建立，避免 race 遺失 traceback 尾端。reader task 在取消 startup、unload、crash、shutdown 均收束；子程序後代仍握 pipe 不能造成無限等待，輸出收尾有固定短 timeout。不在 manager lock 內等 subprocess／reader；不得改 process group ownership、ports、lease draining、concurrent startup 及取消語意。最後 LOGGER error 記錄 ID、code、model／generation；完整輸出仍在本機 log。
- **REQ-5:** `/admin/models/{id}/load` 失敗仍 HTTP 503、type=`backend_unavailable`，補 code/message/diagnostic_id。未知模型仍 404；認證／localhost 規則不變。`/admin/models` 的 resident model 可含 `diagnostic`（上述三欄），供查詢上次失敗。public proxy acquire 失敗仍 503、type=`backend_unavailable`、message=`Model backend is unavailable; consult the management API with the diagnostic ID.`、code=`backend_unavailable`、diagnostic_id；不含具體分類與原始原因。gateway 自己產生的 upstream HTTPError 仍 502，但使用固定安全摘要與 ID、詳情只寫 log。backend 已回覆的 HTTP body／SSE 原樣轉送，不在本次修改。
- **REQ-6:** `/health`、`/ready` 不可藉 resident model.error 或 diagnostic 洩漏詳細原因；對失敗 error 回固定摘要／保留 diagnostic_id，移除詳細 diagnostic。管理端仍有安全具體原因，成功管理 load body 維持既有欄位並可 additive diagnostic。不要修改 manager.health 呼叫簽名（FakeManager 相容）。
- **REQ-7:** CLI load 保留 API JSON（code/message/ID）且回傳 curl 的原始非零 exit status；unload/models 相同現有 echo 問題一併修復。Bearer credential 仍從 stdin header 傳入，不能放進 argv。不增加 jq 依赖。

【新設計】管理端例：
```json
{"error":{"type":"backend_unavailable","code":"kv_cache_memory_insufficient","message":"Not enough GPU memory to allocate KV cache blocks.","diagnostic_id":"d728af5e6f64450287e798207b704edd"}}
```

## 3. 兩片依序完成，main 各片驗證後再派下一片

### Slice 1：Backend 輸出診斷與 lifecycle

- Covers：REQ-1～4。
- Create：`vllm_gateway/diagnostics.py`、`tests/test_diagnostics.py`。
- Modify：`vllm_gateway/manager.py`、`tests/test_manager.py`。不得改 app/script/docs。
- Consumes：【已驗證】Backend/generation/_start_generation/_watch_process/_stop_backend 與 FakeProcess.wait/finish。
- Produces：【新設計】BackendDiagnostic、BackendStartupError 與 manager Backend.diagnostic／輸出 task 狀態；`_backend_dict` 添加 diagnostic（None 或三欄 dict）；其他 raw error 欄位可維持內部相容，由 Slice 2 處理 public exposure。pre-spawn 與 monkeypatched startup 的一般 exception 也要包裝為有診斷的 exception；CancelledError 不包裝。
- Fixtures：asyncio.StreamReader fed 合成 vLLM log；FakeProcess 控制退出與 reader EOF；不同 backend 各自不同輸出；timeout/cancellation 由 Event 控制。可用 stdlib Python dummy child 測 pipe drain，不需 vLLM／GPU／網路。
- Steps：先增加測試並確認紅，再寫最小實作，驗證綠；覆盖 KV/OOM/unknown、分塊／ANSI／非法字元、超64KiB且early evidence保留、成功後大量輸出、退出尾端、繼承pipe不EOF、cancel/shutdown task無洩漏、cooldown同ID、restart新ID、parallel隔離、foreign ownership與現有 lease tests。
- Verify：`uv run --extra test python -m pytest -q tests/test_diagnostics.py tests/test_manager.py`；`uv run --extra dev ruff check vllm_gateway/diagnostics.py vllm_gateway/manager.py tests/test_diagnostics.py tests/test_manager.py`。

### Slice 2：HTTP／CLI 權限分層

- Covers：REQ-5～7。
- Modify：`vllm_gateway/app.py`、`scripts/vllm`、`tests/test_app.py`、`tests/test_script.py`、`docs/development.md`。不得修改 manager/diagnostics 的既定契約；若發現缺口回報 main。
- Consumes：Slice 1 diagnostic exception／安全 summary／Backend status。app 純 presentation，不啟停 process。
- Produces：上述管理／推論 JSON contract、CLI 非零碼、繁中使用文件（重新啟動 gateway 才生效、script install 指令、ID log 查詢與 code 表）。
- Fixtures：FakeManager.acquire/load 注入同一 BackendStartupError 與含 secret/path 的一般 RuntimeError；health resident_models 含 raw error＋diagnostic；fake HTTP server load 回503 JSON。
- Steps：先測試紅再實作；測管理 authentication 不變、管理 detailed/public generic 同ID、general exception不洩漏、public health/ready不洩漏、502不洩漏、CLI curl22且保留code/ID與secret不進argv、既有stream/lease測試。
- Verify：`uv run --extra test python -m pytest -q tests/test_app.py tests/test_script.py`；`bash -n scripts/vllm`。

## 4. 驗收不需要動 GPU

- 完整 suite：`uv run --extra test python -m pytest -q`。
- Lint/format：`uv run --extra dev ruff check .`、`uv run --extra dev ruff format --check .`。
- CLI/package：`uv run python -m vllm_gateway --help`、`uv build`。
- 成功標準：新測試與既有測試皆綠、沒有 pending reader task、分類與權限隔離符合 contract。沒改 runtime/catalog/backend command，故本次不啟動實際模型。

## 附錄：已確認的骨架

核心主張：以 per-generation 輸出辨識原因，管理端詳細而推論端安全。必要支撐線為 bounded lifecycle、安全固定分類、HTTP 權限分層與 CLI 非零碼；不做自動資源修復與 raw log API。
