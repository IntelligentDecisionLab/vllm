# SGLang inference service

這個專案提供一個單一的 OpenAI-compatible endpoint，預設運行在 `11432`。它會讀取相鄰 `../llama.cpp/models-preset.ini` 的 named presets，並在收到請求時啟動對應的 SGLang backend。

目前會暴露以下模型名稱：

- `qwen3.8-27b`（預設）
- `qwen3.6-27b`
- `qwen3.6-35b-a3b`
- `llama3.2-1b`

服務預設只常駐一個 backend。這讓所有模型都能透過相同的 API 使用，同時避免 27B/35B 權重一起佔滿 VRAM；同一個已載入模型的請求仍由 SGLang scheduler 做高併發 batching。切換模型時，服務會等目前請求完成、停止舊 backend，再啟動新模型。可用 `--idle-seconds` 在閒置時自動釋放 VRAM。

## 安裝與啟動

本機有 NVIDIA GPU 時：

```bash
uv sync --extra runtime
uv run python main.py --preload
```

`--preload` 會在 `11432` 綁定前先載入 preset 排序中的第一個生成模型 `qwen3.8-27b`。不加此選項時，服務會立即啟動，但第一個使用某模型的 request 會負責載入它；也可以用 `--default-model llama3.2-1b` 選擇較小的預設模型。

這台機器有兩張 GPU，因此預設使用 `--tensor-parallel-size 2`、`--data-parallel-size 1`：同一份模型跨兩張 GPU 做 TP，對 27B 與長 context 較安全。若模型能完整放進單卡，可改成 DP2 來提高獨立 request 的吞吐量：

```bash
uv run python main.py --tensor-parallel-size 1
uv run python main.py --tensor-parallel-size 1 --data-parallel-size 2
uv run python main.py --default-model qwen3.8-27b --preload
```

wrapper 也支援相同設定：`SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=2 ~/.local/bin/sglang start`。

啟動器預設使用 `--cuda-graph-backend-prefill tc_piecewise`，以避開這台 Blackwell GPU 上預設 prefill FlashInfer graph 的相容性問題；可在 `~/.local/bin/sglang` 的設定區塊修改 `CUDA_GRAPH_BACKEND_PREFILL`，或用 `SGLANG_CUDA_GRAPH_BACKEND_PREFILL` 暫時覆寫。

同理，預設 `ATTENTION_BACKEND=triton`；若升級 runtime 後想改回其他 kernel，可直接修改 wrapper 設定或使用 `SGLANG_ATTENTION_BACKEND`。

wrapper 也會預設使用 `/usr/local/cuda/bin`（本機指向 CUDA 13.2）編譯 SGLang 的 JIT kernel；若 CUDA 安裝位置不同，可修改 `CUDA_HOME` 或設定 `SGLANG_CUDA_HOME`。

如果 SGLang 是在另一個 virtualenv/container 安裝的，請用該環境的 Python 啟動 `main.py`；gateway 會用同一個 Python 執行 `python -m sglang.launch_server`。

## API

```bash
curl http://127.0.0.1:11432/health
curl http://127.0.0.1:11432/v1/models

curl http://127.0.0.1:11432/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "用一句話介紹 SGLang"}],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }'
```

所有沒有指定 `model` 的 JSON request 都使用預設的生成模型。也可以直接使用 OpenAI SDK，把 `base_url` 設為 `http://127.0.0.1:11432/v1`。

## 模型與相容性

gateway 會依照 preset 的 `hf-repo`/`hf-file` 從 Hugging Face cache 解析 GGUF；本機 `model = ...` preset 則直接使用該路徑。SGLang backend 以 `--load-format gguf` 啟動，embedding preset 會加上 `--is-embedding`。

`qwen3.5-9b-mtp` 的 `spec-type = draft-mtp` metadata 會保留在 catalog，但目前 gateway 不會把 llama.cpp 專用的 MTP 參數假裝成 SGLang speculative decoding。另有 `qwen3.5-9b` 非 MTP preset，指向官方原生 `Qwen/Qwen3.5-9B` Transformers 權重；可用 `--default-model qwen3.5-9b` 將它設為預設模型。先前下載的 `unsloth/Qwen3.5-9B-GGUF` Q8 檔案仍保留在 Hugging Face cache，但目前 SGLang GGUF loader 不支援其 `qwen35` architecture。若原生模型需要額外參數，可用 `--sglang-arg` 傳入，例如：

`reasoning = off` preset 會在 chat request 沒有明確指定時自動加入 `chat_template_kwargs.enable_thinking = false`。

```bash
uv run python main.py --sglang-arg=--disable-cuda-graph
```

「暴露所有 preset」與「SGLang 能載入每一個 GGUF architecture」是兩件事：如果目前安裝的 SGLang 尚未支援某個 GGUF architecture，該模型會在第一次請求時回傳 `503 backend_unavailable`，其他模型仍可使用。服務會保留完整 preset catalog，方便在升級 SGLang 後直接重試。
