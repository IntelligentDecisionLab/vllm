FROM vllm/vllm-openai:v0.28.0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY vllm_gateway ./vllm_gateway

# The gateway starts vLLM with sys.executable, so it must live in the image's own Python.
RUN pip install --no-cache-dir --no-deps . \
    && python3 -c "import fastapi, uvicorn, httpx, huggingface_hub, pynvml, vllm, vllm_gateway"

# /state is a named volume; Docker copies this directory's mode into it so any uid can write.
RUN mkdir -p /state/cache && chmod -R 777 /state

ENV HF_HUB_CACHE=/data/huggingface/hub \
    VLLM_STATE_DIR=/state \
    HOME=/state \
    XDG_CACHE_HOME=/state/cache \
    VLLM_CACHE_ROOT=/state/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/state/cache/torchinductor \
    USER=gateway \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1

EXPOSE 11432
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11432/health', timeout=4)"

# umask 002 keeps files written into the shared Hugging Face cache group-writable.
ENTRYPOINT ["sh", "-c", "umask 002; exec python3 -m vllm_gateway"]
