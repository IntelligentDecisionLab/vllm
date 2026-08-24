import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sglang_service.app import _apply_model_defaults, create_app
from sglang_service.catalog import ModelCatalog, ModelSpec, resolve_model_path
from sglang_service.cli import build_parser
from sglang_service.manager import Backend, BackendLease, CapacityError, ModelManager

REPO_ROOT = Path(__file__).parents[1]
MODELS_TOML = REPO_ROOT / "models.toml"


def test_loads_models_toml() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)

    model_names = [spec.name for spec in catalog.specs]
    assert "qwen3.8-27b" in model_names
    assert "qwen3.6-27b" in model_names
    assert catalog.default.name == "qwen3.8-27b"

    # Qwen 3.8 FP8
    qwen38 = catalog.get("qwen3.8-27b")
    assert qwen38.model_path == "Qwen/Qwen3.8-27B-FP8"
    assert qwen38.source == "Qwen/Qwen3.8-27B-FP8"
    assert not qwen38.is_gguf
    assert qwen38.context_length == 262144
    assert qwen38.kv_cache_dtype == "fp8_e4m3"
    assert qwen38.tensor_parallel_size == 2
    assert qwen38.data_parallel_size == 1
    assert qwen38.idle_seconds == 0.0

    # Qwen 3.6 FP8
    qwen36 = catalog.get("qwen3.6-27b")
    assert qwen36.model_path == "Qwen/Qwen3.6-27B-FP8"
    assert not qwen36.is_gguf
    assert qwen36.reasoning == "off"
    assert qwen36.context_length == 262144
    assert qwen36.kv_cache_dtype == "fp8_e4m3"
    assert qwen36.tensor_parallel_size == 2
    assert qwen36.data_parallel_size == 1


def test_model_alias_and_normalization() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)

    # Alias lookup
    assert catalog.get("qwen3.8-27b-fp8").name == "qwen3.8-27b"
    assert catalog.get("qwen3.6-27b-fp8").name == "qwen3.6-27b"

    # Normalization lookup (underscores vs dashes)
    assert catalog.get("qwen3_8_27b").name == "qwen3.8-27b"
    assert catalog.get("qwen3_6_27b").name == "qwen3.6-27b"


def test_backend_command_generation_fp8() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    spec = catalog.get("qwen3.8-27b")
    model_path = resolve_model_path(spec)

    command = manager._command(spec, model_path, 30000)

    assert "--load-format" not in command
    assert command[command.index("--model") + 1] == "Qwen/Qwen3.8-27B-FP8"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--data-parallel-size") + 1] == "1"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--served-model-name") + 1] == "qwen3.8-27b"
    assert command[command.index("--port") + 1] == "30000"


def test_backend_command_manager_overrides() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(
        catalog,
        tensor_parallel_size=1,
        context_length_override=65536,
        vllm_args=("--enforce-eager",),
    )
    spec = catalog.get("qwen3.8-27b")
    command = manager._command(spec, "Qwen/Qwen3.8-27B-FP8", 30000)

    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--data-parallel-size") + 1] == "1"
    assert command[command.index("--max-model-len") + 1] == "65536"
    assert "--enforce-eager" in command


def test_backend_command_rope_scaling() -> None:
    toml_content = """
    [models.test_rope]
    model_path = "Qwen/Qwen3.8-27B-FP8"
    rope_scaling = "yarn"
    rope_scale = 2.0
    yarn_orig_ctx = 32768
    rope_theta = 1000000.0
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(temp_path)
        manager = ModelManager(catalog)
        spec = catalog.get("test_rope")
        command = manager._command(spec, spec.model_path, 30000)

        assert "--hf-overrides" in command
        override_json = json.loads(command[command.index("--hf-overrides") + 1])
        assert override_json["rope_scaling"]["rope_type"] == "yarn"
        assert override_json["rope_scaling"]["factor"] == 2.0
        assert override_json["rope_scaling"]["original_max_position_embeddings"] == 32768
        assert override_json["rope_theta"] == 1000000.0
    finally:
        temp_path.unlink()


def test_backend_command_custom_kv_cache_dtype() -> None:
    spec = ModelSpec(name="custom-kv", model_path="custom/repo", kv_cache_dtype="fp8_e4m3")
    catalog = ModelCatalog([spec], MODELS_TOML)
    manager = ModelManager(catalog)
    command = manager._command(spec, "custom/repo", 30000)
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"


def test_backend_command_without_context_length() -> None:
    spec = ModelSpec(name="no-ctx", model_path="custom/repo", context_length=None)
    catalog = ModelCatalog([spec], MODELS_TOML)
    manager = ModelManager(catalog)
    command = manager._command(spec, "custom/repo", 30000)
    assert "--max-model-len" not in command


def test_resolve_model_path() -> None:
    # HF repo ID returned directly
    hf_spec = ModelSpec(name="hf-test", model_path="Qwen/Qwen3.8-27B-FP8")
    assert resolve_model_path(hf_spec) == "Qwen/Qwen3.8-27B-FP8"

    # Local directory that exists
    with tempfile.TemporaryDirectory() as tmpdir:
        local_spec = ModelSpec(name="local-test", model_path=tmpdir)
        assert resolve_model_path(local_spec) == str(Path(tmpdir).resolve())

    # Nonexistent local path raises FileNotFoundError
    nonexistent_spec = ModelSpec(name="bad-local", model_path="/nonexistent/directory/model")
    with pytest.raises(FileNotFoundError):
        resolve_model_path(nonexistent_spec)


def test_model_spec_public_dict() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    spec = catalog.get("qwen3.8-27b")
    info = spec.public_dict(loaded=True)

    assert info["id"] == "qwen3.8-27b"
    assert info["object"] == "model"
    assert info["loaded"] is True
    metadata = info["metadata"]
    assert metadata["format"] == "safetensors"
    assert metadata["source"] == "Qwen/Qwen3.8-27B-FP8"
    assert metadata["kv_cache_dtype"] == "fp8_e4m3"
    assert metadata["tensor_parallel_size"] == 2
    assert metadata["data_parallel_size"] == 1


def test_cli_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.port == 11432
    assert args.graceful_shutdown_timeout == 30
    assert args.config_file.name == "models.toml"
    assert args.config.name == "models.toml"
    assert args.preset_file.name == "models.toml"


def test_cli_custom_arguments() -> None:
    args = build_parser().parse_args([
        "--config", "/tmp/custom_models.toml",
        "--port", "8000",
        "--tp-size", "4",
        "--default-model", "qwen3.6-27b",
    ])

    assert args.config_file == Path("/tmp/custom_models.toml")
    assert args.port == 8000
    assert args.tensor_parallel_size == 4
    assert args.default_model == "qwen3.6-27b"


def test_cli_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_CONFIG_FILE", "/tmp/env_models.toml")
    monkeypatch.setenv("VLLM_SERVICE_PORT", "12345")
    monkeypatch.setenv("VLLM_TP_SIZE", "4")
    monkeypatch.setenv("VLLM_DEFAULT_MODEL", "qwen3.6-27b")

    args = build_parser().parse_args([])
    assert args.config_file == Path("/tmp/env_models.toml")
    assert args.port == 12345
    assert args.tensor_parallel_size == 4
    assert args.default_model == "qwen3.6-27b"


def test_catalog_error_handling() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    with pytest.raises(KeyError, match="unknown model 'nonexistent'"):
        catalog.get("nonexistent")

    with pytest.raises(FileNotFoundError):
        ModelCatalog.from_toml_file(Path("/nonexistent/file.toml"))

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("# empty file\n")
        f.flush()
        empty_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="no models defined"):
            ModelCatalog.from_toml_file(empty_path)
    finally:
        empty_path.unlink()


def test_apply_model_defaults() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    qwen36 = catalog.get("qwen3.6-27b")

    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    _apply_model_defaults(payload, qwen36)
    assert payload.get("chat_template_kwargs") == {"enable_thinking": False}

    qwen38 = catalog.get("qwen3.8-27b")
    payload2 = {"messages": [{"role": "user", "content": "Hello"}]}
    _apply_model_defaults(payload2, qwen38)
    assert "chat_template_kwargs" not in payload2


def test_app_endpoints() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    app = create_app(catalog, manager)

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert "qwen3.8-27b" in health_data["models"]
        assert "qwen3.6-27b" in health_data["models"]

        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 503

        models_resp = client.get("/v1/models")
        assert models_resp.status_code == 200
        models_data = models_resp.json()
        assert models_data["object"] == "list"
        model_ids = [m["id"] for m in models_data["data"]]
        assert "qwen3.8-27b" in model_ids
        assert "qwen3.6-27b" in model_ids


def test_custom_toml_structures() -> None:
    content = """
    [default]
    kv_cache_dtype = "fp8_e5m2"
    tensor_parallel_size = 4

    [models.model_a]
    model_path = "/tmp/local_weights"
    context_length = 65536
    vllm_args = ["--enforce-eager", "--gpu-memory-utilization", "0.85"]

    [models.model_b]
    model_path = "org/model-b-fp8"
    tensor_parallel_size = 1
    data_parallel_size = 2
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(path)
        assert len(catalog.specs) == 2
        a = catalog.get("model_a")
        assert a.kv_cache_dtype == "fp8_e5m2"
        assert a.tensor_parallel_size == 4
        assert a.context_length == 65536
        assert a.vllm_args == ("--enforce-eager", "--gpu-memory-utilization", "0.85")

        b = catalog.get("model_b")
        assert b.kv_cache_dtype == "fp8_e5m2"
        assert b.tensor_parallel_size == 1
        assert b.data_parallel_size == 2
    finally:
        path.unlink()


def test_from_file_and_preset_file_compat() -> None:
    # from_file with toml
    c1 = ModelCatalog.from_file(MODELS_TOML)
    assert c1.default.name == "qwen3.8-27b"

    # from_preset_file with toml
    c2 = ModelCatalog.from_preset_file(MODELS_TOML)
    assert c2.default.name == "qwen3.8-27b"


def test_toml_array_of_tables() -> None:
    content = """
    [default]
    kv_cache_dtype = "fp8_e5m2"
    tensor_parallel_size = 2

    [[models]]
    name = "array_model_1"
    model_path = "org/model-1"

    [[models]]
    name = "array_model_2"
    model_path = "org/model-2"
    tensor_parallel_size = 4
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(path)
        assert len(catalog.specs) == 2
        m1 = catalog.get("array_model_1")
        assert m1.tensor_parallel_size == 2
        assert m1.kv_cache_dtype == "fp8_e5m2"

        m2 = catalog.get("array_model_2")
        assert m2.tensor_parallel_size == 4
    finally:
        path.unlink()


def test_lookup_by_source_and_hf_repo() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    # Direct HF repo ID lookup
    assert catalog.get("Qwen/Qwen3.8-27B-FP8").name == "qwen3.8-27b"
    assert catalog.get("Qwen/Qwen3.6-27B-FP8").name == "qwen3.6-27b"


def test_default_section_isolation() -> None:
    content = """
    [default]
    default = true
    kv_cache_dtype = "fp8_e5m2"

    [models.m1]
    model_path = "org/m1"

    [models.m2]
    model_path = "org/m2"
    default = true
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(path)
        # m1 shouldn't have inherited default=true from [default]
        assert catalog.get("m1").is_default is False
        assert catalog.get("m2").is_default is True
        assert catalog.default.name == "m2"
    finally:
        path.unlink()


def test_cli_empty_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TP_SIZE", "")
    monkeypatch.setenv("VLLM_CONTEXT_LENGTH", "")
    monkeypatch.setenv("VLLM_CONFIG_FILE", "")
    monkeypatch.setenv("VLLM_GRACEFUL_SHUTDOWN_TIMEOUT", "")
    monkeypatch.setenv("VLLM_MODEL_IDLE_SECONDS", "")
    monkeypatch.setenv("VLLM_SERVICE_PORT", "")

    args = build_parser().parse_args([])
    assert args.tensor_parallel_size is None
    assert args.context_length is None
    assert args.config_file.name == "models.toml"
    assert args.graceful_shutdown_timeout == 30
    assert args.idle_seconds == 0.0
    assert args.port == 11432


def test_cli_rejects_invalid_numeric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BACKEND_PORT", "not-a-port")
    with pytest.raises(ValueError, match="VLLM_BACKEND_PORT must be an integer"):
        build_parser()

    monkeypatch.delenv("VLLM_BACKEND_PORT")
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "not-a-number")
    with pytest.raises(ValueError, match="VLLM_GPU_MEMORY_UTILIZATION must be a number"):
        build_parser()


def test_resolve_model_path_env_var_and_tilde() -> None:
    # Nonexistent dollar-variable or absolute path
    spec = ModelSpec(name="env-path", model_path="$NONEXISTENT_VAR_DIR/weights")
    with pytest.raises(FileNotFoundError):
        resolve_model_path(spec)

    spec_tilde = ModelSpec(name="tilde-path", model_path="~/nonexistent_model_dir_xyz")
    with pytest.raises(FileNotFoundError):
        resolve_model_path(spec_tilde)


def test_companion_script_defaults() -> None:
    script_path = Path.home() / ".local" / "bin" / "vllm"
    assert script_path.is_file()
    content = script_path.read_text(encoding="utf-8")
    assert 'PROJECT_DIR="${VLLM_PROJECT_DIR:-${SGLANG_DIR:-$HOME/Projects/vllm}}"' in content
    assert 'TP_SIZE="${VLLM_TP_SIZE:-${SGLANG_TP_SIZE:-}}"' in content
    assert 'DP_SIZE="${VLLM_DP_SIZE:-${SGLANG_DP_SIZE:-}}"' in content
    assert 'PORT="${VLLM_SERVICE_PORT:-${SGLANG_SERVICE_PORT:-11432}}"' in content
    assert "fuser" not in content
    assert "pgrep" not in content
    assert "models|load <model>|unload <model>" in content


def test_app_proxy_validation_errors() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    app = create_app(catalog, manager)

    with TestClient(app) as client:
        # Invalid JSON
        r1 = client.post("/v1/chat/completions", content=b"{bad-json", headers={"content-type": "application/json"})
        assert r1.status_code == 400
        assert r1.json()["error"]["message"].startswith("invalid JSON body")

        # Non-string model: int, bool, list, dict
        for invalid_model in (12345, 0, False, True, ["model"], {"name": "model"}):
            r2 = client.post("/v1/chat/completions", json={"model": invalid_model})
            assert r2.status_code == 400
            assert r2.json()["error"]["message"] == "model must be a string"

        # Unknown model
        r3 = client.post("/v1/chat/completions", json={"model": "unknown-model"})
        assert r3.status_code == 404
        assert "unknown model 'unknown-model'" in r3.json()["error"]["message"]


def test_app_get_single_model_endpoint() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    app = create_app(catalog, manager)

    with TestClient(app) as client:
        # Known model: qwen3.8-27b
        r1 = client.get("/v1/models/qwen3.8-27b")
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["id"] == "qwen3.8-27b"
        assert data1["object"] == "model"
        assert data1["metadata"]["format"] == "safetensors"
        assert data1["metadata"]["source"] == "Qwen/Qwen3.8-27B-FP8"

        # Known model by alias: qwen3.6-27b-fp8
        r2 = client.get("/v1/models/qwen3.6-27b-fp8")
        assert r2.status_code == 200
        assert r2.json()["id"] == "qwen3.6-27b"

        # Unknown model
        r3 = client.get("/v1/models/nonexistent-model")
        assert r3.status_code == 404
        assert "unknown model 'nonexistent-model'" in r3.json()["error"]["message"]


def test_parse_kv_cache_dtype_hyphenated() -> None:
    from sglang_service.catalog import _parse_kv_cache_dtype

    assert _parse_kv_cache_dtype(explicit_dtype="fp8-e5m2") == "fp8_e5m2"
    assert _parse_kv_cache_dtype(explicit_dtype="fp8-e4m3") == "fp8_e4m3"
    assert _parse_kv_cache_dtype(explicit_dtype="fp4-e2m1") == "fp4_e2m1"
    assert _parse_kv_cache_dtype(cache_k="fp8-e5m2") == "fp8_e5m2"
    assert _parse_kv_cache_dtype(explicit_dtype="FP8-E5M2") == "fp8_e5m2"
    assert _parse_kv_cache_dtype(explicit_dtype="Q8_0") == "fp8_e5m2"


def test_toml_common_name_isolation() -> None:
    content = """
    name = "global_catalog_name"

    [default]
    name = "default_catalog_name"
    kv_cache_dtype = "fp8_e5m2"

    ["custom-model"]
    model_path = "org/custom-model"
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(path)
        assert len(catalog.specs) == 1
        spec = catalog.get("custom-model")
        assert spec.name == "custom-model"
        assert spec.model_path == "org/custom-model"
    finally:
        path.unlink()


@pytest.mark.anyio
async def test_backend_lease_async_context_manager() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    spec = catalog.default
    from sglang_service.manager import Backend, BackendLease

    backend = Backend(spec=spec, port=30000, active_requests=1)
    lease = BackendLease(manager, backend)

    async with lease as l:
        assert l is lease
        assert l.backend.active_requests == 1

    assert l._released is True
    assert backend.active_requests == 0


def test_catalog_single_models_table() -> None:
    content = """
    [default]
    kv_cache_dtype = "fp8_e5m2"

    [models]
    name = "single_model"
    model_path = "org/single-model"
    tensor_parallel_size = 2
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_toml_file(path)
        assert len(catalog.specs) == 1
        spec = catalog.get("single_model")
        assert spec.name == "single_model"
        assert spec.model_path == "org/single-model"
        assert spec.tensor_parallel_size == 2
        assert spec.kv_cache_dtype == "fp8_e5m2"
    finally:
        path.unlink()


def test_catalog_whitespace_model_lookup() -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    # Whitespace-only string falls back to default model
    assert catalog.get("   ").name == "qwen3.8-27b"
    assert catalog.get("").name == "qwen3.8-27b"
    # Leading/trailing whitespace in model name is handled
    assert catalog.get("  qwen3.8-27b  ").name == "qwen3.8-27b"
    assert catalog.get("  qwen3.6-27b  ").name == "qwen3.6-27b"


def test_ini_preset_preserves_default() -> None:
    content = """
    [*]
    kv-cache-dtype = fp8_e5m2

    [model_a]
    model = /tmp/a

    [model_b]
    model = /tmp/b
    default = true
    """
    with tempfile.NamedTemporaryFile(suffix=".ini", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        catalog = ModelCatalog.from_file(path)
        assert catalog.get("model_a").is_default is False
        assert catalog.get("model_b").is_default is True
        assert catalog.default.name == "model_b"
    finally:
        path.unlink()


@pytest.mark.anyio
async def test_app_proxy_lease_cleanup_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    app = create_app(catalog, manager)

    from sglang_service.manager import Backend
    mock_backend = Backend(spec=catalog.default, port=30000, active_requests=0)

    async def mock_acquire(name: str) -> BackendLease:
        mock_backend.active_requests += 1
        return BackendLease(manager, mock_backend)

    async def mock_release(backend: Backend) -> None:
        backend.active_requests = max(0, backend.active_requests - 1)

    async def mock_send_error(*args, **kwargs):
        import httpx
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(manager, "acquire", mock_acquire)
    monkeypatch.setattr(manager, "release", mock_release)
    monkeypatch.setattr(manager._client, "send", mock_send_error)

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "qwen3.8-27b", "messages": []})
        assert resp.status_code == 502
        assert mock_backend.active_requests == 0


@pytest.mark.anyio
async def test_app_proxy_lease_cleanup_on_unhandled_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)
    app = create_app(catalog, manager)

    from sglang_service.manager import Backend
    mock_backend = Backend(spec=catalog.default, port=30000, active_requests=0)

    async def mock_acquire(name: str) -> BackendLease:
        mock_backend.active_requests += 1
        return BackendLease(manager, mock_backend)

    async def mock_release(backend: Backend) -> None:
        backend.active_requests = max(0, backend.active_requests - 1)

    async def mock_send_crash(*args, **kwargs):
        raise RuntimeError("unexpected fatal crash")

    monkeypatch.setattr(manager, "acquire", mock_acquire)
    monkeypatch.setattr(manager, "release", mock_release)
    monkeypatch.setattr(manager._client, "send", mock_send_crash)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/chat/completions", json={"model": "qwen3.8-27b", "messages": []})
        assert resp.status_code == 500
        assert mock_backend.active_requests == 0


@pytest.mark.anyio
async def test_manager_keeps_two_models_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog, max_loaded_models=2, shutdown_timeout=0.01)

    started_backends: list[str] = []
    stopped_backends: list[str] = []

    async def fake_start_backend(backend: Backend) -> None:
        started_backends.append(backend.spec.name)
        backend.process = type("MockProcess", (), {"returncode": None, "pid": 12345})()

    async def fake_stop_backend(backend: Backend) -> None:
        stopped_backends.append(backend.spec.name)
        backend.process = None

    async def fake_watch(*args) -> None:
        return None

    monkeypatch.setattr(manager, "_start_backend", fake_start_backend)
    monkeypatch.setattr(manager, "_stop_backend", fake_stop_backend)
    monkeypatch.setattr(manager, "_watch_process", fake_watch)

    lease1 = await manager.acquire("qwen3.8-27b")
    lease2 = await manager.acquire("qwen3.8-27b")
    lease3 = await manager.acquire("qwen3.6-27b")

    assert started_backends == ["qwen3.8-27b", "qwen3.6-27b"]
    assert stopped_backends == []
    assert lease1.backend is lease2.backend
    assert lease1.backend is not lease3.backend
    assert lease1.backend.port != lease3.backend.port
    assert lease1.backend.active_requests == 2
    assert lease3.backend.active_requests == 1

    await lease1.release()
    await lease2.release()
    await lease3.release()
    status = await manager.health()
    assert status["loaded_models"] == ["qwen3.6-27b", "qwen3.8-27b"]
    await manager.shutdown()
    assert sorted(stopped_backends) == ["qwen3.6-27b", "qwen3.8-27b"]


def test_cli_namespace_models_file() -> None:
    args = build_parser().parse_args(["--config", "/tmp/custom.toml"])
    assert args.models_file == Path("/tmp/custom.toml")
    assert args.models_config == Path("/tmp/custom.toml")
    assert args.config == Path("/tmp/custom.toml")
    assert args.preset_file == Path("/tmp/custom.toml")


@pytest.mark.anyio
async def test_manager_capacity_rejects_second_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog, max_loaded_models=1, shutdown_timeout=0.01)

    async def fake_start_backend(backend: Backend) -> None:
        backend.process = type("MockProcess", (), {"returncode": None, "pid": 12345})()

    async def fake_stop_backend(backend: Backend) -> None:
        backend.process = None

    async def fake_watch(*args) -> None:
        return None

    monkeypatch.setattr(manager, "_start_backend", fake_start_backend)
    monkeypatch.setattr(manager, "_stop_backend", fake_stop_backend)
    monkeypatch.setattr(manager, "_watch_process", fake_watch)

    lease = await manager.acquire("qwen3.8-27b")
    with pytest.raises(CapacityError, match="active or loading"):
        await manager.acquire("qwen3.6-27b")

    await lease.release()
    second = await manager.acquire("qwen3.6-27b")
    assert second.backend.spec.name == "qwen3.6-27b"
    await second.release()
    await manager.shutdown()


@pytest.mark.anyio
async def test_manager_start_backend_failure_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = ModelCatalog.from_toml_file(MODELS_TOML)
    manager = ModelManager(catalog)

    stopped: list[str] = []

    async def fake_start_crash(backend: Backend) -> None:
        backend.process = type("MockProcess", (), {"returncode": 1, "pid": 99999})()
        raise RuntimeError("failed to start vLLM backend")

    async def fake_stop_backend(backend: Backend) -> None:
        stopped.append(backend.spec.name)
        backend.process = None

    monkeypatch.setattr(manager, "_start_backend", fake_start_crash)
    monkeypatch.setattr(manager, "_stop_backend", fake_stop_backend)

    with pytest.raises(RuntimeError, match="failed to start vLLM backend"):
        await manager.acquire("qwen3.8-27b")

    assert "qwen3.8-27b" in stopped
    status = await manager.backend_status("qwen3.8-27b")
    assert status["state"] == "failed"
    assert "failed to start vLLM backend" in status["error"]
    await manager.shutdown()


def test_sglang_download_single_table_and_aliases() -> None:
    content = """
    [models]
    name = "single-entry"
    hf-repo = "custom/hf-repo"
    hf-file = "model.safetensors"
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    import subprocess
    cmd = [
        "python3",
        "-c",
        """
import sys, tomllib
from pathlib import Path
config_path = Path(sys.argv[1])
models = []
with open(config_path, "rb") as f:
    data = tomllib.load(f)
if isinstance(data.get("models"), dict):
    models_map = data["models"]
    has_subdicts = any(isinstance(v, dict) for v in models_map.values())
    if has_subdicts:
        for name, val in models_map.items():
            if isinstance(val, dict):
                repo = val.get("model_path") or val.get("model") or val.get("path") or val.get("hf_repo") or val.get("hf-repo") or val.get("repo_id")
                file_ = val.get("hf_file") or val.get("hf-file")
                models.append((name, repo, file_))
    elif any(k in models_map for k in ("model_path", "model", "path", "hf_repo", "hf-repo", "repo_id")):
        m_name = str(models_map.get("name") or "default")
        repo = models_map.get("model_path") or models_map.get("model") or models_map.get("path") or models_map.get("hf_repo") or models_map.get("hf-repo") or models_map.get("repo_id")
        file_ = models_map.get("hf_file") or models_map.get("hf-file")
        models.append((m_name, repo, file_))
assert len(models) == 1
assert models[0] == ("single-entry", "custom/hf-repo", "model.safetensors")
print("OK")
""",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "OK" in res.stdout
    finally:
        path.unlink()
