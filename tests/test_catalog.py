import json
from pathlib import Path

from sglang_service.catalog import ModelCatalog
from sglang_service.cli import build_parser
from sglang_service.manager import ModelManager


def test_loads_llama_cpp_presets() -> None:
    catalog = ModelCatalog.from_preset_file(Path(__file__).parents[2] / "llama.cpp" / "models-preset.ini")

    assert [spec.name for spec in catalog.specs] == [
        "qwen3.8-27b",
        "qwen3.6-27b",
        "qwen3.6-35b-a3b",
        "llama3.2-1b",
    ]
    assert catalog.default.name == "qwen3.8-27b"
    
    # Qwen 3.8
    qwen38 = catalog.get("qwen3.8-27b")
    assert qwen38.is_gguf
    assert qwen38.context_length == 262144
    assert qwen38.kv_cache_dtype == "fp8_e5m2"
    assert qwen38.idle_seconds == 300.0

    # Qwen 3.6
    qwen36 = catalog.get("qwen3.6-27b")
    assert qwen36.reasoning == "off"
    assert qwen36.context_length == 131072
    assert qwen36.kv_cache_dtype == "fp8_e5m2"


def test_backend_command_includes_rope_and_kv_cache() -> None:
    catalog = ModelCatalog.from_preset_file(Path(__file__).parents[2] / "llama.cpp" / "models-preset.ini")
    manager = ModelManager(catalog, tensor_parallel_size=1, data_parallel_size=2)
    spec = catalog.get("qwen3.8-27b")
    command = manager._command(spec, "/tmp/model.gguf", 30000)

    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--data-parallel-size") + 1] == "2"
    assert command[command.index("--load-format") + 1] == "gguf"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e5m2"
    assert command[command.index("--context-length") + 1] == "262144"


def test_cli_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.port == 11432
    assert args.graceful_shutdown_timeout == 30
