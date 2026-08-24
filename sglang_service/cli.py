from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .catalog import ModelCatalog
from .manager import ModelManager


def _default_config_file() -> Path:
    env_path = (
        os.getenv("VLLM_CONFIG_FILE")
        or os.getenv("VLLM_MODELS_FILE")
        or os.getenv("VLLM_MODELS_CONFIG")
        or os.getenv("VLLM_CONFIG")
        or os.getenv("VLLM_PRESET_FILE")
        or os.getenv("VLLM_PRESETS_FILE")
        or os.getenv("SGLANG_CONFIG_FILE")
        or os.getenv("SGLANG_MODELS_FILE")
        or os.getenv("SGLANG_PRESETS_FILE")
    )
    if env_path and env_path.strip():
        return Path(env_path.strip())
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models.toml"


def _env_int(key: str, default: int | None = None) -> int | None:
    val = os.getenv(key)
    if val is not None and val.strip():
        try:
            return int(val.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer, got {val!r}") from exc
    return default


def _env_float(key: str, default: float | None = None) -> float | None:
    val = os.getenv(key)
    if val is not None and val.strip():
        try:
            return float(val.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be a number, got {val!r}") from exc
    return default


class SglangNamespace(argparse.Namespace):
    @property
    def config(self) -> Path:
        return getattr(self, "config_file", None)

    @property
    def models_config(self) -> Path:
        return getattr(self, "config_file", None)

    @property
    def models_file(self) -> Path:
        return getattr(self, "config_file", None)

    @property
    def preset_file(self) -> Path:
        return getattr(self, "config_file", None)


class SglangArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        if namespace is None:
            namespace = SglangNamespace()
        return super().parse_args(args=args, namespace=namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = SglangArgumentParser(description="Serve native FP8 models through vLLM on an OpenAI API.")
    parser.add_argument("--host", default=os.getenv("VLLM_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("VLLM_SERVICE_PORT", 11432))
    parser.add_argument(
        "--config",
        "--config-file",
        "--models-config",
        "--preset-file",
        dest="config_file",
        type=Path,
        default=_default_config_file(),
        help="Path to models.toml configuration file",
    )
    parser.add_argument("--default-model", default=os.getenv("VLLM_DEFAULT_MODEL"))
    parser.add_argument("--preload", action="store_true", help="load the default model before accepting requests")
    parser.add_argument(
        "--preload-model",
        action="append",
        default=[],
        help="model to load before accepting requests; repeat to preload multiple models",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=_env_int("VLLM_BACKEND_PORT", 30000),
        help="first port in the private vLLM backend range",
    )
    parser.add_argument(
        "--backend-port-count",
        type=int,
        default=_env_int("VLLM_BACKEND_PORT_COUNT", 64),
        help="number of private backend ports available to the allocator",
    )
    parser.add_argument(
        "--max-loaded-models",
        type=int,
        default=_env_int("VLLM_MAX_LOADED_MODELS"),
        help="maximum simultaneously resident models; defaults to min(2, catalog size)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=_env_int("VLLM_TP_SIZE"),
        help="tensor parallel size override",
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp-size",
        type=int,
        default=_env_int("VLLM_DP_SIZE"),
        help="data parallel size override",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("VLLM_GPU_MEMORY_UTILIZATION"),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=_env_int("VLLM_MAX_NUM_SEQS"),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=_env_int("VLLM_MAX_NUM_BATCHED_TOKENS"),
    )
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("VLLM_ATTENTION_BACKEND"),
        help="attention kernel backend passed to every vLLM model; unset lets vLLM auto-select",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=_env_int("VLLM_CONTEXT_LENGTH"),
        help="override the context length in every model entry",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=_env_float("VLLM_STARTUP_TIMEOUT", 900.0),
    )
    parser.add_argument(
        "--graceful-shutdown-timeout",
        type=int,
        default=_env_int("VLLM_GRACEFUL_SHUTDOWN_TIMEOUT", 30),
        help="maximum seconds Uvicorn waits for active requests during shutdown",
    )
    parser.add_argument(
        "--backend-shutdown-timeout",
        type=float,
        default=_env_float("VLLM_BACKEND_SHUTDOWN_TIMEOUT", 30.0),
        help="maximum seconds to drain and stop each private backend",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=_env_float("VLLM_MODEL_IDLE_SECONDS", 0.0),
        help="unload an idle model after this many seconds; 0 keeps it loaded",
    )
    parser.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help="extra argument passed to every vLLM backend; repeat as needed",
    )
    parser.add_argument("--log-level", default=os.getenv("VLLM_SERVICE_LOG_LEVEL", "info"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    catalog = ModelCatalog.from_file(args.config_file)
    raw_default = args.default_model.strip() if isinstance(args.default_model, str) and args.default_model.strip() else None
    default_model = raw_default or catalog.default.name
    catalog.get(default_model)  # validate before binding the public port
    manager = ModelManager(
        catalog,
        backend_port=args.backend_port,
        backend_port_count=args.backend_port_count,
        max_loaded_models=args.max_loaded_models,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        attention_backend=args.attention_backend,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.backend_shutdown_timeout,
        idle_seconds=args.idle_seconds,
        context_length_override=args.context_length,
        vllm_args=tuple(args.vllm_arg),
    )
    preload_models = [catalog.canonicalize(name) for name in args.preload_model]
    if args.preload and default_model not in preload_models:
        preload_models.insert(0, default_model)
    app = create_app(
        catalog,
        manager,
        preload_model=preload_models or None,
        default_model=default_model,
        pinned_default=bool(raw_default),
        catalog_path=args.config_file,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=args.graceful_shutdown_timeout,
    )
