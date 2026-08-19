from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .catalog import ModelCatalog
from .manager import ModelManager


def _default_preset_file() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "llama.cpp" / "models-preset.ini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve llama.cpp presets through SGLang on an OpenAI API.")
    parser.add_argument("--host", default=os.getenv("SGLANG_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SGLANG_SERVICE_PORT", "11432")))
    parser.add_argument(
        "--preset-file",
        type=Path,
        default=Path(os.getenv("SGLANG_PRESET_FILE", str(_default_preset_file()))),
        help="llama.cpp models-preset.ini file",
    )
    parser.add_argument("--default-model", default=os.getenv("SGLANG_DEFAULT_MODEL"))
    parser.add_argument("--preload", action="store_true", help="load the default model before accepting requests")
    parser.add_argument("--backend-port", type=int, default=int(os.getenv("SGLANG_BACKEND_PORT", "30000")))
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=int(os.getenv("SGLANG_TP_SIZE", "2")),
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp-size",
        type=int,
        default=int(os.getenv("SGLANG_DP_SIZE", "1")),
        help="number of SGLang data-parallel replicas; use 2 when the model fits on one GPU",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.90")),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=int(os.getenv("SGLANG_MAX_RUNNING_REQUESTS", "256")),
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=int(os.getenv("SGLANG_CHUNKED_PREFILL_SIZE", "4096")),
    )
    parser.add_argument(
        "--cuda-graph-backend-prefill",
        choices=("full", "breakable", "tc_piecewise", "disabled"),
        default=os.getenv("SGLANG_CUDA_GRAPH_BACKEND_PREFILL", "tc_piecewise"),
        help="prefill CUDA graph backend; tc_piecewise is compatible with this Blackwell setup",
    )
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("SGLANG_ATTENTION_BACKEND", "triton"),
        help="attention kernel backend passed to every SGLang model",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.environ["SGLANG_CONTEXT_LENGTH"]) if "SGLANG_CONTEXT_LENGTH" in os.environ else None,
        help="override the context length in every preset",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=float(os.getenv("SGLANG_STARTUP_TIMEOUT", "900")),
    )
    parser.add_argument(
        "--graceful-shutdown-timeout",
        type=int,
        default=int(os.getenv("SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT", "30")),
        help="maximum seconds Uvicorn waits for active requests during shutdown",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=float(os.getenv("SGLANG_MODEL_IDLE_SECONDS", "0")),
        help="unload an idle model after this many seconds; 0 keeps it loaded",
    )
    parser.add_argument(
        "--sglang-arg",
        action="append",
        default=[],
        help="extra argument passed to every SGLang backend; repeat as needed",
    )
    parser.add_argument("--log-level", default=os.getenv("SGLANG_SERVICE_LOG_LEVEL", "info"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    catalog = ModelCatalog.from_preset_file(args.preset_file)
    default_model = args.default_model or catalog.default.name
    catalog.get(default_model)  # validate before binding the public port
    manager = ModelManager(
        catalog,
        backend_port=args.backend_port,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_backend_prefill=args.cuda_graph_backend_prefill,
        attention_backend=args.attention_backend,
        startup_timeout=args.startup_timeout,
        idle_seconds=args.idle_seconds,
        context_length_override=args.context_length,
        sglang_args=tuple(args.sglang_arg),
    )
    app = create_app(catalog, manager, preload_model=default_model if args.preload else None)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=args.graceful_shutdown_timeout,
    )
