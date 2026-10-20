from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .catalog import ModelCatalog, default_hub_cache
from .manager import ModelManager


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Hugging Face models through vLLM on a shared OpenAI API.")
    parser.add_argument("--host", default=os.getenv("VLLM_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("VLLM_SERVICE_PORT", 11432))
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=Path(os.environ["HF_HUB_CACHE"]) if os.getenv("HF_HUB_CACHE") else default_hub_cache(),
        help="Hugging Face hub cache that holds the models",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.getenv("VLLM_STATE_DIR") or "~/.local/state/vllm-gateway").expanduser(),
        help="directory that stores aliases.json",
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
    parser.add_argument("--log-level", default=os.getenv("VLLM_SERVICE_LOG_LEVEL", "info"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    catalog = ModelCatalog.load(args.state_dir / "aliases.json", args.hub_cache)
    manager = ModelManager(
        catalog,
        backend_port=args.backend_port,
        backend_port_count=args.backend_port_count,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.backend_shutdown_timeout,
    )
    uvicorn.run(
        create_app(catalog, manager),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=args.graceful_shutdown_timeout,
    )
