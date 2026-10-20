from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .catalog import ModelCatalog, ModelSpec
from .manager import (
    BackendBusyError,
    CapacityError,
    ModelManager,
    PortUnavailable,
    diagnose_backend_error,
)

LOGGER = logging.getLogger("vllm_gateway")

# Upstream headers that FastAPI/uvicorn will regenerate — forwarding them
# produces duplicate headers (two `date`, two `server`, etc.)
UPSTREAM_GENERATED_HEADERS = {"date", "server"}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def _apply_model_defaults(payload: dict[str, object], spec: ModelSpec) -> None:
    """Translate catalog request defaults into vLLM request options."""

    if spec.reasoning != "off" or not isinstance(payload.get("messages"), list):
        return
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
        payload["chat_template_kwargs"] = chat_template_kwargs
    if isinstance(chat_template_kwargs, dict):
        chat_template_kwargs.setdefault("enable_thinking", False)


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() not in UPSTREAM_GENERATED_HEADERS
    }


def _backend_error(exc: BaseException) -> JSONResponse:
    diagnostic = diagnose_backend_error(exc)
    LOGGER.error("backend unavailable diagnostic_id=%s code=%s", diagnostic.diagnostic_id, diagnostic.code)
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "backend_unavailable",
                "code": diagnostic.code,
                "message": diagnostic.message,
                "diagnostic_id": diagnostic.diagnostic_id,
            }
        },
    )


def create_app(
    catalog: ModelCatalog,
    manager: ModelManager,
    *,
    preload_model: str | Iterable[str] | None = None,
    default_model: str | None = None,
    pinned_default: bool = False,
    catalog_path: str | Path | None = None,
) -> FastAPI:
    request_default = catalog.canonicalize(default_model)
    config_path = Path(catalog_path).expanduser().resolve() if catalog_path is not None else None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await manager.start(preload_model)
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="vLLM Model Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.catalog = catalog
    app.state.manager = manager

    @app.get("/health")
    async def health() -> dict[str, object]:
        return await manager.health()

    @app.get("/ready")
    async def ready() -> JSONResponse:
        status = await manager.health()
        if not status["loaded_models"]:
            return JSONResponse(status_code=503, content={**status, "status": "not_ready"})
        return JSONResponse(content=status)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        status = await manager.health()
        return {"object": "list", "data": app.state.catalog.public_models(status["loaded_models"])}

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> JSONResponse:
        status = await manager.health()
        try:
            spec = app.state.catalog.get(model_id)
            return JSONResponse(content=spec.public_dict(loaded=(spec.name in status["loaded_models"])))
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )

    @app.get("/admin/models")
    async def admin_models() -> JSONResponse:
        return JSONResponse(content=await manager.health())

    @app.post("/admin/models/{model_id:path}/load")
    async def load_model(model_id: str) -> JSONResponse:
        try:
            return JSONResponse(content=await manager.load(model_id))
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except (CapacityError, FileNotFoundError, PortUnavailable, RuntimeError, TimeoutError, ValueError) as exc:
            LOGGER.exception("failed to load model %s", model_id)
            return _backend_error(exc)

    @app.post("/admin/reload")
    async def reload_config() -> JSONResponse:
        if config_path is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "no config file was configured for this service",
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            new_catalog = ModelCatalog.from_file(config_path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("config reload from %s failed, keeping current config: %s", config_path, exc)
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"failed to load configuration: {exc}", "type": "invalid_request_error"}},
            )
        try:
            result = await manager.reload_catalog(new_catalog)
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(exc), "type": "backend_unavailable"}},
            )
        nonlocal request_default
        app.state.catalog = manager.catalog
        if not pinned_default:
            request_default = manager.catalog.default.name
        return JSONResponse(content=result)

    @app.post("/admin/models/{model_id:path}/unload")
    async def unload_model(model_id: str) -> JSONResponse:
        try:
            unloaded = await manager.unload(model_id)
            status = await manager.backend_status(model_id)
            return JSONResponse(content={"unloaded": unloaded, "model": status})
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except BackendBusyError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"message": str(exc), "type": "model_busy"}},
            )

    @app.api_route("/{path:path}", methods=PROXY_METHODS)
    async def proxy(path: str, request: Request) -> Response:
        normalized_path = path.strip("/").lower()
        if normalized_path == "admin" or normalized_path.startswith("admin/"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        body = await request.body()
        outgoing_body = body
        model_name: str | None = None
        payload: dict[str, object] | None = None

        content_type = request.headers.get("content-type", "")
        if body and "application/json" in content_type:
            try:
                decoded = json.loads(body)
            except json.JSONDecodeError as exc:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"invalid JSON body: {exc.msg}", "type": "invalid_request_error"}},
                )
            if isinstance(decoded, dict):
                payload = decoded
                if "model" in decoded and decoded["model"] is not None:
                    if not isinstance(decoded["model"], str):
                        return JSONResponse(
                            status_code=400,
                            content={"error": {"message": "model must be a string", "type": "invalid_request_error"}},
                        )
                    raw_model = decoded["model"].strip()
                    model_name = raw_model or request_default
                else:
                    model_name = request_default
                decoded["model"] = model_name

        if not model_name or not str(model_name).strip():
            model_name = request_default

        try:
            spec = app.state.catalog.get(model_name)
            if payload is not None:
                payload["model"] = spec.name
                _apply_model_defaults(payload, spec)
                outgoing_body = json.dumps(payload, separators=(",", ":")).encode()
            lease = await manager.acquire(spec.name)
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except (CapacityError, FileNotFoundError, PortUnavailable, RuntimeError, TimeoutError, ValueError) as exc:
            LOGGER.exception("failed to start model %s", model_name)
            return _backend_error(exc)

        backend = lease.backend
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}
        if outgoing_body:
            headers["content-length"] = str(len(outgoing_body))
        else:
            headers.pop("content-length", None)

        try:
            try:
                upstream = await manager.send(
                    backend,
                    request.method,
                    path,
                    query=request.url.query,
                    headers=headers,
                    content=outgoing_body,
                )
                LOGGER.debug(
                    "proxy %s %s → vLLM %s %d (%s)",
                    request.method,
                    str(request.url),
                    request.method,
                    upstream.status_code,
                    backend.spec.name,
                )
            except httpx.HTTPError as exc:
                await lease.release()
                diagnostic_id = uuid4().hex
                LOGGER.warning(
                    "proxy %s %s → vLLM HTTP error diagnostic_id=%s: %s",
                    request.method,
                    str(request.url),
                    diagnostic_id,
                    exc,
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": "The backend request failed; consult the gateway log with the diagnostic ID.",
                            "type": "bad_gateway",
                            "code": "backend_request_failed",
                            "diagnostic_id": diagnostic_id,
                        }
                    },
                )

            response_headers = _forward_headers(upstream.headers)
            if "text/event-stream" in upstream.headers.get("content-type", ""):
                if upstream.status_code >= 400:
                    # vLLM unexpectedly returned an error with SSE content-type.
                    # Read body and return as JSON error instead of an empty stream.
                    try:
                        err_body = await upstream.aread()
                        LOGGER.warning(
                            "vLLM returned %d with SSE content-type for %s %s: %s",
                            upstream.status_code,
                            request.method,
                            str(request.url),
                            err_body[:512],
                        )
                    except Exception:
                        err_body = b""
                    finally:
                        await upstream.aclose()
                        await lease.release()
                    return Response(
                        content=err_body,
                        status_code=upstream.status_code,
                        headers={"content-type": "application/json"},
                        media_type=None,
                    )

                async def stream() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        await lease.release()

                return StreamingResponse(
                    stream(),
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type=None,
                )

            try:
                content = await upstream.aread()
                # Log error responses for debugging
                if upstream.status_code >= 400:
                    LOGGER.warning(
                        "vLLM returned %d for %s %s: %s",
                        upstream.status_code,
                        request.method,
                        str(request.url),
                        content[:512],
                    )
                return Response(
                    content=content,
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type=None,
                )
            finally:
                await upstream.aclose()
                await lease.release()
        except BaseException:
            await lease.release()
            raise

    return app
