from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .catalog import ModelCatalog, ModelSpec
from .manager import BackendLease, ModelManager

LOGGER = logging.getLogger("sglang_service")
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
    """Translate llama.cpp preset defaults into SGLang request options."""

    if spec.reasoning != "off" or not isinstance(payload.get("messages"), list):
        return
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
        payload["chat_template_kwargs"] = chat_template_kwargs
    if isinstance(chat_template_kwargs, dict):
        chat_template_kwargs.setdefault("enable_thinking", False)


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def create_app(
    catalog: ModelCatalog,
    manager: ModelManager,
    *,
    preload_model: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await manager.start(preload_model)
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="SGLang Model Service",
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
        if status["loaded_model"] is None:
            return JSONResponse(status_code=503, content={**status, "status": "not_ready"})
        return JSONResponse(content=status)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        status = await manager.health()
        return {"object": "list", "data": catalog.public_models(status["loaded_model"])}

    @app.api_route("/{path:path}", methods=PROXY_METHODS)
    async def proxy(path: str, request: Request) -> Response:
        body = await request.body()
        outgoing_body = body
        model_name: str | None = None
        payload: dict[str, object] | None = None

        content_type = request.headers.get("content-type", "")
        if body and "application/json" in content_type:
            try:
                decoded = json.loads(body)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc.msg}") from exc
            if isinstance(decoded, dict):
                payload = decoded
                model_name = decoded.get("model")
                if not model_name:
                    decoded["model"] = catalog.default.name
                    model_name = catalog.default.name

        if model_name is not None and not isinstance(model_name, str):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "model must be a string", "type": "invalid_request_error"}},
            )

        if model_name is None:
            model_name = catalog.default.name

        try:
            spec = catalog.get(model_name)
            if payload is not None:
                _apply_model_defaults(payload, spec)
                outgoing_body = json.dumps(payload, separators=(",", ":")).encode()
            lease = await manager.acquire(spec.name)
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
            LOGGER.exception("failed to start model %s", model_name)
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(exc), "type": "backend_unavailable"}},
            )

        backend = lease.backend
        url = f"http://127.0.0.1:{backend.port}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["content-length"] = str(len(outgoing_body))

        try:
            upstream_request = manager._client.build_request(
                request.method,
                url,
                headers=headers,
                content=outgoing_body,
            )
            upstream = await manager._client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            await lease.release()
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"SGLang backend request failed: {exc}", "type": "bad_gateway"}},
            )

        response_headers = _forward_headers(upstream.headers)
        if "text/event-stream" in upstream.headers.get("content-type", ""):

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
            return Response(
                content=content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=None,
            )
        finally:
            await upstream.aclose()
            await lease.release()

    return app
