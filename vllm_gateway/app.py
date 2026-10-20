from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .catalog import ModelCatalog, scan_local_models
from .downloads import DownloadError, DownloadManager
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
# First path segments owned by the gateway. They are never proxied to a backend.
RESERVED_PREFIXES = {"models", "aliases"}


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"message": message, "type": error_type}})


def _invalid_request(message: str) -> JSONResponse:
    return _error(400, message, "invalid_request_error")


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


def create_app(catalog: ModelCatalog, manager: ModelManager, downloads: DownloadManager | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await manager.start()
            yield
        finally:
            if downloads is not None:
                await downloads.shutdown()
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

    async def alias_view(name: str) -> dict[str, object]:
        spec = app.state.catalog.get(name)
        status = await manager.backend_status(name)
        return {
            "alias": spec.name,
            "repo_id": spec.repo_id,
            "context_length": spec.context_length,
            "state": status["state"],
            "port": status["port"],
            "active_requests": status["active_requests"],
            "last_used": status["last_used"],
            "error": status["error"],
            "diagnostic": status["diagnostic"],
        }

    @app.get("/models/local")
    async def local_models() -> JSONResponse:
        hub_cache = app.state.catalog.hub_cache
        if hub_cache is None:
            return JSONResponse(content={"object": "list", "data": []})
        infos = await asyncio.to_thread(scan_local_models, hub_cache)
        downloading = downloads.active_repos() if downloads is not None else set()
        data = []
        for info in infos:
            if info.repo_id in downloading:
                continue
            aliases = [spec.name for spec in app.state.catalog.specs if spec.repo_id == info.repo_id]
            data.append({**info.as_dict(), "aliases": aliases})
        return JSONResponse(content={"object": "list", "data": data})

    @app.post("/models/download")
    async def start_download(request: Request) -> JSONResponse:
        if downloads is None:
            return _error(501, "downloads are not enabled", "not_implemented")
        try:
            body = await request.json()
        except ValueError:
            return _invalid_request("body must be JSON")
        if not isinstance(body, dict):
            return _invalid_request("body must be a JSON object")
        try:
            job = await downloads.submit(body.get("repo_id"), body.get("revision"))
        except DownloadError as exc:
            return _error(exc.status_code, exc.message, "download_error")
        return JSONResponse(status_code=202, content=job.as_dict())

    @app.get("/models/downloads")
    async def list_downloads() -> JSONResponse:
        jobs = downloads.jobs() if downloads is not None else []
        return JSONResponse(content={"object": "list", "data": [job.as_dict() for job in jobs]})

    @app.get("/models/downloads/{job_id}")
    async def get_download(job_id: str) -> JSONResponse:
        try:
            if downloads is None:
                raise KeyError(f"unknown download job {job_id!r}")
            return JSONResponse(content=downloads.get(job_id).as_dict())
        except KeyError as exc:
            return _error(404, str(exc), "invalid_request_error")

    @app.get("/aliases")
    async def list_aliases() -> JSONResponse:
        data = [await alias_view(spec.name) for spec in app.state.catalog.specs]
        return JSONResponse(content={"object": "list", "data": data})

    @app.post("/aliases", status_code=201)
    async def create_alias(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return _invalid_request("body must be JSON")
        if not isinstance(body, dict):
            return _invalid_request("body must be a JSON object")
        try:
            spec = app.state.catalog.build(body.get("alias"), body.get("model"), body.get("context_length"))
            await manager.validate_fits(spec)
            app.state.catalog.insert(spec)
        except KeyError as exc:
            return _error(404, str(exc), "invalid_request_error")
        except ValueError as exc:
            return _invalid_request(str(exc))
        return JSONResponse(status_code=201, content=await alias_view(spec.name))

    @app.delete("/aliases/{alias}")
    async def delete_alias(alias: str) -> JSONResponse:
        try:
            await manager.forget(alias)
        except KeyError as exc:
            return _error(404, str(exc), "invalid_request_error")
        except BackendBusyError as exc:
            return _error(409, str(exc), "model_busy")
        app.state.catalog.remove(alias)
        return JSONResponse(content={"deleted": alias})

    @app.post("/aliases/{alias}/load")
    async def load_alias(alias: str) -> JSONResponse:
        try:
            await manager.load(alias)
            return JSONResponse(content=await alias_view(alias))
        except KeyError as exc:
            return _error(404, str(exc), "invalid_request_error")
        except (CapacityError, FileNotFoundError, PortUnavailable, RuntimeError, TimeoutError, ValueError) as exc:
            LOGGER.exception("failed to load model %s", alias)
            return _backend_error(exc)

    @app.post("/aliases/{alias}/unload")
    async def unload_alias(alias: str) -> JSONResponse:
        try:
            unloaded = await manager.unload(alias)
            return JSONResponse(content={"unloaded": unloaded, "alias": await alias_view(alias)})
        except KeyError as exc:
            return _error(404, str(exc), "invalid_request_error")
        except BackendBusyError as exc:
            return _error(409, str(exc), "model_busy")

    @app.api_route("/{path:path}", methods=PROXY_METHODS)
    async def proxy(path: str, request: Request) -> Response:
        if path.strip("/").split("/", 1)[0].lower() in RESERVED_PREFIXES:
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
                return _invalid_request(f"invalid JSON body: {exc.msg}")
            if isinstance(decoded, dict):
                payload = decoded
                raw_model = decoded.get("model")
                if raw_model is not None and not isinstance(raw_model, str):
                    return _invalid_request("model must be a string")
                model_name = raw_model.strip() if raw_model else None

        if not model_name:
            return _invalid_request("model is required and must be one of your aliases")

        try:
            spec = app.state.catalog.get(model_name)
            if payload is not None:
                payload["model"] = spec.name
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
