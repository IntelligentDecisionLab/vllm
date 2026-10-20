from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from vllm_gateway.app import create_app
from vllm_gateway.catalog import ModelCatalog, ModelSpec
from vllm_gateway.diagnostics import BackendStartupError, diagnostic_from_exception
from vllm_gateway.manager import Backend, BackendBusyError, BackendLease, BackendState


class AliveProcess:
    pid = 4242
    returncode = None


class FakeAsyncStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content


class FakeManager:
    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog
        self.backends = {
            spec.name: Backend(
                spec=spec,
                port=31000 + index,
                state=BackendState.READY,
                process=AliveProcess(),  # type: ignore[arg-type]
            )
            for index, spec in enumerate(catalog.specs)
        }
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self.busy: set[str] = set()

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def acquire(self, name: str | None) -> BackendLease:
        spec = self.catalog.get(name)
        backend = self.backends[spec.name]
        backend.active_requests += 1
        return BackendLease(self, backend)  # type: ignore[arg-type]

    async def release(self, backend: Backend) -> None:
        backend.active_requests = max(0, backend.active_requests - 1)

    async def send(
        self,
        backend: Backend,
        method: str,
        path: str,
        *,
        query: str = "",
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> httpx.Response:
        payload = json.loads(content) if content else {}
        self.sent.append((backend.spec.name, path, payload))
        request = httpx.Request(method, f"http://127.0.0.1:{backend.port}/{path}")
        if payload.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "server": "fake"},
                stream=FakeAsyncStream(f"data: {backend.spec.name}\n\n".encode()),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "server": "fake"},
            json={"backend": backend.spec.name, "model": payload.get("model")},
            request=request,
        )

    async def health(self) -> dict[str, Any]:
        loaded = sorted(self.backends)
        return {
            "status": "ok",
            "service": "vllm-model-router",
            "loaded_model": loaded[0],
            "loaded_models": loaded,
            "active_requests": sum(item.active_requests for item in self.backends.values()),
            "models": loaded,
            "resident_models": [],
            "capacity": {"max_loaded_models": 2, "resident_count": 2},
            "stopping": False,
        }

    async def load(self, name: str | None) -> dict[str, Any]:
        return await self.backend_status(name)

    async def unload(self, name: str | None) -> bool:
        spec = self.catalog.get(name)
        if spec.name in self.busy:
            raise BackendBusyError(f"model {spec.name!r} has active requests")
        return True

    async def backend_status(self, name: str | None) -> dict[str, Any]:
        spec = self.catalog.get(name)
        backend = self.backends[spec.name]
        return {
            "name": spec.name,
            "state": backend.state.value,
            "port": backend.port,
            "pid": 4242,
            "active_requests": backend.active_requests,
            "last_used": backend.last_used,
            "error": None,
        }


def make_app() -> tuple[TestClient, FakeManager]:
    catalog = ModelCatalog(
        [
            ModelSpec(name="model-a", model_path="/m/a", repo_id="org/a", context_length=4096),
            ModelSpec(name="model-b", model_path="/m/b", repo_id="org/b", context_length=8192),
        ]
    )
    manager = FakeManager(catalog)
    app = create_app(catalog, manager)  # type: ignore[arg-type]
    return TestClient(app), manager


def test_proxy_routes_each_model_to_its_resident_backend() -> None:
    client, manager = make_app()
    with client:
        response_a = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "A"}]},
        )
        response_b = client.post(
            "/v1/chat/completions",
            json={"model": "model-b", "messages": [{"role": "user", "content": "B"}]},
        )

    assert response_a.json() == {"backend": "model-a", "model": "model-a"}
    assert response_b.json() == {"backend": "model-b", "model": "model-b"}
    assert manager.sent[0][0] == "model-a"
    assert manager.sent[0][2]["model"] == "model-a"
    assert manager.sent[1][0] == "model-b"
    assert manager.sent[1][2]["model"] == "model-b"
    assert all(backend.active_requests == 0 for backend in manager.backends.values())
    assert "server" not in response_a.headers


def test_streaming_response_releases_lease() -> None:
    client, manager = make_app()
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        ) as response:
            assert response.status_code == 200
            assert "data: model-a" in "".join(response.iter_text())

    assert manager.backends["model-a"].active_requests == 0


def test_models_endpoint_marks_multiple_models_loaded() -> None:
    client, _ = make_app()
    with client:
        response = client.get("/v1/models")
        ready = client.get("/ready")

    assert response.status_code == 200
    assert {item["id"] for item in response.json()["data"] if item["loaded"]} == {
        "model-a",
        "model-b",
    }
    assert ready.status_code == 200


def test_unknown_admin_routes_do_not_proxy() -> None:
    client, manager = make_app()
    with TestClient(client.app) as public:
        assert public.get("/health").status_code == 200
        assert public.get("/ready").status_code == 200
        assert public.get("/v1/models").status_code == 200
        assert public.post("/v1/chat/completions", json={"model": "model-a", "messages": []}).status_code == 200
        for path in ("/admin/not-an-operation", "/Admin/models", "http://testserver//admin/models"):
            assert public.post(path).status_code == 404
    assert [path for _, path, _ in manager.sent] == ["v1/chat/completions"]


def test_local_management_load_and_unload() -> None:
    client, manager = make_app()
    with client:
        listed = client.get("/admin/models")
        loaded = client.post("/admin/models/model-a/load")
        unloaded = client.post("/admin/models/model-a/unload")
        manager.busy.add("model-b")
        busy = client.post("/admin/models/model-b/unload")

    assert listed.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["name"] == "model-a"
    assert unloaded.status_code == 200
    assert unloaded.json()["unloaded"] is True
    assert busy.status_code == 409
    assert busy.json()["error"]["type"] == "model_busy"


def test_load_and_proxy_share_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = make_app()
    diagnostic = diagnostic_from_exception(RuntimeError("secret"), code="kv_cache_memory_insufficient")

    async def failed(*args: object) -> None:
        raise BackendStartupError("secret=/home/private traceback", diagnostic)

    monkeypatch.setattr(manager, "load", failed)
    monkeypatch.setattr(manager, "acquire", failed)
    with client:
        admin = client.post("/admin/models/model-a/load")
        public = client.post("/v1/chat/completions", json={"model": "model-a", "messages": []})
    assert admin.status_code == public.status_code == 503
    assert admin.json()["error"] == {"type": "backend_unavailable", **diagnostic.as_dict()}
    assert public.json()["error"] == admin.json()["error"]
    assert "secret" not in admin.text + public.text
    assert "/home" not in admin.text + public.text


@pytest.mark.parametrize("exception", [RuntimeError, FileNotFoundError, ValueError, TimeoutError])
def test_generic_startup_errors_never_echo_internal_text(
    monkeypatch: pytest.MonkeyPatch,
    exception: type[Exception],
) -> None:
    client, manager = make_app()

    async def failed(*args: object) -> None:
        raise exception("credential-secret /home/private http://127.0.0.1:9999")

    monkeypatch.setattr(manager, "load", failed)
    monkeypatch.setattr(manager, "acquire", failed)
    with client:
        for path, body in (("/admin/models/model-a/load", None), ("/v1/chat/completions", {"model": "model-a"})):
            response = client.post(path, json=body)
            assert response.status_code == 503
            assert len(response.json()["error"]["diagnostic_id"]) == 32
            assert "credential-secret" not in response.text
            assert "/home/private" not in response.text
            assert "127.0.0.1" not in response.text


def test_health_and_ready_report_backend_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = make_app()
    diagnostic = diagnostic_from_exception(RuntimeError("secret"), code="cuda_out_of_memory")
    status = {
        "loaded_models": [],
        "resident_models": [
            {"name": "model-a", "state": "failed", "error": diagnostic.message, "diagnostic": diagnostic.as_dict()}
        ],
    }

    async def health() -> dict[str, object]:
        return status

    monkeypatch.setattr(manager, "health", health)
    with client:
        for path in ("/health", "/ready"):
            model = client.get(path).json()["resident_models"][0]
            assert model["diagnostic"] == diagnostic.as_dict()
            assert model["error"] == diagnostic.message


def test_proxy_transport_failure_is_safe_and_releases_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = make_app()

    async def send(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("secret http://127.0.0.1:9999 /private")

    monkeypatch.setattr(manager, "send", send)
    with client:
        response = client.post("/v1/chat/completions", json={"model": "model-a"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_request_failed"
    assert len(response.json()["error"]["diagnostic_id"]) == 32
    assert "secret" not in response.text
    assert "127.0.0.1" not in response.text
    assert manager.backends["model-a"].active_requests == 0


def test_missing_or_blank_model_is_rejected() -> None:
    client, manager = make_app()
    with client:
        for body in ({"messages": []}, {"model": None}, {"model": "  "}):
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == 400
            assert "model is required" in response.json()["error"]["message"]
        assert (
            client.post("/v1/chat/completions", content=b"x", headers={"content-type": "text/plain"}).status_code == 400
        )
    assert manager.sent == []


def test_invalid_json_and_non_string_model_are_rejected() -> None:
    client, _ = make_app()
    with client:
        bad = client.post("/v1/chat/completions", content=b"{bad-json", headers={"content-type": "application/json"})
        assert bad.status_code == 400
        assert bad.json()["error"]["message"].startswith("invalid JSON body")
        for invalid_model in (12345, 0, False, True, ["model-a"], {"name": "model-a"}):
            response = client.post("/v1/chat/completions", json={"model": invalid_model})
            assert response.status_code == 400
            assert response.json()["error"]["message"] == "model must be a string"


def test_only_exact_aliases_route() -> None:
    client, manager = make_app()
    with client:
        for name in ("org/a", "Model-A", "unknown"):
            response = client.post("/v1/chat/completions", json={"model": name})
            assert response.status_code == 404
            assert f"unknown model {name!r}" in response.json()["error"]["message"]
    assert manager.sent == []


def test_get_single_model() -> None:
    client, _ = make_app()
    with client:
        found = client.get("/v1/models/model-a")
        missing = client.get("/v1/models/nonexistent")

    assert found.status_code == 200
    assert found.json()["id"] == "model-a"
    assert found.json()["loaded"] is True
    assert found.json()["metadata"] == {"repo_id": "org/a", "context_length": 4096}
    assert missing.status_code == 404
    assert "unknown model 'nonexistent'" in missing.json()["error"]["message"]


def test_unhandled_proxy_error_releases_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    client, manager = make_app()

    async def crash(*args: object, **kwargs: object) -> httpx.Response:
        raise RuntimeError("unexpected fatal crash")

    monkeypatch.setattr(manager, "send", crash)
    with TestClient(client.app, raise_server_exceptions=False) as raw:
        response = raw.post("/v1/chat/completions", json={"model": "model-a"})

    assert response.status_code == 500
    assert manager.backends["model-a"].active_requests == 0
