import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightllm.server import api_http_rl


def _client(monkeypatch, enable_rl):
    monkeypatch.setattr(api_http_rl, "get_env_start_args", lambda: SimpleNamespace(enable_rl=enable_rl))
    app = FastAPI()
    app.include_router(api_http_rl.router)
    return TestClient(app)


@pytest.mark.parametrize(
    "method,path",
    [(method, route.path) for route in api_http_rl.router.routes for method in route.methods],
)
def test_rl_routes_not_found_when_rl_disabled(monkeypatch, method, path):
    client = _client(monkeypatch, enable_rl=False)

    response = client.request(method, path, json={})
    assert response.status_code == 404


def test_rl_routes_reachable_when_rl_enabled(monkeypatch):
    async def pause_generation():
        pass

    async def flush_cache(request):
        return SimpleNamespace(success=True, msg="")

    httpserver_manager = SimpleNamespace(pause_generation=pause_generation, flush_cache=flush_cache)
    monkeypatch.setitem(
        sys.modules,
        "lightllm.server.api_http",
        SimpleNamespace(g_objs=SimpleNamespace(httpserver_manager=httpserver_manager)),
    )
    client = _client(monkeypatch, enable_rl=True)

    assert client.post("/pause_generation").status_code == 200
    assert client.post("/flush_cache").status_code == 200
