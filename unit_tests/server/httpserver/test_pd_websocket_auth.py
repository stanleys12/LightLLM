import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from lightllm.server import api_http, api_http_pd
from lightllm.server.httpserver import pd_loop
from lightllm.server.pd_auth import get_pd_auth_headers
from lightllm.server.pd_io_struct import PD_Master_Obj
from lightllm.utils.envs_utils import get_pd_auth_token


@pytest.fixture
def pd_auth_token(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_AUTH_TOKEN", "secret")
    get_pd_auth_token.cache_clear()
    yield "secret"
    get_pd_auth_token.cache_clear()


@pytest.fixture
def client(monkeypatch):
    manager = SimpleNamespace(register_pd=AsyncMock(), remove_pd=AsyncMock(), update_req_status=AsyncMock())
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(httpserver_manager=manager))
    loads = Mock(side_effect=AssertionError("unauthenticated payload was unpickled"))
    monkeypatch.setattr(api_http_pd.pickle, "loads", loads)
    app = FastAPI()
    app.include_router(api_http_pd.router)
    with TestClient(app) as client:
        yield client, manager


@pytest.mark.parametrize("path", ["/pd_register", "/kv_move_status"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}])
def test_unauthenticated_peer_is_rejected_before_accept(pd_auth_token, client, path, headers):
    client, manager = client
    with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect(path, headers=headers):
        pass
    assert exc_info.value.code == 1008
    manager.register_pd.assert_not_awaited()
    manager.update_req_status.assert_not_awaited()


def test_authenticated_peer_registers(pd_auth_token, client):
    client, manager = client
    regist_json = {"node_id": 1, "mode": "prefill"}
    with client.websocket_connect("/pd_register", headers=get_pd_auth_headers()) as websocket:
        websocket.send_text(json.dumps(regist_json))
        websocket.close()
    manager.register_pd.assert_awaited_once()
    assert manager.register_pd.await_args.args[0] == regist_json


def test_unset_token_keeps_unauthenticated_registration(monkeypatch, client):
    monkeypatch.delenv("LIGHTLLM_PD_AUTH_TOKEN", raising=False)
    get_pd_auth_token.cache_clear()
    client, manager = client
    with client.websocket_connect("/pd_register") as websocket:
        websocket.send_text(json.dumps({"node_id": 1, "mode": "prefill"}))
        websocket.close()
    manager.register_pd.assert_awaited_once()


@pytest.mark.parametrize("token,expected", [("secret", {"Authorization": "Bearer secret"}), (None, None)])
def test_pd_node_sends_token_when_registering(monkeypatch, token, expected):
    if token is None:
        monkeypatch.delenv("LIGHTLLM_PD_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_PD_AUTH_TOKEN", token)
    get_pd_auth_token.cache_clear()
    seen = {}

    @asynccontextmanager
    async def connect(uri, **kwargs):
        seen[uri] = kwargs["additional_headers"]
        raise asyncio.CancelledError()
        yield

    monkeypatch.setattr(pd_loop.websockets, "connect", connect)
    manager = SimpleNamespace(args=SimpleNamespace(pd_node_id=1))
    asyncio.run(pd_loop._pd_handle_task(manager, PD_Master_Obj(1, "master:8000"), None))
    get_pd_auth_token.cache_clear()
    assert seen == {"ws://master:8000/pd_register": expected}
