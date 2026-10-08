from types import SimpleNamespace

from fastapi.testclient import TestClient

from lightllm.server.core.objs.start_args_type import StartArgs

RL_PATHS = ["/abort_request", "/pause_generation", "/flush_cache", "/update_weights_from_tensor"]


def _post_rl_paths(monkeypatch, enable_rl):
    from lightllm.server import api_http

    monkeypatch.setattr(api_http.app.router, "routes", list(api_http.app.router.routes))
    monkeypatch.setattr(api_http, "init_tokenizer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_http.SamplingParams, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.CompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.ChatCompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http, "MetricClient", lambda _port: SimpleNamespace(counter_inc=lambda *_args: None))
    monkeypatch.setattr(api_http, "get_shm_port_args", lambda: SimpleNamespace(metric_port=1234))
    monkeypatch.setattr(api_http, "HttpServerManager", lambda args: SimpleNamespace(rl_controller=None))
    monkeypatch.setattr(api_http, "TokenLoad", lambda _name, _dp_size: object())
    monkeypatch.setattr(api_http, "get_unique_server_name", lambda: "test-server")
    monkeypatch.setattr(api_http.setproctitle, "setproctitle", lambda _title: None)

    global_objs = api_http.G_Objs(app=api_http.app)
    global_objs.set_args(StartArgs(model_dir="/tmp/test-model", enable_rl=enable_rl))
    monkeypatch.setattr(api_http, "g_objs", global_objs)

    client = TestClient(api_http.app, raise_server_exceptions=False)
    return {client.post(path, json={}).status_code for path in RL_PATHS}


def test_rl_routes_are_not_mounted_without_enable_rl(monkeypatch):
    assert _post_rl_paths(monkeypatch, enable_rl=False) == {404}


def test_rl_routes_are_mounted_with_enable_rl(monkeypatch):
    assert 404 not in _post_rl_paths(monkeypatch, enable_rl=True)
