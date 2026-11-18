import json
import os

import httpx
import pytest

import llm
from llm_strata import connection

from fakes import FakeClient, FakeResponse, models_payload


def test_normalize_base():
    assert connection.normalize_base("http://x:1") == "http://x:1/v1"
    assert connection.normalize_base("http://x:1/") == "http://x:1/v1"
    assert connection.normalize_base("http://x:1/v1") == "http://x:1/v1"
    assert connection.normalize_base("http://x:1/v1/") == "http://x:1/v1"


def test_base_candidates_priority_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(connection.llm, "user_dir", lambda: tmp_path)
    connection.save_config({"base_url": "http://pinned/v1"})
    monkeypatch.setenv("STRATA_BASE_URL", "http://env-url")

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    f2 = runtime / "app-2.ready.json"
    f2.write_text(
        json.dumps({"api_state": "ready", "api_url": "http://127.0.0.1:2/v1"})
    )
    f3 = runtime / "app-3.ready.json"
    f3.write_text(json.dumps({"api_state": "crashed", "api_url": "http://dead"}))
    f1 = runtime / "app-1.ready.json"
    f1.write_text(json.dumps({"api_url": "http://127.0.0.1:99/v1"}))
    os.utime(f2, (3000, 3000))
    os.utime(f3, (2000, 2000))
    os.utime(f1, (1000, 1000))
    monkeypatch.setattr(connection, "runtime_dirs", lambda: [runtime])

    urls = [url for url, _ in connection.base_candidates()]
    assert urls == [
        "http://pinned/v1",
        "http://env-url/v1",
        "http://127.0.0.1:2/v1",
        "http://127.0.0.1:99/v1",
        "http://127.0.0.1:8080/v1",
    ]
    # Crashed ready-files are skipped
    assert "http://dead/v1" not in urls
    assert len(urls) == len(set(urls))


def test_key_candidates_priority(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".strata" / "data"
    state.mkdir(parents=True)
    (state / "state.json").write_text(json.dumps({"serving": {"apiKey": "sk-state"}}))
    native = state / "native-instances"
    native.mkdir()
    (native / "inst1.json").write_text(json.dumps({"api_key": "hex-native"}))

    monkeypatch.setattr(connection.llm, "user_dir", lambda: tmp_path)
    connection.save_config({"api_key": "sk-setup"})
    monkeypatch.setattr(connection, "_home", lambda: home)
    monkeypatch.setenv("STRATA_API_KEY", "sk-env")
    monkeypatch.setattr(llm, "get_key", lambda **kw: "sk-llmkeys")

    keys = [k for k, _ in connection.key_candidates()]
    assert keys[:5] == ["sk-setup", "sk-env", "sk-llmkeys", "sk-state", "hex-native"]
    assert keys[-1] == ""


def test_probe_parses_models(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    FakeClient.responses = [
        FakeResponse(
            200,
            models_payload(
                [
                    {"id": "full-model", "label": "Full Model"},
                    {"id": "short", "alias_of": "full-model"},
                ]
            ),
        )
    ]
    monkeypatch.setattr(httpx, "Client", FakeClient)

    reachable, models = connection.probe("http://x/v1", "key")
    assert reachable is True
    assert models is not None
    assert [m.id for m in models] == ["full-model", "short"]
    assert models[1].alias_of == "full-model"
    assert models[0].label == "Full Model"


def test_probe_unauthorized_and_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)

    FakeClient.responses = [FakeResponse(401, {"error": "missing or wrong API key"})]
    monkeypatch.setattr(httpx, "Client", FakeClient)
    reachable, models = connection.probe("http://x/v1", "wrong")
    assert (reachable, models) == (True, None)

    def boom(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", boom)
    assert connection.probe("http://x/v1", "key") == (False, None)


def test_discover_pairs_url_with_working_key(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(
        connection, "base_candidates", lambda: [("http://x/v1", "test")]
    )
    monkeypatch.setattr(
        connection,
        "key_candidates",
        lambda: [("bad", "k1"), ("good", "k2"), ("", "no key")],
    )
    FakeClient.responses = [
        FakeResponse(401, {}),
        FakeResponse(200, models_payload([{"id": "m1"}])),
    ]
    monkeypatch.setattr(httpx, "Client", FakeClient)

    endpoints = connection.discover()
    assert len(endpoints) == 1
    ep = endpoints[0]
    assert ep.api_key == "good"
    assert ep.source == "test + k2"
    assert [m.id for m in ep.models] == ["m1"]


def test_discover_down_server_tries_url_once(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(
        connection, "base_candidates", lambda: [("http://down/v1", "test")]
    )
    monkeypatch.setattr(
        connection, "key_candidates", lambda: [("a", "k1"), ("b", "k2")]
    )

    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", boom)
    assert connection.discover() == []
    assert len(calls) == 1


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    assert connection.load_config() == {}
    connection.save_config({"base_url": "http://a/v1", "api_key": "k"})
    assert connection.load_config() == {"base_url": "http://a/v1", "api_key": "k"}
    assert connection.clear_config() is True
    assert connection.clear_config() is False
