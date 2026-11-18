import json
import os
from types import SimpleNamespace

import httpx
import llm
import pytest
from fakes import FakeClient, FakeResponse, models_payload

from llm_strata import connection

# Keep stray environment variables from the developer's shell out of the
# hermetic discovery tests (STRATA_RUNTIME_DIR and friends change the results).
_strata_env = (
    "STRATA_BASE_URL",
    "STRATA_API_KEY",
    "STRATA_RUNTIME_DIR",
    "LOCALAPPDATA",
    "XDG_DATA_HOME",
)


@pytest.fixture(autouse=True)
def clean_strata_env(monkeypatch):
    for name in _strata_env:
        monkeypatch.delenv(name, raising=False)


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


def test_home_helper_is_the_real_implementation():
    from pathlib import Path

    assert isinstance(connection._home(), Path)


def test_runtime_dirs_windows(tmp_path, monkeypatch):
    fake_os = SimpleNamespace(name="nt", environ=os.environ)
    monkeypatch.setattr(connection, "os", fake_os)
    monkeypatch.setattr(connection, "_home", lambda: tmp_path / "home")
    monkeypatch.setenv("STRATA_RUNTIME_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    assert connection.runtime_dirs() == [
        tmp_path / "custom",
        tmp_path / "AppData" / "strata" / "runtime",
        tmp_path / "home" / ".strata" / "runtime",
    ]

    # Without LOCALAPPDATA the ~/.strata fallback still applies
    monkeypatch.delenv("LOCALAPPDATA")
    assert connection.runtime_dirs() == [
        tmp_path / "custom",
        tmp_path / "home" / ".strata" / "runtime",
    ]


def test_runtime_dirs_posix_xdg_and_home_fallback(tmp_path, monkeypatch):
    fake_os = SimpleNamespace(name="posix", environ=os.environ)
    monkeypatch.setattr(connection, "os", fake_os)
    monkeypatch.setattr(connection, "_home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert connection.runtime_dirs() == [
        tmp_path / "xdg" / "strata" / "runtime",
        tmp_path / "home" / ".strata" / "runtime",
    ]

    # No XDG_DATA_HOME -> ~/.local/share under home
    monkeypatch.delenv("XDG_DATA_HOME")
    assert connection.runtime_dirs() == [
        tmp_path / "home" / ".local" / "share" / "strata" / "runtime",
        tmp_path / "home" / ".strata" / "runtime",
    ]


def test_base_candidates_skips_missing_dirs_and_broken_files(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(connection, "runtime_dirs", lambda: [tmp_path / "missing"])
    assert connection.base_candidates() == [("http://127.0.0.1:8080/v1", "default")]

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "app-1.ready.json").write_text("{ not valid json")
    monkeypatch.setattr(connection, "runtime_dirs", lambda: [runtime])
    assert connection.base_candidates() == [("http://127.0.0.1:8080/v1", "default")]


def test_key_candidates_tolerate_broken_strata_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data = home / ".strata" / "data"
    native = data / "native-instances"
    native.mkdir(parents=True)
    (data / "state.json").write_text("{ broken json")
    (native / "bad.json").write_text("also broken")
    (native / "good.json").write_text('{"api_key": "good-native"}')
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(connection, "_home", lambda: home)

    keys = connection.key_candidates()
    assert [k for k, _ in keys] == ["good-native", ""]
    assert keys[0][1].startswith("native:good")


def test_probe_non_json_body_and_entries_without_id(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    class BrokenJson(FakeResponse):
        def json(self):
            raise ValueError("no json")

    FakeClient.responses = [BrokenJson(200, None)]
    assert connection.probe("http://x/v1", "k") == (True, None)

    FakeClient.responses = [
        FakeResponse(200, models_payload([{"label": "no id"}, {"id": "ok-model"}]))
    ]
    reachable, models = connection.probe("http://x/v1", "k")
    assert reachable is True
    assert models is not None
    assert [m.id for m in models] == ["ok-model"]


def test_probe_sends_bearer_key_only_when_key_present(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    FakeClient.responses = [FakeResponse(200, models_payload([{"id": "m"}]))]
    connection.probe("http://x/v1", "topsecret")
    request = FakeClient.instances[-1].requests[0]
    assert request.headers["authorization"] == "Bearer topsecret"

    FakeClient.responses = [FakeResponse(200, models_payload([{"id": "m"}]))]
    connection.probe("http://x/v1", "")
    request = FakeClient.instances[-1].requests[0]
    assert "authorization" not in request.headers


def test_discover_dedups_repeated_base_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "get_key", lambda **kw: None)
    monkeypatch.setattr(
        connection,
        "base_candidates",
        lambda: [("http://x/v1", "s1"), ("http://x/v1", "s2")],
    )
    monkeypatch.setattr(connection, "key_candidates", lambda: [("", "no key")])

    calls = []

    def fake_probe(url, key):
        calls.append((url, key))
        return True, ()

    monkeypatch.setattr(connection, "probe", fake_probe)
    endpoints = connection.discover()
    assert calls == [("http://x/v1", "")]
    assert len(endpoints) == 1
    assert endpoints[0].source == "s1 + no key"


def test_health_strips_v1_and_survives_errors(tmp_path, monkeypatch):
    ep = connection.Endpoint("http://x/v1", "key", "test", ())
    FakeClient.responses = [
        FakeResponse(200, {"status": "ok", "model": "m", "loaded": True})
    ]
    monkeypatch.setattr(httpx, "Client", FakeClient)

    assert connection.health(ep) == {"status": "ok", "model": "m", "loaded": True}
    request = FakeClient.instances[-1].requests[0]
    assert str(request.url) == "http://x/health"  # /v1 stripped, no key sent
    assert "authorization" not in request.headers

    FakeClient.responses = [FakeResponse(503, {})]
    assert connection.health(ep) is None

    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", boom)
    assert connection.health(ep) is None
