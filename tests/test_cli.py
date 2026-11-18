import click
from click.testing import CliRunner

import llm_strata.cli as cli_mod
from llm_strata.connection import Endpoint, ModelInfo


def build_cli():
    @click.group()
    def cli():
        pass

    cli_mod.add_commands(cli)
    return cli


def run(args, monkeypatch, endpoints=None, base_candidates=None, probe=None):
    if endpoints is not None:
        monkeypatch.setattr(cli_mod, "discover", lambda: endpoints)
    if base_candidates is not None:
        monkeypatch.setattr(cli_mod, "base_candidates", base_candidates)
    if probe is not None:
        monkeypatch.setattr(cli_mod, "probe", probe)
    result = CliRunner().invoke(build_cli(), args)
    return result


def ep_one():
    return Endpoint(
        base_url="http://x/v1",
        api_key="secret-key-abcdefgh",
        source="test",
        models=(
            ModelInfo(id="big-model", label="Big Model"),
            ModelInfo(id="big", alias_of="big-model"),
        ),
    )


def test_models_listsprimaries_and_aliases(monkeypatch):
    result = run(["strata", "models"], monkeypatch, endpoints=[ep_one()])
    assert result.exit_code == 0
    out = result.output
    assert "http://x/v1" in out
    assert "strata/big-model" in out
    assert "Big Model" in out
    assert "strata/big" in out  # as alias annotation
    assert out.count("strata/big-model") >= 1
    # alias-only entry is not listed as its own row
    assert "  strata/big\n" not in out


def test_models_down_message(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [])
    monkeypatch.setattr(
        cli_mod, "base_candidates", lambda: [("http://down/v1", "test")]
    )
    monkeypatch.setattr(cli_mod, "probe", lambda url, key: (False, None))
    result = run(["strata", "models"], monkeypatch)
    assert result.exit_code == 1
    assert "Could not reach any Strata server" in result.output


def test_models_up_but_unauthorized_message(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [])
    monkeypatch.setattr(
        cli_mod, "base_candidates", lambda: [("http://up/v1", "runtime:x.ready.json")]
    )
    monkeypatch.setattr(cli_mod, "probe", lambda url, key: (True, None))
    result = run(["strata", "models"], monkeypatch)
    assert result.exit_code == 1
    assert "no API key was accepted" in result.output
    assert "llm strata setup --url URL --key KEY" in result.output


def test_status_json_masks_key(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [ep_one()])
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})
    monkeypatch.setattr(cli_mod, "config_path", lambda: "/tmp/none")
    monkeypatch.setattr(cli_mod, "health", lambda endpoint: {"status": "ok", "loaded": True})
    result = run(["strata", "status", "--json"], monkeypatch)
    assert result.exit_code == 0
    import json

    payload = json.loads(result.output)
    (entry,) = payload["endpoints"]
    assert entry["api_key"] == "secret-key...efgh"
    assert "secret-key-abcdefgh" not in result.output
    assert entry["health"]["loaded"] is True


def test_setup_discover_and_forget(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "config_path", lambda: tmp_path / "strata.json")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})
    monkeypatch.setattr(cli_mod, "clear_config", lambda: False)
    monkeypatch.setattr(cli_mod, "discover", lambda: [ep_one()])
    saved = {}
    monkeypatch.setattr(cli_mod, "save_config", lambda cfg: saved.setdefault("cfg", cfg))
    monkeypatch.setattr(cli_mod, "key_candidates", lambda: [("secret-key-abcdefgh", "test")])
    monkeypatch.setattr(cli_mod, "probe", lambda url, key: (True, ep_one().models))

    result = run(["strata", "setup"], monkeypatch)
    assert result.exit_code == 0
    assert saved["cfg"]["base_url"] == "http://x/v1"
    assert saved["cfg"]["api_key"] == "secret-key-abcdefgh"
    assert "big-model" in result.output

    result = run(["strata", "forget"], monkeypatch)
    assert result.exit_code == 0


def test_models_json_output(monkeypatch):
    result = run(["strata", "models", "--json"], monkeypatch, endpoints=[ep_one()])
    assert result.exit_code == 0
    import json

    (entry,) = json.loads(result.output)
    assert entry["base_url"] == "http://x/v1"
    assert entry["source"] == "test"
    assert entry["api_key"] is True
    by_id = {m["id"]: m for m in entry["models"]}
    assert set(by_id) == {"big-model", "big"}
    assert by_id["big-model"]["model_id"] == "strata/big-model"
    assert by_id["big-model"]["label"] == "Big Model"
    assert by_id["big"]["alias_of"] == "big-model"


def test_models_endpoint_without_rows(monkeypatch):
    empty = Endpoint(base_url="http://x/v1", api_key="k", source="test", models=())
    result = run(["strata", "models"], monkeypatch, endpoints=[empty])
    assert result.exit_code == 0
    assert "(no models loaded)" in result.output


def test_mask_short_and_long_keys():
    from llm_strata.cli import _mask

    assert _mask(None) == "(no key)"
    assert _mask("123456789012") == "123..."
    assert _mask("12345678901234") == "1234567890...1234"


def test_setup_without_discoverable_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "config_path", lambda: tmp_path / "strata.json")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})
    result = run(
        ["strata", "setup"],
        monkeypatch,
        endpoints=[],
        base_candidates=lambda: [("http://up/v1", "test")],
        probe=lambda url, key: (True, None),
    )
    assert result.exit_code == 1
    assert "no API key was accepted" in result.output


def test_setup_url_unreachable(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "config_path", lambda: tmp_path / "strata.json")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})
    result = run(
        ["strata", "setup", "--url", "http://dead/v1"],
        monkeypatch,
        probe=lambda url, key: (False, None),
    )
    assert result.exit_code == 1
    assert "No Strata server answered at http://dead/v1." in result.output


def test_setup_rejects_url_when_no_key_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "config_path", lambda: tmp_path / "strata.json")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})
    # duplicate candidates exercise the dedup; none of them works
    monkeypatch.setattr(cli_mod, "key_candidates", lambda: [("a", "x"), ("a", "x")])
    monkeypatch.setattr(cli_mod, "probe", lambda url, key: (True, None))
    result = run(["strata", "setup", "--url", "http://x/v1"], monkeypatch)
    assert result.exit_code == 1
    assert "but no key was accepted" in result.output
    assert not (tmp_path / "strata.json").exists()


def test_status_output_health_line(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [ep_one()])
    monkeypatch.setattr(
        cli_mod,
        "health",
        lambda endpoint: {
            "model": "m1",
            "loaded": True,
            "api_key": True,
            "max_context": 131072,
        },
    )
    result = run(["strata", "status"], monkeypatch)
    assert result.exit_code == 0
    assert "health: model=m1 loaded=True api_key=True max_context=131072" in result.output
    assert "big-model" in result.output


def test_status_unreachable_health(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [ep_one()])
    monkeypatch.setattr(cli_mod, "health", lambda endpoint: None)
    result = run(["strata", "status"], monkeypatch)
    assert result.exit_code == 0
    assert "health: unreachable" in result.output


def test_status_no_server(monkeypatch):
    monkeypatch.setattr(cli_mod, "discover", lambda: [])
    result = run(
        ["strata", "status"],
        monkeypatch,
        base_candidates=lambda: [("http://down/v1", "test")],
        probe=lambda url, key: (False, None),
    )
    assert result.exit_code == 1
    assert "Could not reach any Strata server" in result.output


def test_forget_removes_pinned_config(monkeypatch):
    monkeypatch.setattr(cli_mod, "config_path", lambda: "C:/none/strata.json")
    monkeypatch.setattr(cli_mod, "clear_config", lambda: True)
    result = run(["strata", "forget"], monkeypatch)
    assert result.exit_code == 0
    assert "auto-discovery restored" in result.output
