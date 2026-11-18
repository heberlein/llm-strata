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
