import llm_strata
from llm_strata.connection import Endpoint, ModelInfo


class Collector:
    def __init__(self):
        self.registered = []

    def __call__(self, model, aliases=None):
        self.registered.append((model, list(aliases or [])))


def endpoint(*entries, base_url="http://x/v1", api_key="k", source="test"):
    return Endpoint(base_url, api_key, source, tuple(entries))


def test_registers_primary_models_with_alias_entries(monkeypatch):
    ep = endpoint(
        ModelInfo(id="qwen-max", label="Qwen Max"),
        ModelInfo(id="qwen", alias_of="qwen-max"),
        ModelInfo(id="qwen-mini"),
    )
    monkeypatch.setattr(llm_strata, "discover", lambda: [ep])
    collector = Collector()
    llm_strata.register_models(register=collector)

    ids = [(m.model_id, m.api_base, m.key) for m, _ in collector.registered]
    assert ids == [
        ("strata/qwen-max", "http://x/v1", "k"),
        ("strata/qwen-mini", "http://x/v1", "k"),
    ]
    # The alias_of entry becomes an llm alias, not a second model
    assert collector.registered[0][1] == ["strata/qwen"]
    assert collector.registered[1][1] == []
    assert collector.registered[0][0].model_name == "qwen-max"


def test_register_commands_adds_strata_commands():
    import click

    @click.group()
    def llm_cli():
        pass

    llm_strata.register_commands(llm_cli)
    strata_group = llm_cli.commands["strata"]
    assert isinstance(strata_group, click.Group)
    for name in ("models", "status", "setup", "forget"):
        assert name in strata_group.commands


def test_no_discovery_registers_nothing(monkeypatch):
    monkeypatch.setattr(llm_strata, "discover", lambda: [])
    collector = Collector()
    llm_strata.register_models(register=collector)
    assert collector.registered == []


def test_same_model_on_two_endpoints_registered_once(monkeypatch):
    ep1 = endpoint(ModelInfo(id="m1"), base_url="http://one/v1", api_key="a")
    ep2 = endpoint(ModelInfo(id="m1"), base_url="http://two/v1", api_key="b")
    monkeypatch.setattr(llm_strata, "discover", lambda: [ep1, ep2])
    collector = Collector()
    llm_strata.register_models(register=collector)
    assert [m.api_base for m, _ in collector.registered] == ["http://one/v1"]
