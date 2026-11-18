"""llm plugin for models served by the Strata local model runtime.

Registers every model advertised by a reachable Strata server as
``strata/<model-id>``, using the verbatim IDs from ``GET /v1/models``.
When no Strata server is reachable at llm startup, no models are
registered - see ``llm strata setup`` / ``llm strata status``.
"""

from __future__ import annotations

from llm import hookimpl

from .connection import discover
from .model import StrataChat, StrataOptions

__version__ = "0.1.0"

__all__ = ["StrataChat", "StrataOptions"]


def _model_id_for(entry) -> str:
    return f"strata/{entry.id}"


@hookimpl
def register_models(register):
    registered_ids: set[str] = set()
    for endpoint in discover():
        entries = list(endpoint.models)
        by_id = {entry.id: entry for entry in entries}
        for entry in entries:
            # Entries with alias_of duplicate another entry under a shorter
            # name - expose those as llm aliases of the primary instead of
            # registering a second model.
            if entry.alias_of and entry.alias_of in by_id:
                continue
            model_id = _model_id_for(entry)
            if model_id in registered_ids:
                continue
            registered_ids.add(model_id)
            aliases = [
                _model_id_for(other) for other in entries if other.alias_of == entry.id
            ]
            register(
                StrataChat(
                    model_id=model_id,
                    key=endpoint.api_key,
                    model_name=entry.id,
                    api_base=endpoint.base_url,
                ),
                aliases=aliases,
            )


@hookimpl
def register_commands(cli):
    from .cli import add_commands

    add_commands(cli)
