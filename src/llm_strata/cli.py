"""``llm strata ...`` commands."""

from __future__ import annotations

import click

from .connection import (
    Endpoint,
    base_candidates,
    clear_config,
    config_path,
    discover,
    health,
    load_config,
    normalize_base,
    probe,
    save_config,
    key_candidates,
)


def _mask(key: str | None) -> str:
    if not key:
        return "(no key)"
    if len(key) <= 12:
        return key[:3] + "..."
    return key[:10] + "..." + key[-4:]


def _no_endpoint_message() -> str:
    """Explain why no endpoint worked: down, or up but key rejected."""
    reachable = [
        (url, source) for url, source in base_candidates() if probe(url, None)[0]
    ]
    if reachable:
        where = "; ".join(f"{url} (found via {source})" for url, source in reachable)
        return (
            f"Strata answered at {where}, but no API key was accepted.\n"
            "The runtime discovers keys automatically; if yours rotated or lives "
            "elsewhere, pin it with `llm strata setup --url URL --key KEY` or "
            "store it with `llm keys set strata`."
        )
    return (
        "Could not reach any Strata server.\n"
        "Start Strata first - open the desktop app, or run `strata launch` "
        "- then retry. If your server runs somewhere unusual, pin it with "
        "`llm strata setup --url URL`."
    )


def _models_table(endpoint: Endpoint) -> list[tuple[str, str, str]]:
    rows = []
    for entry in endpoint.models:
        if entry.alias_of:
            continue
        aliases = [e.id for e in endpoint.models if e.alias_of == entry.id]
        rows.append(
            (
                f"strata/{entry.id}",
                entry.label or "",
                ", ".join(f"strata/{alias}" for alias in aliases),
            )
        )
    return rows


def _print_endpoint(endpoint: Endpoint) -> None:
    click.echo(f"{endpoint.base_url}  [{endpoint.source}]")
    rows = _models_table(endpoint)
    if not rows:
        click.echo("  (no models loaded)")
    for model_id, label, aliases in rows:
        suffix = ""
        if label:
            suffix += f"  - {label}"
        if aliases:
            suffix += f"  (aliases: {aliases})"
        click.echo(f"  {model_id}{suffix}")


def add_commands(cli) -> None:
    @cli.group(name="strata")
    def strata_():
        """Interact with the Strata local model runtime."""

    @strata_.command()
    @click.option("--json", "as_json", is_flag=True, help="Output JSON")
    def models(as_json: bool):
        """List the models discovered from running Strata servers."""
        endpoints = discover()
        if not endpoints:
            raise click.ClickException(_no_endpoint_message())
        if as_json:
            payload = [
                {
                    "base_url": endpoint.base_url,
                    "source": endpoint.source,
                    "api_key": bool(endpoint.api_key),
                    "models": [
                        {
                            "id": entry.id,
                            "model_id": f"strata/{entry.id}",
                            "label": entry.label,
                            "alias_of": entry.alias_of,
                            "aliases": list(entry.aliases),
                        }
                        for entry in endpoint.models
                    ],
                }
                for endpoint in endpoints
            ]
            click.echo(click.unstyle(_json_dumps(payload)))
            return
        for endpoint in endpoints:
            _print_endpoint(endpoint)

    @strata_.command()
    @click.option("--json", "as_json", is_flag=True, help="Output JSON")
    def status(as_json: bool):
        """Show the resolved Strata endpoints and their health."""
        endpoints = discover()
        config = load_config()
        if as_json:
            payload = {
                "config": {
                    "path": str(config_path()),
                    "base_url": config.get("base_url"),
                    "api_key": _mask(config.get("api_key")),
                },
                "endpoints": [
                    {
                        "base_url": endpoint.base_url,
                        "source": endpoint.source,
                        "api_key": _mask(endpoint.api_key),
                        "models": [entry.id for entry in endpoint.models],
                        "health": health(endpoint),
                    }
                    for endpoint in endpoints
                ],
            }
            click.echo(_json_dumps(payload))
            return
        if not endpoints:
            raise click.ClickException(_no_endpoint_message())
        for endpoint in endpoints:
            _print_endpoint(endpoint)
            data = health(endpoint)
            if data is None:
                click.echo("  health: unreachable")
                continue
            click.echo(
                "  health: "
                f"model={data.get('model')} loaded={data.get('loaded')} "
                f"api_key={data.get('api_key')} "
                f"max_context={data.get('max_context')}"
            )

    @strata_.command()
    @click.option("--url", help="Base URL to pin, e.g. http://127.0.0.1:8080/v1")
    @click.option("--key", help="API key to pin (leave out to discover one)")
    def setup(url: str | None, key: str | None):
        """Pin a Strata base URL and API key in llm's config.

        Without options, auto-discovers the first reachable server and
        pins what it finds. Later ``llm`` runs prefer the pinned values;
        remove them again with ``llm strata forget``.
        """
        config = load_config()
        base_url = normalize_base(url) if url else config.get("base_url")
        api_key = key or config.get("api_key")

        if not base_url:
            endpoints = discover()
            if not endpoints:
                raise click.ClickException(_no_endpoint_message())
            endpoint = endpoints[0]
            base_url, api_key = endpoint.base_url, endpoint.api_key
            click.echo(f"Discovered {base_url} [{endpoint.source}]")

        # Verify the pairing before saving; if an explicit/pinned key is
        # rejected, fall back to the other discovery candidates.
        reachable, _ = probe(base_url, None)
        if not reachable:
            raise click.ClickException(f"No Strata server answered at {base_url}.")
        keys_to_try: list[tuple[str | None, str]] = []
        if api_key:
            keys_to_try.append((api_key, "given"))
        keys_to_try += [(k or None, s) for k, s in key_candidates()]
        seen: set[tuple[str | None, str]] = set()
        for candidate, source in keys_to_try:
            if (candidate, source) in seen:
                continue
            seen.add((candidate, source))
            _, models = probe(base_url, candidate)
            if models is not None:
                save_config({"base_url": base_url, "api_key": candidate})
                click.echo(f"Pinned {base_url} with key {_mask(candidate)} [{source}]")
                click.echo(f"Config written to {config_path()}")
                fake = Endpoint(base_url, candidate, source, tuple(models))
                for model_id, label, aliases in _models_table(fake):
                    suffix = f"  - {label}" if label else ""
                    if aliases:
                        suffix += f"  (aliases: {aliases})"
                    click.echo(f"  {model_id}{suffix}")
                return
        raise click.ClickException(
            f"Strata answered at {base_url}, but no key was accepted. "
            "Provide one with `llm strata setup --url URL --key KEY` "
            "(Strata prints its key with `strata launch --show-key`)."
        )

    @strata_.command()
    def forget():
        """Remove the pinned Strata endpoint (revert to auto-discovery)."""
        if clear_config():
            click.echo(f"Removed {config_path()} - auto-discovery restored.")
        else:
            click.echo(f"Nothing pinned ({config_path()} does not exist).")


def _json_dumps(payload) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
