"""Endpoint discovery for the Strata local model runtime.

Strata serves an OpenAI-compatible API, but its address is not fixed:

- The desktop app starts an inference engine on a port chosen per model
  install. The bound port is recorded in ``*.ready.json`` files under the
  Strata runtime directory (Windows: ``%LOCALAPPDATA%\\strata\\runtime``).
- The engine's API key is stored in the matching instance config under
  ``~/.strata/data/native-instances/*.json``.
- A server launched via ``strata launch`` defaults to port 8080.

Resolution order for the base URL:

1. ``base_url`` pinned by ``llm strata setup`` (stored in llm's config
   directory as ``strata.json``)
2. ``STRATA_BASE_URL`` environment variable
3. ``api_url`` from every ``*.ready.json`` in the Strata runtime directory
   (``STRATA_RUNTIME_DIR`` overrides its location)
4. ``http://127.0.0.1:8080/v1`` (the ``strata launch`` default)

Resolution order for the API key:

1. ``api_key`` pinned by ``llm strata setup``
2. ``STRATA_API_KEY`` environment variable
3. Key stored for alias ``strata`` via ``llm keys set strata``
4. ``serving.apiKey`` from ``~/.strata/data/state.json`` (desktop app)
5. ``api_key`` from ``~/.strata/data/native-instances/*.json`` (engine)

Every (base URL, key) pairing is verified with ``GET {base_url}/models``
before the endpoint is used, so a wrong key never reaches a request.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import llm

PROBE_TIMEOUT = 2.0
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


@dataclass(frozen=True)
class ModelInfo:
    """One entry from Strata's ``GET /v1/models`` response."""

    id: str
    label: str | None = None
    alias_of: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Endpoint:
    """A Strata server whose ``/v1/models`` probe succeeded."""

    base_url: str
    api_key: str | None
    source: str
    models: tuple[ModelInfo, ...]


def config_path() -> Path:
    return llm.user_dir() / "strata.json"


def load_config() -> dict:
    path = config_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(data: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def clear_config() -> bool:
    path = config_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def normalize_base(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _home() -> Path:
    return Path.home()


def runtime_dirs() -> list[Path]:
    dirs: list[Path] = []
    if env := os.environ.get("STRATA_RUNTIME_DIR"):
        dirs.append(Path(env))
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "strata" / "runtime")
    else:
        data = os.environ.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
        dirs.append(Path(data) / "strata" / "runtime")
    dirs.append(_home() / ".strata" / "runtime")
    return dirs


def state_file() -> Path:
    return _home() / ".strata" / "data" / "state.json"


def native_instances_dir() -> Path:
    return _home() / ".strata" / "data" / "native-instances"


def base_candidates() -> list[tuple[str, str]]:
    """Candidate base URLs as (url, source) pairs, in priority order."""
    candidates: list[tuple[str, str]] = []
    if base := load_config().get("base_url"):
        candidates.append((normalize_base(base), "setup"))
    if env := os.environ.get("STRATA_BASE_URL"):
        candidates.append((normalize_base(env), "STRATA_BASE_URL"))
    for directory in runtime_dirs():
        if not directory.is_dir():
            continue
        ready_files = sorted(
            directory.glob("*.ready.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for ready in ready_files:
            try:
                data = json.loads(ready.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("api_state") not in (None, "ready"):
                continue
            api_url = data.get("api_url")
            if api_url:
                candidates.append((normalize_base(api_url), f"runtime:{ready.name}"))
    candidates.append((normalize_base(DEFAULT_BASE_URL), "default"))
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for url, source in candidates:
        if url not in seen:
            seen.add(url)
            unique.append((url, source))
    return unique


def key_candidates() -> list[tuple[str, str]]:
    """Candidate API keys as (key, source) pairs, in priority order."""
    keys: list[tuple[str, str]] = []
    if key := load_config().get("api_key"):
        keys.append((key, "setup"))
    if env := os.environ.get("STRATA_API_KEY"):
        keys.append((env, "STRATA_API_KEY"))
    if stored := llm.get_key(key_alias="strata"):
        keys.append((stored, "llm keys"))
    try:
        state = json.loads(state_file().read_text(encoding="utf-8"))
        serving_key = (state.get("serving") or {}).get("apiKey")
        if serving_key:
            keys.append((serving_key, "state.json"))
    except (OSError, ValueError):
        pass
    native_dir = native_instances_dir()
    if native_dir.is_dir():
        instance_files = sorted(
            native_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for instance in instance_files:
            try:
                data = json.loads(instance.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if api_key := data.get("api_key"):
                keys.append((api_key, f"native:{instance.stem[:8]}"))
    keys.append(("", "no key"))
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for key, source in keys:
        if key not in seen:
            seen.add(key)
            unique.append((key, source))
    return unique


def probe(base_url: str, api_key: str | None) -> tuple[bool, tuple[ModelInfo, ...] | None]:
    """Try one (base URL, key) pair.

    Returns ``(reachable, models)``:

    - ``(True, models)`` on HTTP 200 - models parsed from the response
    - ``(True, None)`` on any other status (e.g. 401 for a wrong key)
    - ``(False, None)`` when the server could not be contacted at all
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(trust_env=False, timeout=PROBE_TIMEOUT) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError:
        return False, None
    if response.status_code != 200:
        return True, None
    try:
        data = response.json()
    except ValueError:
        return True, None
    models = []
    for entry in data.get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        meta = entry.get("meta") or {}
        models.append(
            ModelInfo(
                id=model_id,
                label=entry.get("label") or meta.get("label"),
                alias_of=entry.get("alias_of"),
                aliases=tuple(entry.get("aliases") or meta.get("aliases") or ()),
            )
        )
    return True, tuple(models)


def discover() -> list[Endpoint]:
    """Return every reachable Strata endpoint with a working key."""
    endpoints: list[Endpoint] = []
    tried: set[str] = set()
    candidates = key_candidates()
    for base_url, base_source in base_candidates():
        if base_url in tried:
            continue
        tried.add(base_url)
        for api_key, key_source in candidates:
            reachable, models = probe(base_url, api_key)
            if not reachable:
                break  # server not listening, no point trying more keys here
            if models is not None:
                endpoints.append(
                    Endpoint(
                        base_url=base_url,
                        api_key=api_key or None,
                        source=f"{base_source} + {key_source}",
                        models=models,
                    )
                )
                break
    return endpoints


def health(endpoint: Endpoint) -> dict | None:
    """Proxy ``GET /health`` for an endpoint. Returns None if unreachable."""
    root = endpoint.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with httpx.Client(trust_env=False, timeout=PROBE_TIMEOUT) as client:
            response = client.get(root + "/health")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None
