# llm-strata

An [llm](https://github.com/simonw/llm) plugin that exposes the models served
locally by the **Strata** runtime as regular llm models, named `strata/<model-id>`.

## Install

```bash
llm install llm-strata
```

or from a clone of this repo:

```bash
llm install -e .
```

## Usage

The plugin discovers the running Strata server and its API key automatically,
so once Strata is serving a model you can just run:

```bash
llm -m strata/qwen3.8-flash-next "hello"
llm -m strata/qwen3.8-flash-next -o temperature 0.3 -o max_tokens 200 "hello"
```

Streaming, system prompts, token usage and tool calling
(`llm -m strata/... --tool ...`) all work as usual. Reasoning output from
thinking models is surfaced as a separate `reasoning` part.

Only three model options are exposed: `temperature`, `top_p`, `max_tokens`.
Everything else (context size, quantization, chat template) is decided by the
Strata runtime.

## Discovery

Without any configuration, the plugin looks for a server in this order:

1. A pin saved with `llm strata setup`
2. `STRATA_BASE_URL` environment variable
3. `*.ready.json` files in the Strata runtime directories
   (e.g. `%LOCALAPPDATA%\strata\runtime`)
4. `http://127.0.0.1:8080/v1`

API keys are looked up in this order:

1. A pin saved with `llm strata setup`
2. `STRATA_API_KEY` environment variable
3. An `strata` key stored via `llm keys set strata`
4. `serving.apiKey` in the Strata `state.json`
5. `api_key` in the Strata `native-instances` files

Every URL/key pair is verified against `GET /v1/models` before use, so a stale
key never produces a confusing 401 later.

## Commands

```bash
llm strata models             # list models the plugin found
llm strata status             # endpoint + health details (--json for machine output)
llm strata setup              # auto-detect and pin the current endpoint and key
llm strata setup --url URL --key KEY   # pin explicitly
llm strata forget             # remove the pin, fall back to auto-discovery
```

`llm install llm-strata` followed by `llm plugins` should show the `strata`
plugin with a `setup` command; if `llm -m strata/...` says *Unknown model*,
start Strata (or run `llm strata models` to see why nothing was found).

## Development

```bash
uv sync
uv run pytest
```
