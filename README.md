# llm-strata

An [llm](https://github.com/simonw/llm) plugin that exposes the models served
locally by [Strata](https://github.com/Niko1221/Strata) — a one-click local AI
runtime for Windows/Linux with an OpenAI-compatible API on localhost — as
regular llm models, named `strata/<model-id>`.

If you don't know either: **llm** is a CLI for running prompts against any LLM
with a local logging database; **Strata** runs open models (e.g.
Qwen3.8-Flash-Next) on your own hardware and serves them at
`http://127.0.0.1:8080/v1`. This plugin lets `llm` talk to Strata's models
like any hosted model.

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
so once Strata is serving a model you can just run it. The exact model IDs
depend on which models your Strata serves, so list them first:

```bash
llm strata models
# strata/qwen3.8-flash-next-iq3_s  (aliases: strata/iq3_s)

llm -m strata/qwen3.8-flash-next-iq3_s "hello"
llm -m strata/qwen3.8-flash-next-iq3_s -o temperature 0.3 -o max_tokens 200 "hello"
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

## Alternative: no plugin needed

llm (0.36+) ships a built-in command for any OpenAI-compatible endpoint, so
you can talk to Strata without this plugin at all. You need the server URL and
its API key. Both are shown in the Strata UI under **Serving**, and from the
CLI:

```bash
strata launch --status              # address of the running strata-launch server
strata launch --show-key            # its API key
strata connect opencode --show-key  # URL + key the same way an app would be configured
```

In the launch monitor, press `7` (the *Connect* view) and `k` to reveal the
key. The default address is `http://127.0.0.1:8080`; a local server without an
API key (the default) needs no `--key` at all, while sharing beyond this
computer makes Strata generate an `sk-strata-…` key that you must pass.

These CLI commands report servers started with `strata launch`; when the
desktop app is the one serving, read the URL and key from its UI instead.
Use them in the example below.

```bash
# list the models Strata is serving (URL/key as shown in the Strata UI;
# through the gateway only the short model aliases are listed)
llm openai endpoint http://127.0.0.1:47831/v1 --key sk-strata-… --models

# one-off prompt, interactive chat, tools...
llm openai endpoint http://127.0.0.1:47831/v1 --key sk-strata-… \
  -m iq3_s --chat
```

`--key` also accepts a stored key alias (e.g. `llm keys set strata --value KEY`,
then `--key strata`).

The difference is what you give up: `llm openai endpoint` does **not log to the
llm database** and you must pass the URL, key and model ID every time. This
plugin registers the models permanently, so `llm -m strata/...` just works,
prompts are logged and replayable (`llm logs`), conversations persist, and the
URL/key are discovered from the Strata runtime automatically — including its
rotating keys, which is the part you'd otherwise do by hand.

## Development

```bash
uv sync
uv run pytest
```
