# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.0 - unreleased

### Added

- llm plugin that registers every model served by a running
  [Strata](https://github.com/Niko1221/Strata) runtime as `strata/<id>`,
  with Strata-side aliases registered as llm aliases.
- Auto-discovery of the Strata endpoint and API key: pinned config
  (`llm strata setup`), `STRATA_BASE_URL` / `STRATA_API_KEY` environment
  variables, Strata runtime `*.ready.json` files, and the desktop app's
  `state.json` / `native-instances` files. Every URL/key pair is verified
  with `GET /v1/models` before use.
- CLI commands `llm strata models|status|setup|forget` (with `--json`
  output for models and status).
- Streaming support including reasoning tokens, and tool-calling support
  via a restricted options class (`temperature`, `max_tokens`,
  `top_p` only).
