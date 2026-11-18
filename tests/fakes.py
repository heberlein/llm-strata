"""Shared fake HTTP/OpenAI plumbing for the tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class FakeClient:
    """Stand-in for httpx.Client: returns canned responses, records requests."""

    instances: list["FakeClient"] = []

    def __init__(self, **kwargs):
        self.responses = getattr(FakeClient, "responses", [])
        self.requests: list[httpx.Request] = []
        FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        self.requests.append(httpx.Request("GET", url, headers=headers or {}))
        return self.responses.pop(0)


def chunk(
    content=None,
    reasoning=None,
    tool_calls=None,
    finish_reason=None,
    usage=None,
    delta_role=None,
):
    """One OpenAI-style streaming chunk, duck-typed for llm's combine_chunks."""
    delta = SimpleNamespace(
        role=delta_role,
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        index=0, delta=delta, finish_reason=finish_reason, logprobs=None
    )
    if usage is not None:
        prompt_tokens, completion_tokens = usage
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            model_dump=lambda: {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
    return SimpleNamespace(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=1,
        model="test",
        choices=[choice],
        usage=usage,
    )


def tool_delta(index=0, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    """client.chat.completions.create() stand-in."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_client(result):
    completions = FakeCompletions(result)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions)), completions


def models_payload(models: list[dict]) -> dict:
    return {"object": "list", "data": models}


def dump(obj) -> str:
    return json.dumps(obj)
