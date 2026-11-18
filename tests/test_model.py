from types import SimpleNamespace

import httpx2
import llm
import openai
import pytest
from fakes import FakeCompletions, chunk, tool_delta
from pydantic import ValidationError

from llm_strata.model import StrataChat, StrataOptions


def make_model(result):
    model = StrataChat(
        "strata/test",
        key="k",
        model_name="test-model",
        api_base="http://127.0.0.1:1/v1",
    )
    completions = FakeCompletions(result)
    model.get_client = lambda key=None, *, async_=False: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return model, completions


def test_str_shows_model_id_and_base():
    model, _ = make_model([])
    assert str(model) == "Strata Chat: strata/test (http://127.0.0.1:1/v1)"


def events_by_type(events):
    out = {}
    for event in events:
        if isinstance(event, str):
            out.setdefault("text", []).append(event)
        else:
            out.setdefault(event.type, []).append(event.chunk)
    return out


def test_streaming_text_and_reasoning():
    chunks = [
        chunk(content="", delta_role="assistant"),
        chunk(reasoning="Hmm, "),
        chunk(reasoning="42."),
        chunk(content="He"),
        chunk(content="llo"),
        chunk(content=None, finish_reason="stop", usage=(10, 4)),
    ]
    model, completions = make_model(iter(chunks))
    prompt = llm.Prompt("hi", model)
    response = llm.Response(prompt, model, stream=True)

    events = events_by_type(response.stream_events())
    assert "".join(events["text"]) == "Hello"
    assert "".join(events["reasoning"]) == "Hmm, 42."
    assert response.text() == "Hello"

    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["stream"] is True
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["stream_options"] == {"include_usage": True}

    assert response.input_tokens == 10
    assert response.output_tokens == 4
    payload = response.json()
    assert payload is not None and payload["id"] == "chatcmpl-test"


def test_streaming_tool_calls_accumulate():
    chunks = [
        chunk(tool_calls=[tool_delta(index=0, id="call-1", name="get_time")]),
        chunk(tool_calls=[tool_delta(index=0, arguments='{"city": ')]),
        chunk(tool_calls=[tool_delta(index=0, arguments='"Berlin"}')]),
        chunk(content=None, finish_reason="tool_calls", usage=(5, 7)),
    ]
    model, _ = make_model(iter(chunks))
    prompt = llm.Prompt("hi", model)
    response = llm.Response(prompt, model, stream=True)

    events = events_by_type(response.stream_events())
    assert events["tool_call_name"] == ["get_time"]
    assert "".join(events["tool_call_args"]) == '{"city": "Berlin"}'

    (tool_call,) = response.tool_calls()
    assert tool_call.name == "get_time"
    assert tool_call.arguments == {"city": "Berlin"}
    assert tool_call.tool_call_id == "call-1"


def test_stream_tool_call_fragment_without_function_is_skipped():
    broken = SimpleNamespace(index=1, id="call-x", type="function", function=None)
    chunks = [
        chunk(tool_calls=[tool_delta(index=0, id="call-1", name="get_time")]),
        chunk(tool_calls=[broken]),
        chunk(content=None, finish_reason="tool_calls", usage=(5, 7)),
    ]
    model, _ = make_model(iter(chunks))
    prompt = llm.Prompt("hi", model)
    response = llm.Response(prompt, model, stream=True)
    events = events_by_type(response.stream_events())
    assert events["tool_call_name"] == ["get_time"]
    assert [t.tool_call_id for t in response.tool_calls()] == ["call-1"]


def test_non_streaming_text_reasoning_and_tools():
    message = SimpleNamespace(
        role="assistant",
        content="The time is now.",
        reasoning_content="I called the tool.",
        tool_calls=[
            SimpleNamespace(
                id="call-9",
                type="function",
                function=SimpleNamespace(name="get_time", arguments='{"city": "Rome"}'),
            ),
            # A custom (free-form) tool call must be skipped, not crash.
            SimpleNamespace(id="call-custom", type="custom", name="freeform"),
        ],
    )
    completion = SimpleNamespace(
        id="cmpl-1",
        model="test-model",
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            model_dump=lambda: {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        ),
        model_dump=lambda: {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "The time is now.",
                        "reasoning_content": "I called the tool.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
    )
    model, completions = make_model(completion)
    prompt = llm.Prompt("hi", model)
    response = llm.Response(prompt, model, stream=False)

    events = events_by_type(response.stream_events())
    assert "".join(events["text"]) == "The time is now."
    assert events["reasoning"] == ["I called the tool."]
    assert response.text() == "The time is now."

    (tool_call,) = response.tool_calls()
    assert tool_call.name == "get_time"
    assert tool_call.arguments == {"city": "Rome"}
    assert tool_call.tool_call_id == "call-9"

    call = completions.calls[0]
    assert call["stream"] is False
    assert "stream_options" not in call

    assert response.input_tokens == 12
    assert response.output_tokens == 8
    payload = response.json()
    assert payload is not None and payload["id"] == "cmpl-1"


def test_connection_error_raises_model_error():
    error = openai.APIConnectionError(request=httpx2.Request("POST", "http://x/v1"))
    model, _ = make_model(error)
    prompt = llm.Prompt("hi", model)
    response = llm.Response(prompt, model, stream=True)
    with pytest.raises(llm.ModelError) as exc:
        list(response.stream_events())
    assert "Cannot reach the Strata server" in str(exc.value)
    assert "http://127.0.0.1:1/v1" in str(exc.value)


def test_system_prompt_and_options_flow_to_wire():
    model, completions = make_model(iter([chunk(content="ok", usage=(1, 1))]))
    prompt = llm.Prompt(
        "hi",
        model,
        system="Be brief.",
        options=StrataOptions(temperature=0.25, max_tokens=50),
    )
    response = llm.Response(prompt, model, stream=True)
    list(response.stream_events())
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "Be brief."}
    assert call["temperature"] == 0.25
    assert call["max_tokens"] == 50
    assert "top_p" not in call  # unset options must not be sent


def test_options_class_is_limited():
    assert set(StrataOptions.model_fields) == {"temperature", "top_p", "max_tokens"}
    model, _ = make_model(iter([chunk(content="ok", usage=(1, 1))]))
    assert model.Options is StrataOptions
    with pytest.raises(ValidationError):
        model.Options(nonexistent=1)
