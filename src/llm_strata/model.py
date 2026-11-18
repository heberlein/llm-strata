"""The model class: an OpenAI-compatible chat model backed by Strata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, overload

import llm
import openai
from llm.default_plugins.openai_models import Chat, combine_chunks, redact_data
from llm.parts import StreamEvent
from llm.utils import remove_dict_none_values
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import Field


class StrataOptions(llm.Options):
    temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature. Higher values make output more random. "
            "Strata default: leave unset to use the server's setting."
        ),
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling probability mass, 0.0 to 1.0.",
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (counting thinking tokens). "
            "Strata default: fill the remaining context."
        ),
    )


@dataclass
class _ToolCallAcc:
    """A tool-call accumulated from the stream fragments at one delta index."""

    id: str | None
    name: str
    arguments: str


class StrataChat(Chat):
    """Chat completion against a Strata server's OpenAI-compatible endpoint.

    Inherits message/kwargs building, tool handling and usage parsing from
    llm's built-in OpenAI plugin, and additionally forwards Strata's
    ``reasoning_content`` stream deltas as ``type="reasoning"`` events.
    """

    needs_key = "strata"
    key_env_var = "STRATA_API_KEY"

    def __init__(
        self,
        model_id: str,
        key: str | None = None,
        model_name: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            key=key,
            model_name=model_name,
            api_base=api_base,
            supports_tools=True,
        )
        # Replace the broad OpenAI options class with the three knobs
        # Strata's endpoint actually accepts.
        self.Options = StrataOptions

    def __str__(self) -> str:
        return f"Strata Chat: {self.model_id} ({self.api_base})"

    @overload
    def _create(
        self,
        client: openai.OpenAI,
        stream: Literal[True],
        messages: list,
        kwargs: dict,
    ) -> openai.Stream[ChatCompletionChunk]: ...

    @overload
    def _create(
        self,
        client: openai.OpenAI,
        stream: Literal[False],
        messages: list,
        kwargs: dict,
    ) -> ChatCompletion: ...

    def _create(
        self,
        client: openai.OpenAI,
        stream: bool,
        messages: list,
        kwargs: dict,
    ) -> ChatCompletion | openai.Stream[ChatCompletionChunk]:
        try:
            return client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        except openai.APIConnectionError as error:
            raise llm.ModelError(
                f"Cannot reach the Strata server at {self.api_base}. "
                "Start Strata (open the desktop app or run `strata launch`) "
                "and try again, or check `llm strata status`."
            ) from error

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
        key: str | None = None,
    ) -> Iterator[StreamEvent | str]:
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None

        if stream:
            completion = self._create(client, True, messages, kwargs)
            chunks: list[ChatCompletionChunk] = []
            tool_calls: dict[int, _ToolCallAcc] = {}
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta:
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        yield StreamEvent(type="reasoning", chunk=reasoning)
                    for tool_call in delta.tool_calls or []:
                        function = tool_call.function
                        if function is None:
                            continue
                        args = function.arguments or ""
                        acc = tool_calls.get(tool_call.index)
                        if acc is None:
                            acc = _ToolCallAcc(
                                id=tool_call.id,
                                name=function.name or "",
                                arguments=args,
                            )
                            tool_calls[tool_call.index] = acc
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=acc.name,
                                tool_call_id=acc.id,
                            )
                        else:
                            acc.arguments += args
                        if args:
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=args,
                                tool_call_id=acc.id,
                            )
                    if delta.content:
                        # Empty-string deltas are noise (the role chunk).
                        yield StreamEvent(type="text", chunk=delta.content)
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
            for acc in tool_calls.values():
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=acc.id,
                        name=acc.name,
                        arguments=json.loads(acc.arguments or "{}"),
                    )
                )
        else:
            completion = self._create(client, False, messages, kwargs)
            usage = completion.usage.model_dump() if completion.usage else None
            response.response_json = remove_dict_none_values(completion.model_dump())
            message = completion.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                yield StreamEvent(type="reasoning", chunk=reasoning)
            for tool_call in message.tool_calls or []:
                if tool_call.type != "function":
                    continue
                function = tool_call.function
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=function.name,
                        arguments=json.loads(function.arguments or "{}"),
                    )
                )
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=function.name or "",
                    tool_call_id=tool_call.id,
                )
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=function.arguments or "",
                    tool_call_id=tool_call.id,
                )
            if message.content is not None:
                yield StreamEvent(type="text", chunk=message.content)

        self.set_usage(response, usage)
        response._prompt_json = redact_data({"messages": messages})
