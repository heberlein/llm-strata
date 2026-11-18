"""The model class: an OpenAI-compatible chat model backed by Strata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Optional

import openai
from pydantic import Field

import llm
from llm.default_plugins.openai_models import Chat, combine_chunks, redact_data
from llm.parts import StreamEvent
from llm.utils import remove_dict_none_values


class StrataOptions(llm.Options):
    temperature: Optional[float] = Field(
        default=None,
        description=(
            "Sampling temperature. Higher values make output more random. "
            "Strata default: leave unset to use the server's setting."
        ),
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability mass, 0.0 to 1.0.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (counting thinking tokens). "
            "Strata default: fill the remaining context."
        ),
    )


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

    def _create(self, client, stream: bool, messages: list, kwargs: dict):
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
    ) -> Iterator[llm.StreamEvent | str]:
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None

        if stream:
            completion = self._create(client, True, messages, kwargs)
            chunks = []
            tool_calls: dict[int, object] = {}
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
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        index = tool_call.index
                        if index not in tool_calls:
                            tool_calls[index] = tool_call
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_call.function.name or "",
                                tool_call_id=tool_call.id,
                            )
                        else:
                            tool_calls[
                                index
                            ].function.arguments += tool_call.function.arguments
                        if tool_call.function.arguments:
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=tool_call.function.arguments,
                                tool_call_id=tool_calls[index].id,
                            )
                    if delta.content:
                        # Empty-string deltas are noise (the role chunk).
                        yield StreamEvent(type="text", chunk=delta.content)
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
            for tool_call in tool_calls.values():
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
        else:
            completion = self._create(client, False, messages, kwargs)
            usage = completion.usage.model_dump()
            response.response_json = remove_dict_none_values(completion.model_dump())
            message = completion.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                yield StreamEvent(type="reasoning", chunk=reasoning)
            for tool_call in message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name or "",
                    tool_call_id=tool_call.id,
                )
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=tool_call.function.arguments or "",
                    tool_call_id=tool_call.id,
                )
            if message.content is not None:
                yield StreamEvent(type="text", chunk=message.content)

        self.set_usage(response, usage)
        response._prompt_json = redact_data({"messages": messages})
