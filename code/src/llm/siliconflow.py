"""SiliconFlow OpenAI-compatible client with token accounting."""

from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

from src.config import SILICONFLOW_BASE_URL, SILICONFLOW_MODEL, get_api_key, load_env


class SiliconFlowClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        load_env()
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY") or get_api_key()
        self.model = model or SILICONFLOW_MODEL
        self.base_url = base_url or SILICONFLOW_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.total_tokens = 0
        self.llm_calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        self.llm_calls += 1
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if resp.usage is not None:
            usage = {
                "prompt_tokens": int(resp.usage.prompt_tokens or 0),
                "completion_tokens": int(resp.usage.completion_tokens or 0),
                "total_tokens": int(resp.usage.total_tokens or 0),
            }
            self.total_tokens += usage["total_tokens"]

        choice = resp.choices[0]
        message = choice.message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )

        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls or None,
            "usage": usage,
            "finish_reason": choice.finish_reason,
        }
