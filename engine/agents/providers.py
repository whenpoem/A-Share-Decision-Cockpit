from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel

from engine.config import LLMProviderConfig

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM provider cannot return structured output."""


@dataclass
class LLMCallRecord:
    provider_name: str
    success: bool
    detail: str


class OpenAICompatibleProvider:
    provider_name = "llm"

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        if not self.config.api_key:
            raise LLMUnavailableError(f"{self.provider_name} API key is missing")
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover
            raise LLMUnavailableError(f"{self.provider_name} request failed: {exc}") from exc
        try:
            return schema.model_validate(json.loads(message))
        except Exception as exc:
            raise LLMUnavailableError(f"{self.provider_name} schema validation failed: {exc}") from exc


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_name = "deepseek"


class QwenProvider(OpenAICompatibleProvider):
    provider_name = "qwen"


class ProviderChain:
    def __init__(self, primary: OpenAICompatibleProvider, fallback: OpenAICompatibleProvider) -> None:
        self.providers = [primary, fallback]

    def generate(self, system_prompt: str, user_prompt: str, schema: type[T]) -> tuple[T, list[LLMCallRecord]]:
        records: list[LLMCallRecord] = []
        last_error = None
        for provider in self.providers:
            try:
                payload = provider.generate(system_prompt, user_prompt, schema)
                records.append(LLMCallRecord(provider.provider_name, True, "ok"))
                return payload, records
            except LLMUnavailableError as exc:
                last_error = exc
                records.append(LLMCallRecord(provider.provider_name, False, str(exc)))
        raise LLMUnavailableError(str(last_error or "all providers failed"))

