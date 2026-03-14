from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel

from engine.config import LLMProviderConfig

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM provider cannot return structured output."""

    def __init__(self, message: str, *, call_records: list["LLMCallRecord"] | None = None) -> None:
        super().__init__(message)
        self.call_records = call_records or []


@dataclass
class LLMCallRecord:
    provider_name: str
    success: bool
    detail: str


def _extract_json_object(content: str) -> str:
    text = content.strip()
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated JSON object")


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
            json_payload = _extract_json_object(message)
        except Exception as exc:
            preview = message[:240].replace("\n", " ")
            raise LLMUnavailableError(
                f"{self.provider_name} json extraction failed: {exc}; preview={preview}"
            ) from exc
        try:
            return schema.model_validate(json.loads(json_payload))
        except Exception as exc:
            raise LLMUnavailableError(f"{self.provider_name} schema validation failed: {exc}") from exc


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_name = "deepseek"


class ProviderChain:
    def __init__(self, *providers: OpenAICompatibleProvider) -> None:
        self.providers = [provider for provider in providers if provider is not None]

    def generate(self, system_prompt: str, user_prompt: str, schema: type[T]) -> tuple[T, list[LLMCallRecord]]:
        records: list[LLMCallRecord] = []
        for provider in self.providers:
            try:
                payload = provider.generate(system_prompt, user_prompt, schema)
                records.append(LLMCallRecord(provider.provider_name, True, "ok"))
                return payload, records
            except LLMUnavailableError as exc:
                records.append(LLMCallRecord(provider.provider_name, False, str(exc)))
        if not records:
            raise LLMUnavailableError("no llm providers configured")
        raise LLMUnavailableError(
            " | ".join(f"{record.provider_name}: {record.detail}" for record in records),
            call_records=records,
        )
