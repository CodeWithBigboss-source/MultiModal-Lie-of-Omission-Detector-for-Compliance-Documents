"""
Thin wrapper around the Gemini API.

Centralizes: retry/backoff on free-tier 429s, automatic fallback from
gemini-2.5-flash to gemini-2.5-flash-lite, and structured JSON parsing +
schema validation for every call.
"""

import json
import os
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

T = TypeVar("T", bound=BaseModel)

PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"


class RateLimitError(Exception):
    pass


class ModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
        self.client = genai.Client(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(RateLimitError),
    )
    def _call(self, model: str, contents: list, response_schema: Type[T]) -> str:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1,
                ),
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                raise RateLimitError(str(e)) from e
            raise

    def structured_call(
        self,
        prompt_parts: list,
        response_schema: Type[T],
        prefer_lite: bool = False,
    ) -> T:
        model = FALLBACK_MODEL if prefer_lite else PRIMARY_MODEL
        try:
            raw = self._call(model, prompt_parts, response_schema)
        except RateLimitError:
            raw = self._call(FALLBACK_MODEL, prompt_parts, response_schema)

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return response_schema.model_validate(data)