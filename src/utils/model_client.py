"""
Thin wrapper around the Gemini API.

Centralizes: retry/backoff on free-tier 429s and 503s, automatic fallback
from gemini-2.5-flash to gemini-2.5-flash-lite, and structured JSON parsing
+ schema validation for every call.
"""

import json
import os
from typing import Type, TypeVar

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

T = TypeVar("T", bound=BaseModel)

PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash-lite"


class RetryableError(Exception):
    pass


def _is_retryable(e: Exception) -> bool:
    """
    Check if an exception should trigger a retry.
    Handles both old-style string matching and new-style SDK exception classes.
    """
    # New google-genai SDK raises typed exceptions — catch them directly
    if isinstance(e, (genai_errors.ServerError, genai_errors.ClientError)):
        code = getattr(e, 'status_code', 0) or 0
        if code in (429, 503):
            return True

    # Fallback: string matching for any SDK version
    err_str = str(e)
    if any(x in err_str for x in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
        return True

    return False


class ModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = genai.Client(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=3, min=5, max=60),
        retry=retry_if_exception_type(RetryableError),
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
            if _is_retryable(e):
                print(f"Retryable error hit ({type(e).__name__}): {e}. Retrying...")
                raise RetryableError(str(e)) from e
            raise

    def structured_call(
        self,
        prompt_parts: list,
        response_schema: Type[T],
        prefer_lite: bool = False,
    ) -> T:
        """
        prompt_parts: list of strings and/or PIL.Image objects.
        response_schema: a Pydantic model class.
        Returns a validated instance of response_schema.
        """
        model = FALLBACK_MODEL if prefer_lite else PRIMARY_MODEL
        try:
            raw = self._call(model, prompt_parts, response_schema)
        except RetryableError:
            # Primary exhausted all retries -> fall back to lite model once more
            print(f"Primary model exhausted. Falling back to {FALLBACK_MODEL}...")
            raw = self._call(FALLBACK_MODEL, prompt_parts, response_schema)

        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        return response_schema.model_validate(data)