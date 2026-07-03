"""
Two clients:
- TextModelClient  → Groq (free, fast, text-only) — used for Steps A and D
- VisionModelClient → Gemini (free, multimodal)   — used for Step B only

Splitting this way cuts Gemini calls by ~60% and avoids 503s on text steps.
"""

import json
import os
from typing import Type, TypeVar

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from groq import Groq
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

T = TypeVar("T", bound=BaseModel)

GEMINI_MODEL   = "gemini-2.5-flash-lite"
GROQ_MODEL     = "llama-3.3-70b-versatile"   # fast, accurate, generous free tier


class RetryableError(Exception):
    pass


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (genai_errors.ServerError, genai_errors.ClientError)):
        code = getattr(e, 'status_code', 0) or 0
        if code in (429, 503):
            return True
    err_str = str(e)
    if any(x in err_str for x in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate_limit"]):
        return True
    return False


# ---------------------------------------------------------------------------
# Text client — Groq (Steps A and D)
# ---------------------------------------------------------------------------
class TextModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env and Colab Secrets.")
        self.client = Groq(api_key=api_key)

    def structured_call(
        self,
        prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Sends a text-only prompt to Groq and validates the JSON response
        against response_schema.
        """
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)

        system_prompt = (
            "You are a compliance analyst assistant. "
            "Always respond with valid JSON only — no markdown, no explanation, no backticks. "
            f"Your response must match this exact JSON schema:\n{schema_json}"
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        cleaned = (
            raw
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        return response_schema.model_validate(data)


# ---------------------------------------------------------------------------
# Vision client — Gemini (Step B only)
# ---------------------------------------------------------------------------
class VisionModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        self.client = genai.Client(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=3, min=5, max=60),
        retry=retry_if_exception_type(RetryableError),
    )
    def _call(self, contents: list, response_schema: Type[T]) -> str:
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
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
                print(f"Retryable error ({type(e).__name__}): retrying...")
                raise RetryableError(str(e)) from e
            raise

    def structured_call(
        self,
        prompt_parts: list,
        response_schema: Type[T],
    ) -> T:
        raw = self._call(prompt_parts, response_schema)
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        return response_schema.model_validate(data)