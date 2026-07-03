"""
Both clients use Groq only — no Gemini dependency.
TextModelClient  → llama-3.3-70b-versatile   — Steps A and D (text)
VisionModelClient → llama-4-scout-17b         — Step B (vision)
"""

import base64
import io
import json
import os
from typing import Type, TypeVar

from groq import Groq
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

T = TypeVar("T", bound=BaseModel)

TEXT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class RetryableError(Exception):
    pass


def _is_retryable(e: Exception) -> bool:
    err_str = str(e)
    return any(x in err_str for x in ["429", "503", "rate_limit", "overloaded"])


def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string for Groq vision API."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Text client — Steps A and D
# ---------------------------------------------------------------------------
class TextModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception_type(RetryableError),
    )
    def structured_call(self, prompt: str, response_schema: Type[T]) -> T:
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        system_prompt = (
            "You are a compliance analyst assistant. "
            "Always respond with valid JSON only — no markdown, no explanation, no backticks. "
            f"Your response must match this exact JSON schema:\n{schema_json}"
        )
        try:
            response = self.client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if _is_retryable(e):
                print("Retryable error (text): retrying...")
                raise RetryableError(str(e)) from e
            raise

        raw = response.choices[0].message.content.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return response_schema.model_validate(data)


# ---------------------------------------------------------------------------
# Vision client — Step B
# ---------------------------------------------------------------------------
class VisionModelClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception_type(RetryableError),
    )
    def structured_call(self, prompt_parts: list, response_schema: Type[T]) -> T:
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)

        # Build Groq vision content — images first, then text prompt
        content = []
        for part in prompt_parts:
            if isinstance(part, Image.Image):
                b64 = _pil_to_base64(part)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            elif isinstance(part, str):
                content.append({"type": "text", "text": part})

        # Append schema instruction at the end
        content.append({
            "type": "text",
            "text": (
                f"\nReturn ONLY valid JSON matching this schema — "
                f"no markdown, no backticks:\n{schema_json}"
            )
        })

        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if _is_retryable(e):
                print("Retryable error (vision): retrying...")
                raise RetryableError(str(e)) from e
            raise

        raw = response.choices[0].message.content.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return response_schema.model_validate(data)