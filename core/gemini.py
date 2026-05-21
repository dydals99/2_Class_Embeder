"""Gemini LLM and embedding client (shared by both pipelines)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, TypeVar

import numpy as np
from google import genai
from google.genai import errors

from config import Config, get_config

T = TypeVar("T")

_RETRYABLE_CODES = frozenset({429, 500, 503})


class GeminiService:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self._client = genai.Client(api_key=self.cfg.gemini_api_key)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        text = f"{system}\n\n{prompt}" if system else prompt

        def _call(model: str) -> str:
            response = self._client.models.generate_content(
                model=model,
                contents=text,
            )
            out = getattr(response, "text", None) or ""
            if not out and response.candidates:
                parts = response.candidates[0].content.parts
                out = "".join(getattr(p, "text", "") or "" for p in parts)
            return out.strip()

        return self._with_retry(_call, operation="generate_content")

    def generate_json(self, prompt: str, *, system: str | None = None) -> dict[str, Any]:
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only. No markdown fences, no commentary."
        )
        raw = self.generate(full_prompt, system=system)
        return _parse_json(raw)

    def embed(self, text: str) -> list[float]:
        trimmed = text.strip()[:8000] or " "

        def _call(_model: str) -> list[float]:
            response = self._client.models.embed_content(
                model=self.cfg.gemini_embedding_model,
                contents=trimmed,
            )
            embeddings = response.embeddings
            if not embeddings:
                raise RuntimeError("Empty embedding response from Gemini.")
            return list(embeddings[0].values)

        return self._with_retry(
            _call,
            operation="embed_content",
            models=[self.cfg.gemini_embedding_model],
        )

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _model_chain(self, models: list[str] | None = None) -> list[str]:
        if models:
            return models
        chain = [self.cfg.gemini_model]
        fb = self.cfg.gemini_fallback_model
        if fb and fb not in chain:
            chain.append(fb)
        return chain

    def _with_retry(
        self,
        fn: Callable[[str], T],
        *,
        operation: str,
        models: list[str] | None = None,
    ) -> T:
        """Retry on 429/503/500; optionally fall back to GEMINI_FALLBACK_MODEL."""
        model_chain = self._model_chain(models)
        last_exc: Exception | None = None

        for model in model_chain:
            for attempt in range(self.cfg.gemini_max_retries):
                try:
                    return fn(model)
                except errors.APIError as exc:
                    last_exc = exc
                    if not _is_retryable(exc):
                        raise
                    if attempt < self.cfg.gemini_max_retries - 1:
                        wait = self.cfg.gemini_retry_base_sec * (2**attempt)
                        print(
                            f"[gemini] {operation} {model} failed ({exc.code}), "
                            f"retry {attempt + 1}/{self.cfg.gemini_max_retries} in {wait:.0f}s..."
                        )
                        time.sleep(wait)
                        continue
                    print(
                        f"[gemini] {operation} {model} exhausted retries ({exc.code})."
                    )
                    break
                except Exception:
                    raise

        if last_exc:
            raise last_exc
        raise RuntimeError(f"Gemini {operation} failed with no models configured.")


def _is_retryable(exc: errors.APIError) -> bool:
    return getattr(exc, "code", 0) in _RETRYABLE_CODES


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {raw[:300]}")
