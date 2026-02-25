# =========================
# model_client.py (UPDATED)
# =========================
"""
Model Client Abstraction Layer

Provides a unified interface for calling multiple LLM providers (Anthropic, Google)
with automatic handling of PDF encoding differences and metrics tracking.
"""

from __future__ import annotations

import os
import base64
import time
import json
import re
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum

# Provider SDKs
from google import genai
from google.genai import types as google_types
import anthropic

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Configuration for a supported model."""
    model_id: str
    provider: ModelProvider
    display_name: str
    input_price_per_million: float
    output_price_per_million: float


# Supported models registry
# NOTE: You can override model IDs via env vars if needed for rotation.
# Example: export MODEL_ID_CLAUDE_SONNET_4_5="claude-sonnet-4-5-latest"
def _env_override(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    "claude-opus-4-5": ModelConfig(
        model_id=_env_override("MODEL_ID_CLAUDE_OPUS_4_5", "claude-opus-4-5-20251101"),
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Opus 4.5",
        input_price_per_million=5.0,
        output_price_per_million=25.0,
    ),
    "claude-sonnet-4-5": ModelConfig(
        model_id=_env_override("MODEL_ID_CLAUDE_SONNET_4_5", "claude-sonnet-4-5-20250929"),
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Sonnet 4.5",
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    "claude-haiku-4-5": ModelConfig(
        model_id=_env_override("MODEL_ID_CLAUDE_HAIKU_4_5", "claude-haiku-4-5-20251001"),
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Haiku 4.5",
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    ),
    "gemini-3-pro": ModelConfig(
        model_id=_env_override("MODEL_ID_GEMINI_3_PRO", "gemini-3-pro-preview"),
        provider=ModelProvider.GOOGLE,
        display_name="Gemini 3 Pro",
        input_price_per_million=1.25,
        output_price_per_million=10.0,
    ),
    "gemini-3-flash": ModelConfig(
        model_id=_env_override("MODEL_ID_GEMINI_3_FLASH", "gemini-3-flash-preview"),
        provider=ModelProvider.GOOGLE,
        display_name="Gemini 3 Flash (Preview)",
        input_price_per_million=0.50,
        output_price_per_million=3.00,
    ),
}

# Default model (may not be available if env keys are missing)
DEFAULT_MODEL = "gemini-3-pro"


@dataclass
class ModelResponse:
    """Normalized response from any model."""
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str


@dataclass
class MetricsTracker:
    """Accumulates metrics across multiple API calls in a workflow."""
    model_name: str
    start_time: float = field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0

    def record_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record metrics from a single API call."""
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)
        self.call_count += 1

    def merge_counts(self, input_tokens: int, output_tokens: int, calls: int = 1) -> None:
        """Merge externally collected usage into this tracker (thread-safe-ish via GIL)."""
        self.total_input_tokens += int(input_tokens or 0)
        self.total_output_tokens += int(output_tokens or 0)
        self.call_count += int(calls or 0)

    def get_metrics(self) -> Dict[str, Any]:
        """Return final metrics summary."""
        elapsed = time.time() - self.start_time
        config = SUPPORTED_MODELS.get(self.model_name)

        if config:
            cost = (
                (self.total_input_tokens * config.input_price_per_million) +
                (self.total_output_tokens * config.output_price_per_million)
            ) / 1_000_000
            display_name = config.display_name
        else:
            cost = 0.0
            display_name = self.model_name

        return {
            "model_name": self.model_name,
            "model_display_name": display_name,
            "total_time_seconds": round(elapsed, 2),
            "_start_time": self.start_time,  # Preserve for accumulation across nodes
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "cost_usd": round(cost, 6),
            "api_calls": self.call_count,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]], model_name: str) -> "MetricsTracker":
        """Rebuild a tracker from serialized metrics_data."""
        d = d or {}
        t = cls(model_name=model_name)
        t.total_input_tokens = int(d.get("tokens", {}).get("input", 0) or 0)
        t.total_output_tokens = int(d.get("tokens", {}).get("output", 0) or 0)
        t.call_count = int(d.get("api_calls", 0) or 0)
        t.start_time = float(d.get("_start_time", t.start_time) or t.start_time)
        return t


def _has_env(var_name: str) -> bool:
    return bool(os.environ.get(var_name, "").strip())


def is_model_available(model_name: str) -> bool:
    config = SUPPORTED_MODELS.get(model_name)
    if not config:
        return False
    if config.provider == ModelProvider.GOOGLE:
        return _has_env("GOOGLE_API_KEY")
    if config.provider == ModelProvider.ANTHROPIC:
        return _has_env("ANTHROPIC_API_KEY")
    return False


def get_default_model() -> str:
    """Choose a default that is actually usable given current env vars."""
    if is_model_available(DEFAULT_MODEL):
        return DEFAULT_MODEL
    for key in SUPPORTED_MODELS.keys():
        if is_model_available(key):
            return key
    # Last resort (may still fail at runtime if no keys are set)
    return DEFAULT_MODEL


def get_supported_models(include_unavailable: bool = False) -> List[Dict[str, str]]:
    """Return list of supported models for frontend dropdown."""
    out: List[Dict[str, str]] = []
    for key, config in SUPPORTED_MODELS.items():
        available = is_model_available(key)
        if (not include_unavailable) and (not available):
            continue
        out.append(
            {
                "id": key,
                "name": config.display_name,
                "provider": config.provider.value,
                "available": str(available).lower(),
            }
        )
    return out


def clean_json_text(text: str) -> str:
    """Strip markdown code fences and trim."""
    s = (text or "").strip()
    # Remove fenced blocks
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    elif s.startswith("```"):
        s = s[len("```"):].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s.strip()


def extract_first_json_substring(text: str) -> str:
    """
    Extract the first balanced JSON object/array substring from text.
    Falls back to cleaned text if no balanced JSON found.
    """
    s = clean_json_text(text)

    # Find first object or array start
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = s.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1].strip()

    # Heuristic: if there is *some* JSON-ish region, try a regex grab
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        return m.group(1).strip()

    return s.strip()


JSONType = Union[Dict[str, Any], List[Any]]


def parse_json_safely(text: str) -> JSONType:
    """
    Parse JSON from LLM output robustly (handles markdown fences and stray pre/post text).
    Raises ValueError if it cannot parse.
    """
    candidate = extract_first_json_substring(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        snippet = candidate[:500].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON: {e}. Snippet: {snippet}") from e


class ModelClient:
    """
    Unified client for calling different LLM providers.

    Handles:
    - Provider-specific SDK initialization (lazy)
    - PDF encoding (base64 for Anthropic, raw bytes for Google)
    - Response normalization
    - Metrics tracking
    - Optional system prompts
    """

    def __init__(self):
        self._google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        self._anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        self.google_client: Optional[genai.Client] = None
        self.anthropic_client: Optional[anthropic.Anthropic] = None

    def _ensure_google(self) -> None:
        if not self._google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required to use Gemini models.")
        if self.google_client is None:
            self.google_client = genai.Client(api_key=self._google_api_key)

    def _ensure_anthropic(self) -> None:
        if not self._anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required to use Claude models.")
        if self.anthropic_client is None:
            self.anthropic_client = anthropic.Anthropic(api_key=self._anthropic_api_key)

    def generate_content(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL,
        pdf_bytes: Optional[bytes] = None,
        response_mime_type: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_google_search: bool = False,
        tracker: Optional[MetricsTracker] = None,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate content using the specified model.
        """
        config = SUPPORTED_MODELS.get(model_name)
        if not config:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")

        # Add explicit JSON-only instruction for BOTH providers if requested.
        if response_mime_type == "application/json":
            prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown fences, no commentary, no leading/trailing text."
            )

        if config.provider == ModelProvider.GOOGLE:
            response = self._call_google(
                config=config,
                prompt=prompt,
                pdf_bytes=pdf_bytes,
                response_mime_type=response_mime_type,
                temperature=temperature,
                max_tokens=max_tokens,
                use_google_search=use_google_search,
                system_prompt=system_prompt,
            )
        else:
            response = self._call_anthropic(
                config=config,
                prompt=prompt,
                pdf_bytes=pdf_bytes,
                response_mime_type=response_mime_type,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

        if tracker:
            tracker.record_call(response.input_tokens, response.output_tokens)

        return response

    def _call_google(
        self,
        config: ModelConfig,
        prompt: str,
        pdf_bytes: Optional[bytes],
        response_mime_type: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        use_google_search: bool,
        system_prompt: Optional[str],
    ) -> ModelResponse:
        """Call Google's Gemini API."""
        self._ensure_google()

        # Build contents
        contents: List[Any] = []
        if pdf_bytes:
            contents.append(google_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
        contents.append(prompt)

        # Build config
        gen_config_kwargs: Dict[str, Any] = {}
        if response_mime_type:
            gen_config_kwargs["response_mime_type"] = response_mime_type
        if temperature is not None:
            gen_config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = max_tokens
        if system_prompt:
            gen_config_kwargs["system_instruction"] = system_prompt

        # Add Google Search tool if requested
        if use_google_search:
            gen_config_kwargs["tools"] = [google_types.Tool(google_search=google_types.GoogleSearch())]

        gen_config = google_types.GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None

        response = self.google_client.models.generate_content(
            model=config.model_id,
            contents=contents,
            config=gen_config,
        )

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        text = getattr(response, "text", "") or ""
        if response_mime_type == "application/json":
            # Keep it as text, but clean common fences
            text = extract_first_json_substring(text)

        return ModelResponse(
            text=text,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            model_id=config.model_id,
        )

    def _call_anthropic(
        self,
        config: ModelConfig,
        prompt: str,
        pdf_bytes: Optional[bytes],
        response_mime_type: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        system_prompt: Optional[str],
    ) -> ModelResponse:
        """Call Anthropic's Claude API."""
        self._ensure_anthropic()

        content: List[Dict[str, Any]] = []

        if pdf_bytes:
            pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                }
            )

        content.append({"type": "text", "text": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": config.model_id,
            "max_tokens": max_tokens or 4096,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if system_prompt:
            request_kwargs["system"] = system_prompt

        response = self.anthropic_client.messages.create(**request_kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        if response_mime_type == "application/json":
            text = extract_first_json_substring(text)

        return ModelResponse(
            text=text.strip(),
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            model_id=config.model_id,
        )


# Module-level singleton for convenience
_client: Optional[ModelClient] = None


def get_client() -> ModelClient:
    """Get or create the singleton ModelClient instance."""
    global _client
    if _client is None:
        _client = ModelClient()
    return _client
