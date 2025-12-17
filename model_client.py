"""
Model Client Abstraction Layer

Provides a unified interface for calling multiple LLM providers (Anthropic, Google)
with automatic handling of PDF encoding differences and metrics tracking.
"""

import os
import base64
import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

# Provider SDKs
from google import genai
from google.genai import types as google_types
import anthropic


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
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    "claude-opus-4-5": ModelConfig(
        model_id="claude-opus-4-5-20251101",
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Opus 4.5",
        input_price_per_million=5.0,
        output_price_per_million=25.0,
    ),
    "claude-sonnet-4-5": ModelConfig(
        model_id="claude-sonnet-4-5-20250929",
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Sonnet 4.5",
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    "claude-haiku-4-5": ModelConfig(
        model_id="claude-haiku-4-5-20251001",
        provider=ModelProvider.ANTHROPIC,
        display_name="Claude Haiku 4.5",
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    ),
    "gemini-3-pro": ModelConfig(
        model_id="gemini-3-pro-preview",
        provider=ModelProvider.GOOGLE,
        display_name="Gemini 3 Pro",
        input_price_per_million=1.25,
        output_price_per_million=10.0,
    ),
    "gemini-3-flash": ModelConfig(
    model_id="gemini-3-flash-preview",
    provider=ModelProvider.GOOGLE,
    display_name="Gemini 3 Flash (Preview)",
    input_price_per_million=0.50,
    output_price_per_million=3.00,
),
}

# Default model for backward compatibility
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
    
    def record_call(self, input_tokens: int, output_tokens: int):
        """Record metrics from a single API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
    
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


class ModelClient:
    """
    Unified client for calling different LLM providers.
    
    Handles:
    - Provider-specific SDK initialization
    - PDF encoding (base64 for Anthropic, raw bytes for Google)
    - Response normalization
    - Metrics tracking
    """
    
    def __init__(self):
        # Initialize Google client
        self.google_client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY", "")
        )
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    
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
    ) -> ModelResponse:
        """
        Generate content using the specified model.
        
        Args:
            prompt: The text prompt
            model_name: Key from SUPPORTED_MODELS (e.g., "claude-sonnet-4-5")
            pdf_bytes: Raw PDF bytes (will be encoded appropriately per provider)
            response_mime_type: For JSON responses, set to "application/json"
            temperature: Model temperature (provider defaults if None)
            max_tokens: Max output tokens (provider defaults if None)
            use_google_search: Enable Google Search grounding (Google only)
            tracker: Optional MetricsTracker to accumulate usage
            
        Returns:
            ModelResponse with normalized text and token counts
        """
        config = SUPPORTED_MODELS.get(model_name)
        if not config:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")
        
        if config.provider == ModelProvider.GOOGLE:
            response = self._call_google(
                config, prompt, pdf_bytes, response_mime_type, 
                temperature, max_tokens, use_google_search
            )
        else:
            response = self._call_anthropic(
                config, prompt, pdf_bytes, response_mime_type,
                temperature, max_tokens
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
    ) -> ModelResponse:
        """Call Google's Gemini API."""
        
        # Build contents
        contents = []
        if pdf_bytes:
            contents.append(
                google_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            )
        contents.append(prompt)
        
        # Build config
        gen_config_kwargs = {}
        if response_mime_type:
            gen_config_kwargs["response_mime_type"] = response_mime_type
        if temperature is not None:
            gen_config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = max_tokens
        
        # Add Google Search tool if requested
        if use_google_search:
            gen_config_kwargs["tools"] = [
                google_types.Tool(google_search=google_types.GoogleSearch())
            ]
        
        gen_config = google_types.GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None
        
        response = self.google_client.models.generate_content(
            model=config.model_id,
            contents=contents,
            config=gen_config,
        )
        
        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        
        return ModelResponse(
            text=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
    ) -> ModelResponse:
        """Call Anthropic's Claude API."""
        
        # Build message content
        content = []
        
        if pdf_bytes:
            # Anthropic requires base64-encoded PDFs
            pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                }
            })
        
        # If JSON response is expected, add explicit instruction
        if response_mime_type == "application/json":
            prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON with no markdown formatting, no ```json blocks, and no text before or after the JSON."
        
        content.append({
            "type": "text",
            "text": prompt,
        })
        
        # Build request kwargs
        request_kwargs = {
            "model": config.model_id,
            "max_tokens": max_tokens or 4096,
            "messages": [{"role": "user", "content": content}],
        }
        
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        
        response = self.anthropic_client.messages.create(**request_kwargs)
        
        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text
        
        # Clean JSON if that's what was requested
        if response_mime_type == "application/json":
            text = self._clean_json_response(text)
        
        return ModelResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_id=config.model_id,
        )
    
    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code blocks and other formatting from JSON responses."""
        text = text.strip()
        
        # Remove ```json ... ``` blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()


# Module-level singleton for convenience
_client: Optional[ModelClient] = None


def get_client() -> ModelClient:
    """Get or create the singleton ModelClient instance."""
    global _client
    if _client is None:
        _client = ModelClient()
    return _client


def get_supported_models() -> List[Dict[str, str]]:
    """Return list of supported models for frontend dropdown."""
    return [
        {
            "id": key,
            "name": config.display_name,
            "provider": config.provider.value,
        }
        for key, config in SUPPORTED_MODELS.items()
    ]
