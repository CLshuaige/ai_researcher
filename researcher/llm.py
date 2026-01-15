"""LLM client implementations and model preset registry.

This module provides:
- Client classes for different LLM providers (OpenAI, Anthropic, etc.)
- A factory function to get the appropriate client based on provider
- A model preset registry for easy model selection
- Helper functions to manage model presets
"""

from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import os

from researcher.config import ModelConfig


# ============================================================================
# LLM Client Implementations
# ============================================================================


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API"""
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class AnthropicClient(BaseLLMClient):
    """Anthropic API client"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=config.api_key)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic API"""
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Extract system message if present
        system_message = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        response = self.client.messages.create(
            model=self.config.model_name,
            system=system_message,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text


def get_llm_client(config: ModelConfig) -> BaseLLMClient:
    """Factory function to get appropriate LLM client"""
    if config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


# ============================================================================
# Model Preset Registry
# ============================================================================


MODEL_PRESETS: Dict[str, ModelConfig] = {
    # OpenAI models
    "openai-gpt4o": ModelConfig(
        provider="openai",
        model_name="gpt-4o",
        api_key=None,  # Read from OPENAI_API_KEY if needed
        base_url=None,
        temperature=0.7,
        max_tokens=16384,
    ),

    # Local Qwen models (OpenAI-compatible HTTP endpoint)
    #
    # These assume you are running a local gateway that exposes an
    # OpenAI-compatible /v1/chat/completions endpoint, and that the
    # model_name below matches what the gateway expects.
    "qwen-30b-local": ModelConfig(
        provider="openai",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        api_key="dummy-key",
        base_url="http://127.0.0.1:8000/v1",
        temperature=0.7,
        max_tokens=4096,
    ),
    "qwen-80b-local": ModelConfig(
        provider="openai",
        model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key="dummy-key",
        base_url="http://127.0.0.1:8000/v1",
        temperature=0.7,
        max_tokens=4096,
    ),
}


# Default model preset name used when caller does not specify one explicitly
DEFAULT_MODEL_PRESET = "openai-gpt4o"


def get_model_preset(name: str) -> ModelConfig:
    """Get a model preset configuration by name.

    Args:
        name: Preset name (e.g. "openai-gpt4o", "qwen-30b-local").

    Returns:
        A ModelConfig instance describing the selected model.

    Raises:
        ValueError: If the preset name is not registered.
    """
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(
            f"Model preset '{name}' not found. "
            f"Available presets: {available}"
        )

    preset = MODEL_PRESETS[name]

    # If api_key is None, try to fill it from environment variables
    if preset.api_key is None:
        if preset.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif preset.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            api_key = None

        return ModelConfig(
            provider=preset.provider,
            model_name=preset.model_name,
            api_key=api_key,
            base_url=preset.base_url,
            temperature=preset.temperature,
            max_tokens=preset.max_tokens,
        )

    return preset


def list_available_models() -> Dict[str, ModelConfig]:
    """Return all available model presets."""
    return MODEL_PRESETS.copy()


def register_model_preset(name: str, config: ModelConfig) -> None:
    """Register a new model preset at runtime.

    Args:
        name: Preset name
        config: ModelConfig instance
    """
    MODEL_PRESETS[name] = config
