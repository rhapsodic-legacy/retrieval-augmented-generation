"""
LLM Provider Abstraction Layer

Supports multiple LLM backends:
- Anthropic (Claude)
- Google (Gemini)

Easily extensible to add more providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Generator
from enum import Enum


class ProviderType(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: ProviderType
    usage: dict = field(default_factory=dict)  # token counts
    metadata: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream a response from the LLM."""
        pass
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, etc.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            model: Claude model to use
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        import anthropic
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response using Claude."""
        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ("user", "assistant")
        ]
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider=ProviderType.ANTHROPIC,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
    
    def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream a response using Claude."""
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ("user", "assistant")
        ]
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider.
    
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, etc.
    """
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Gemini provider.
        
        Args:
            model: Gemini model to use
            api_key: API key (defaults to GOOGLE_API_KEY env var)
        """
        import google.generativeai as genai
        import os
        
        self.model = model
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
        
        self.client = genai.GenerativeModel(model)
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GEMINI
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def _convert_messages(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None
    ) -> tuple[list[dict], Optional[str]]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            if msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
        
        return gemini_messages, system_prompt
    
    def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        import google.generativeai as genai
        
        gemini_messages, system = self._convert_messages(messages, system_prompt)
        
        # Configure generation
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Start chat with history
        chat = self.client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        # Add system prompt to the last message if provided
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        if system:
            last_message = f"[System: {system}]\n\n{last_message}"
        
        response = chat.send_message(last_message, generation_config=generation_config)
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            provider=ProviderType.GEMINI,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            },
        )
    
    def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream a response using Gemini."""
        import google.generativeai as genai
        
        gemini_messages, system = self._convert_messages(messages, system_prompt)
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        chat = self.client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        if system:
            last_message = f"[System: {system}]\n\n{last_message}"
        
        response = chat.send_message(
            last_message,
            generation_config=generation_config,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


def create_provider(
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Factory function to create an LLM provider.
    
    Args:
        provider: "anthropic" or "gemini"
        model: Model name (uses default if not specified)
        api_key: API key (uses env var if not specified)
        
    Returns:
        Configured LLM provider
    """
    provider = provider.lower()
    
    if provider == "anthropic":
        return AnthropicProvider(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
        )
    elif provider == "gemini":
        return GeminiProvider(
            model=model or "gemini-1.5-pro",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'gemini'.")
