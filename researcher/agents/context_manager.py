"""Agent Context Manager for automatic message compression and token limiting.

This module provides a context manager that automatically applies intelligent
message compression and token limiting to all agents to prevent context length errors.
"""

from typing import Dict, Any, Optional
from autogen import ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import (
    TextMessageCompressor,
    MessageTokenLimiter,
)
from autogen.agentchat.contrib.capabilities import transforms_util

# Conservative estimates for common models
MODEL_CONTEXT_LENGTHS: Dict[str, int] = {
    "gpt": 128000,
    "claude": 200000,
    "qwen": 32768,
}

class EITextMessageCompressor(TextMessageCompressor):
    """
    TextMessageCompressor for Except the Instructions in the last message."""

    def apply_transform(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Applies compression to messages in a conversation history based on the specified configuration.

        The function processes each message according to the `compression_args` and `min_tokens` settings, applying
        the specified compression configuration and returning a new list of messages with reduced token counts
        where possible.

        Args:
            messages (List[Dict]): A list of message dictionaries to be compressed.

        Returns:
            List[Dict]: A list of dictionaries with the message content compressed according to the configured
                method and scope.
        """
        # Make sure there is at least one message
        if not messages:
            return messages

        # if the total number of tokens in the messages is less than the min_tokens, return the messages as is
        if not transforms_util.min_tokens_reached(messages, self._min_tokens):
            return messages

        total_savings = 0
        processed_messages = messages.copy()

        for message in processed_messages[:-1]: # Iterate over all but the last message
            # Some messages may not have content.
            if not transforms_util.is_content_right_type(message.get("content")):
                continue

            if not transforms_util.should_transform_message(message, self._filter_dict, self._exclude_filter):
                continue

            if transforms_util.is_content_text_empty(message["content"]):
                continue

            cache_key = transforms_util.cache_key(message["content"], self._min_tokens)
            cached_content = transforms_util.cache_content_get(self._cache, cache_key)
            if cached_content is not None:
                message["content"], savings = cached_content
            else:
                message["content"], savings = self._compress(message["content"])

            transforms_util.cache_content_set(self._cache, cache_key, message["content"], savings)

            assert isinstance(savings, int)
            total_savings += savings

        self._recent_tokens_savings = total_savings
        #print(f"+++++++++++++++++++++++++++++++++\n{processed_messages}\n+++++++++++++++++++++++++++++++++")
        return processed_messages




class AgentContextManager:
    """Manages context compression and token limiting for agents.
    
    Usage:
        manager = AgentContextManager(
            max_context_tokens=32768,
            compression_threshold=0.8  # Compress when 80% of max tokens reached
        )
        manager.apply_to_agent(agent)
    """

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        compression_threshold: float = None,
        compression_params: Optional[Dict[str, Any]] = None,
        enable_compression: bool = True,
        enable_token_limiting: bool = True,
        safety_margin: float = None,
    ):
        """
        Args:
            max_context_tokens: Maximum context window size for the model (includes both input and output).
                If None, will try to infer from model name in llm_config.
            compression_threshold: Fraction of max_context_tokens at which to
                start compression (default: 0.8 = 80%).
            compression_params: Additional parameters for TextMessageCompressor.
            enable_compression: Whether to enable message compression.
            enable_token_limiting: Whether to enable token limiting.
            safety_margin: Fraction of max_context_tokens reserved for input messages.
                The remaining (1 - safety_margin) is reserved for model output.
                Example: safety_margin=0.5 means input can use 50% of context, output gets 50%.
        """
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        self.compression_params = compression_params or {}
        self.enable_compression = enable_compression
        self.enable_token_limiting = enable_token_limiting
        self.safety_margin = safety_margin

    @staticmethod
    def infer_max_tokens_from_model(model_name: str) -> Optional[int]:
        """Infer maximum context tokens from model name.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
            
        Returns:
            Maximum context tokens if known, None otherwise.
        """
        model_name_lower = model_name.lower()
        
        # Check exact match first
        if model_name_lower in MODEL_CONTEXT_LENGTHS:
            return MODEL_CONTEXT_LENGTHS[model_name_lower]
        
        # Check partial matches for model families
        for key, value in MODEL_CONTEXT_LENGTHS.items():
            if key in model_name_lower or model_name_lower in key:
                return value
        
        if "qwen" in model_name_lower:
            return MODEL_CONTEXT_LENGTHS.get("qwen", 32768)
        
        # Default fallback for unknown models
        return None

    def get_max_tokens(self, llm_config: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Get maximum context tokens from config or inference.
        
        Args:
            llm_config: LLM configuration dictionary.
            
        Returns:
            Maximum context tokens.
        """
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        
        if llm_config:
            # Try to get from config_list
            config_list = llm_config.get("config_list", [])
            if config_list and len(config_list) > 0:
                model_name = config_list[0].get("model", "")
                if model_name:
                    inferred = self.infer_max_tokens_from_model(model_name)
                    if inferred:
                        return inferred
        
        return 32768  # Default fallback

    def apply_to_agent(
        self,
        agent: ConversableAgent,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Apply context compression and token limiting to an agent.
        
        This method adds TransformMessages capability to the agent with:
        1. TextMessageCompressor (if enabled) - compresses messages when threshold reached
        2. MessageTokenLimiter (if enabled) - ensures total tokens stay within limit
        
        Args:
            agent: The ConversableAgent to apply transformations to.
            llm_config: Optional LLM configuration to infer max tokens.
        """
        max_tokens = self.get_max_tokens(llm_config)
        if max_tokens is None:
            # If we can't determine max tokens, skip applying transforms
            return

        model_name: Optional[str] = None
        if llm_config:
            config_list = llm_config.get("config_list", [])
            if config_list and len(config_list) > 0:
                model_name = config_list[0].get("model") or None

        # tiktoken model for approximate counting
        limiter_model = "gpt-4-32k"
        if isinstance(model_name, str):
            model_lower = model_name.lower()
            if model_lower.startswith("gpt-"):
                limiter_model = model_name

        transforms = []

        compression_min_tokens: int = 0
        if self.enable_compression:
            # Calculate compression threshold
            compression_min_tokens = int(max_tokens * self.compression_threshold)
            
            compressor = EITextMessageCompressor(                         # Apply TextMessageCompressor
                min_tokens=compression_min_tokens,
                compression_params=self.compression_params,
            )
            transforms.append(compressor)

        if self.enable_token_limiting:
            # Apply safety margin: limit input messages to (safety_margin * max_tokens)
            # This reserves (1 - safety_margin) * max_tokens for model output
            # Example: if max_tokens=32768 and safety_margin=0.5:
            #   - Input messages limited to 16384 tokens
            #   - Output can use up to 16384 tokens
            safe_max_tokens = int(max_tokens * self.safety_margin)

            token_limiter = MessageTokenLimiter(                         # Apply MessageTokenLimiter
                max_tokens=safe_max_tokens,
                min_tokens=compression_min_tokens,
                model=limiter_model,
            )
            transforms.append(token_limiter)

        if transforms:
            transform_capability = transform_messages.TransformMessages(transforms=transforms)
            transform_capability.add_to_agent(agent)

    @classmethod
    def create_for_agent(
        cls,
        agent: ConversableAgent,
        llm_config: Optional[Dict[str, Any]] = None,
        max_context_tokens: Optional[int] = None,
        compression_threshold: float = 0.8,
        compression_params: Optional[Dict[str, Any]] = None,
        enable_compression: bool = True,
        enable_token_limiting: bool = True,
        safety_margin: float = 0.9,
    ) -> None:
        """Convenience method to create and apply context manager to an agent.

        Args:
            agent: The ConversableAgent to apply transformations to.
            llm_config: Optional LLM configuration.
            max_context_tokens: Maximum context length (auto-inferred if None).
            compression_threshold: Fraction at which to start compression.
            compression_params: Additional compression parameters.
            enable_compression: Whether to enable compression.
            enable_token_limiting: Whether to enable token limiting.
            safety_margin: Safety margin for token limiting (default: 0.9).
        """
        manager = cls(
            max_context_tokens=max_context_tokens,
            compression_threshold=compression_threshold,
            compression_params=compression_params,
            enable_compression=enable_compression,
            enable_token_limiting=enable_token_limiting,
            safety_margin=safety_margin,
        )
        manager.apply_to_agent(agent, llm_config)

    @classmethod
    def create_from_config(
        cls,
        config: Dict[str, Any],
        agent: ConversableAgent,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create context manager from configuration and apply to agent.

        Args:
            config: Context management configuration from YAML.
            agent: The ConversableAgent to apply transformations to.
            llm_config: Optional LLM configuration.
        """
        # Extract settings from config
        enable_compression = config.get("enable_compression", False)
        compression_threshold = config.get("compression_threshold", 0.8)
        max_context_tokens = config.get("max_context_tokens")
        safety_margin = config.get("safety_margin", 0.9)

        # Compression settings
        compression_config = config.get("compression", {})
        enable_text_compression = compression_config.get("enable_text_compression", False)
        compression_params = compression_config.get("compression_params", {})

        # Token limiting settings
        token_limiting_config = config.get("token_limiting", {})
        enable_token_limiting = token_limiting_config.get("enable_token_limiting", True)

        # Create and apply the manager
        manager = cls(
            max_context_tokens=max_context_tokens,
            compression_threshold=compression_threshold,
            compression_params=compression_params,
            enable_compression=enable_text_compression and enable_compression,
            enable_token_limiting=enable_token_limiting and enable_compression,
            safety_margin=safety_margin,
        )
        manager.apply_to_agent(agent, llm_config)

    @classmethod
    def apply_message_history_limiting(
        cls,
        agent: ConversableAgent,
        config: Dict[str, Any],
    ) -> None:
        """Apply message history limiting to an agent based on configuration.

        Args:
            agent: The ConversableAgent to apply limiting to.
            config: Message history configuration from YAML.
        """
        from autogen.agentchat.contrib.capabilities import transform_messages
        from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter

        if not config.get("enable_history_limiting", True):
            return

        max_messages = config.get("max_messages", 10)
        keep_first = config.get("keep_first_message", True)

        message_limiter = MessageHistoryLimiter(
            max_messages=max_messages,
            keep_first_message=keep_first
        )
        transform_messages.TransformMessages(transforms=[message_limiter]).add_to_agent(agent)
