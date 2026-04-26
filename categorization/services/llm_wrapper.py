"""
LLM wrapper for Groq API integration.

Provides an abstraction layer for interacting with Groq API via Python SDK
with retry logic, comprehensive error handling, and structured logging.

Architecture:
  - LLMProvider (ABC): Base interface for LLM providers
  - GroqProvider: Groq-specific implementation
  - LLMWrapper: High-level client wrapper with retry logic
"""

import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from groq import Groq

from categorization.exceptions import LLMException, ConfigurationException
from categorization.services.constants import (
    DEFAULT_LLM_MODEL,
    LLM_API_TIMEOUT,
    MAX_RETRIES,
    BASE_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Defines the interface that all LLM implementations must follow.
    This allows for easy swapping between different LLM providers (Groq, OpenAI, etc.)
    """

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response as a string
            
        Raises:
            LLMException: If the LLM call fails
            ConfigurationException: If the provider is not properly configured
        """
        pass


class GroqProvider(LLMProvider):
    """
    Groq API provider implementation.
    
    Uses Groq Python SDK to generate categorization responses with automatic
    retry logic for transient failures.
    
    Attributes:
        client: Groq client instance
        model: Model name (default: llama-3.1-8b-instant)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None) -> None:
        """
        Initialize the Groq provider.
        
        Args:
            api_key: Groq API key (optional, defaults to GROQ_API_KEY env var)
            model: Model name (optional, defaults to DEFAULT_LLM_MODEL)
            
        Raises:
            ConfigurationException: If api_key is not provided or invalid
        """
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key or not isinstance(api_key, str) or api_key.strip() == '':
            raise ConfigurationException(
                "GROQ_API_KEY is required and must be a non-empty string. "
                "Set it in environment variables or pass as parameter."
            )
        
        try:
            self.client = Groq(api_key=api_key, timeout=LLM_API_TIMEOUT)
            self.model = model or DEFAULT_LLM_MODEL
            logger.info(f"Groq provider initialized successfully (model: {self.model})")
        except Exception as e:
            raise ConfigurationException(f"Failed to initialize Groq client: {str(e)}")

    def generate_response(self, prompt: str, request_id: str = None) -> str:
        """
        Generate a response using Groq API with automatic retry logic.
        
        Implements exponential backoff for transient failures.
        
        Args:
            prompt: The categorization prompt
            request_id: Optional request ID for tracing
            
        Returns:
            The LLM response as a string
            
        Raises:
            LLMException: If the API call fails after all retries
        """
        if not prompt or not isinstance(prompt, str):
            raise LLMException("Prompt must be a non-empty string")

        for attempt in range(MAX_RETRIES):
            try:
                # Generate unique trace ID for request tracking
                trace_id = request_id or f"req-{uuid.uuid4().hex[:8]}"
                logger.debug(f"[{trace_id}] Attempt {attempt + 1}/{MAX_RETRIES}: "
                           f"Sending request to Groq API")
                
                # Call Groq API with categorization prompt
                # temperature=0.3 ensures consistent, deterministic results
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert accounting categorization assistant. Always respond with valid JSON only. No markdown, no extra text."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3  # Lower temperature = more consistent categorization
                )
                
                # Extract the text response from the API response object
                response = completion.choices[0].message.content
                
                # Ensure we got a valid response (not empty)
                if not response:
                    raise LLMException("Received empty response from Groq API")
                
                # Log successful response
                logger.debug(f"[{trace_id}] Successfully received response from Groq API "
                           f"(length: {len(response)} chars)")
                return response
            
            except LLMException:
                # Re-raise our own exceptions without retry
                raise
            except Exception as e:
                error_msg = str(e)
                is_last_attempt = attempt == MAX_RETRIES - 1
                
                if is_last_attempt:
                    # All retries exhausted, raise exception
                    logger.error(f"Failed to call Groq API after {MAX_RETRIES} attempts: {error_msg}")
                    raise LLMException(f"Groq API call failed: {error_msg}")
                else:
                    # Transient failure - wait and retry with exponential backoff
                    # Backoff: 1s, 2s, 4s (prevents thundering herd problem)
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Groq API error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}. "
                                 f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)  # Wait before next attempt


class LLMWrapper:
    """
    High-level wrapper for LLM operations.
    
    Provides a clean interface for getting categorization suggestions from the LLM,
    abstracting away provider-specific details.
    
    Attributes:
        provider: An LLMProvider instance
    """

    def __init__(self, provider: LLMProvider) -> None:
        """
        Initialize the LLM wrapper.
        
        Args:
            provider: An LLMProvider instance (e.g., GroqProvider)
            
        Raises:
            ValueError: If provider is not an instance of LLMProvider
        """
        if not provider or not isinstance(provider, LLMProvider):
            raise ValueError(
                "Provider must be an instance of LLMProvider. "
                f"Got: {type(provider)}"
            )
        
        self.provider = provider
        logger.info(f"LLM wrapper initialized with provider: {type(provider).__name__}")

    def categorize(self, prompt: str, request_id: str = None) -> str:
        """
        Get categorization suggestion from the LLM.
        
        Args:
            prompt: The structured prompt for categorization
            request_id: Optional request ID for distributed tracing
            
        Returns:
            The LLM response as a string
            
        Raises:
            LLMException: If the LLM call fails
            ValueError: If prompt is invalid
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        try:
            trace_id = request_id or f"cat-{int(time.time())}"
            logger.info(f"[{trace_id}] Requesting categorization from LLM "
                       f"(prompt length: {len(prompt)} chars)")
            response = self.provider.generate_response(prompt, request_id=trace_id)
            logger.info(f"[{trace_id}] Successfully obtained categorization response")
            return response
        except LLMException:
            raise  # Re-raise LLM exceptions
        except Exception as e:
            logger.error(f"Unexpected error during categorization: {str(e)}")
            raise LLMException(f"Categorization failed: {str(e)}")
