"""Define a client class to interface with OpenAI's GPT-4 API."""

from __future__ import annotations

import base64
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from openai import OpenAI

T = TypeVar("T")
P = ParamSpec("P")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPT4Config:
    """Configuration for GPT-4 API calls with sensible defaults."""

    model: str = "gpt-4.1-2025-04-14"
    """Model identifier/snapshot. Options include:
        - "gpt-4.1-2025-04-14": Flagship model for complex tasks (default)
        - "o3-2025-04-16": Powerful model for solving multi-step problems
    """

    temperature: float = 1.0
    """Sampling temperature used by the model (between 0 and 2)."""

    max_output_tokens: int = 500
    """Upper bound on the number of tokens in response (at least 16; up to 32768 for GPT-4.1)."""

    def to_params(self, **overrides: Any) -> dict[str, Any]:
        """Convert the config into an OpenAI API parameters dictionary.

        :param overrides: Override any config values for this specific call
        :return: Dictionary of parameters for the OpenAI API
        """
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        params.update(overrides)

        return params


class GPT4Client:
    """Simplified OpenAI GPT-4 client with text and multimodal support."""

    def __init__(
        self,
        config: GPT4Config | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
    ) -> None:
        """Initialize the GPT-4 client with the given configuration.

        :param config: GPT-4 configuration for API calls (if None, uses default values)
        :param api_key: OpenAI API key (if None, falls back to OPENAI_API_KEY env var)
        :param max_retries: Maximum retry attempts on failure (defaults to 3)
        :param base_delay_s: Base delay (seconds) for exponential backoff retries (default: 1.0 s)
        """
        self.config = config or GPT4Config()
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

        # Initialize API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided or set in the OPENAI_API_KEY environment variable.",
            )

        self.client = OpenAI(api_key=api_key)

    def _execute_with_retry(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute the given function with an exponential backoff retry.

        :param func: Function to execute
        :param args: Arguments to pass to the function
        :param kwargs: Keyword arguments to pass to the function
        :return: Result of successful function call
        :raises: Last exception if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: PERF203
                if attempt == self.max_retries - 1:
                    error = f"Calling function {func} failed {self.max_retries} times."
                    raise RuntimeError(error) from exc

                delay_s = self.base_delay_s * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {exc}. Retrying in {delay_s} s...")
                time.sleep(delay_s)

        raise RuntimeError(f"Unable to call function {func} after {self.max_retries} attempts.")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a GPT-4 text response to the given prompt.

        :param prompt: Text prompt provided to the LLM
        :param kwargs: Override any config parameters for this call only
            - model: Use a different model temporarily
            - temperature: Adjust randomness for this call
            - max_output_tokens: Adjust response length limit
        :return: Generated text response
        """
        params = self.config.to_params(input=prompt, **kwargs)

        response = self._execute_with_retry(self.client.responses.create, **params)
        return response.output_text.strip()

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """Encode the image at the given path into a string using base64."""
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_multimodal(self, prompt: str, images: list[str | Path], **kwargs: Any) -> str:
        """Generate a GPT-4 text response to the given prompt and image(s).

        Reference: https://platform.openai.com/docs/guides/images-vision?format=url#analyze-images

        :param prompt: Text prompt describing the task or questions about the image(s)
        :param images: List of image paths (local files) or URLs
        :param kwargs: Override any config parameters for this call only
            - model: Use a different model temporarily
            - temperature: Adjust randomness for this call
            - max_output_tokens: Adjust response length limit
        :return: Generated text response
        """
        # Build a list of content in the OpenAI Responses API format
        content = [{"type": "input_text", "text": prompt}]

        for image in images:
            image_str = str(image)
            if image_str.startswith(("http://", "https://")):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": image_str,
                    },
                )
            else:  # Encode local image files using base64
                image_path = Path(image)
                assert image_path.exists(), f"Cannot encode nonexistent image file: {image_path}"

                base64_image = self.encode_image(image_path)
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                )

        # Create input in Responses API format
        input_data = [{"role": "user", "content": content}]
        params = self.config.to_params(input=input_data, **kwargs)

        response = self._execute_with_retry(self.client.responses.create, **params)
        return response.output_text.strip()
