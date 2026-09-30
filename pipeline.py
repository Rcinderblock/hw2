"""Text -> selected prompt -> model -> validated result."""

import logging
from typing import Protocol

from pydantic import ValidationError

from prompts import (
    DEFAULT_PROMPT_VARIANT,
    build_user_prompt,
    get_prompt_variant,
)
from schemas import TextAnalysis

logger = logging.getLogger(__name__)


class ModelClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class InvalidModelResponse(ValueError):
    """Keep rejected output available for inspecting a prompt comparison."""

    def __init__(self, message: str, response: str) -> None:
        super().__init__(message)
        self.response = response


def process_text(
    text: str,
    client: ModelClient,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> TextAnalysis:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    prompt = get_prompt_variant(prompt_variant)
    logger.info("Processing input text with prompt %s", prompt_variant)
    response = client.generate(
        prompt.system_prompt, build_user_prompt(text, prompt_variant)
    )
    if not response.strip():
        raise InvalidModelResponse(
            "The model returned an empty response", response
        )

    try:
        result = TextAnalysis.model_validate_json(response)
    except ValidationError as exc:
        logger.warning("Model response failed schema validation")
        raise InvalidModelResponse(
            "The model returned invalid JSON or fields", response
        ) from exc

    logger.info("Result validated")
    return result
