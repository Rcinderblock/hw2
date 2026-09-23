"""One Day 1 step: text -> prompt -> model -> validated result."""

import logging
from typing import Protocol

from pydantic import ValidationError

from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import TextAnalysis

logger = logging.getLogger(__name__)


class ModelClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def process_text(text: str, client: ModelClient) -> TextAnalysis:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    logger.info("Processing input text")
    response = client.generate(SYSTEM_PROMPT, build_user_prompt(text))
    if not response.strip():
        raise ValueError("The model returned an empty response")

    try:
        result = TextAnalysis.model_validate_json(response)
    except ValidationError as exc:
        logger.warning("Model response failed schema validation")
        raise ValueError("The model returned invalid JSON or fields") from exc

    logger.info("Result validated")
    return result
