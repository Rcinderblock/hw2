"""Text -> selected prompt -> model -> validated result."""

import json
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


def describe_schema_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors(include_url=False, include_input=False):
        field = ".".join(str(part) for part in error["loc"]) or "ответ"
        context = error.get("ctx", {})
        explanations = {
            "missing": "обязательное поле отсутствует",
            "string_type": "ожидалась строка",
            "list_type": "ожидался список",
            "model_type": "ожидался JSON-объект",
            "extra_forbidden": "лишнее поле",
            "string_too_short": "значение не должно быть пустым",
            "string_too_long": (
                f"не больше {context.get('max_length')} символов"
            ),
            "too_short": "ожидалось ровно три ключевые мысли",
            "too_long": "ожидалось ровно три ключевые мысли",
            "literal_error": (
                f"допустимые значения: {context.get('expected')}"
            ),
            "value_error": str(context.get("error", "неверное значение")),
        }
        messages.append(
            f"{field}: {explanations.get(error['type'], error['msg'])}"
        )
    return "Ответ не соответствует схеме: " + "; ".join(messages)


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
        raise InvalidModelResponse("Модель вернула пустой ответ", response)

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponse(
            f"Некорректный JSON: строка {exc.lineno}, столбец {exc.colno}",
            response,
        ) from exc

    try:
        result = TextAnalysis.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Model response failed schema validation")
        raise InvalidModelResponse(
            describe_schema_error(exc), response
        ) from exc

    logger.info("Result validated")
    return result
