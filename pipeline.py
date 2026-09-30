"""Text -> classification -> category instructions -> answer -> result."""

import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from prompts import (
    DEFAULT_PROMPT_VARIANT,
    build_answer_system_prompt,
    build_answer_user_prompt,
    build_user_prompt,
    get_prompt_variant,
)
from schemas import GeneratedAnswer, TextAnalysis, TextClassification

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ModelClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str: ...


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


def parse_model_response(
    response: str, schema: type[ResponseModel], stage: str
) -> ResponseModel:
    if not response.strip():
        raise InvalidModelResponse(
            f"{stage}: модель вернула пустой ответ", response
        )

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponse(
            f"{stage}: Некорректный JSON: строка {exc.lineno}, "
            f"столбец {exc.colno}",
            response,
        ) from exc

    try:
        result = schema.model_validate(payload)
    except ValidationError as exc:
        logger.warning("%s failed schema validation", stage)
        raise InvalidModelResponse(
            f"{stage}: {describe_schema_error(exc)}", response
        ) from exc

    return result


def process_text(
    text: str,
    client: ModelClient,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> TextAnalysis:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    prompt = get_prompt_variant(prompt_variant)
    logger.info("Classifying input with prompt %s", prompt_variant)
    response = client.generate(
        prompt.system_prompt,
        build_user_prompt(text, prompt_variant),
        response_schema=TextClassification,
    )
    classification = parse_model_response(
        response, TextClassification, "Классификация"
    )
    logger.info("Classification validated: %s", classification.category)

    # Only a validated category may select the answer instructions.
    answer_prompt = build_answer_system_prompt(classification.category)
    logger.info("Generating answer for category %s", classification.category)
    response = client.generate(
        answer_prompt,
        build_answer_user_prompt(text, classification),
        response_schema=GeneratedAnswer,
    )
    answer = parse_model_response(
        response, GeneratedAnswer, "Генерация ответа"
    )

    result = TextAnalysis(**classification.model_dump(), **answer.model_dump())
    logger.info("Final result validated")
    return result
