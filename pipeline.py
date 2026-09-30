"""Extract meaning -> classify -> build fields -> answer -> self-check."""

import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    DEFAULT_PROMPT_VARIANT,
    SELF_CHECK_SYSTEM_PROMPT,
    build_answer_system_prompt,
    build_answer_user_prompt,
    build_classification_user_prompt,
    build_self_check_user_prompt,
    build_user_prompt,
    get_prompt_variant,
)
from schemas import (
    GeneratedAnswer,
    MeaningExtraction,
    RequestClassification,
    SelfCheckResult,
    TextAnalysis,
    TextClassification,
)

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
    logger.info("Шаг 1/5: извлечение смысла, промпт %s", prompt_variant)
    response = client.generate(
        prompt.system_prompt,
        build_user_prompt(text, prompt_variant),
        response_schema=MeaningExtraction,
    )
    meaning = parse_model_response(
        response, MeaningExtraction, "Извлечение смысла"
    )

    logger.info("Шаг 2/5: классификация по извлечённому смыслу")
    response = client.generate(
        CLASSIFICATION_SYSTEM_PROMPT,
        build_classification_user_prompt(text, meaning),
        response_schema=RequestClassification,
    )
    classification = parse_model_response(
        response, RequestClassification, "Классификация"
    )
    logger.info(
        "Категория: %s; цель: %s",
        classification.category,
        classification.intent,
    )

    logger.info("Шаг 3/5: сборка и проверка структурированных полей")
    fields = TextClassification(
        **meaning.model_dump(), **classification.model_dump()
    )

    # Only a validated category may select the answer instructions.
    answer_prompt = build_answer_system_prompt(fields.category)
    logger.info("Шаг 4/5: генерация ответа, категория %s", fields.category)
    response = client.generate(
        answer_prompt,
        build_answer_user_prompt(text, fields),
        response_schema=GeneratedAnswer,
    )
    answer = parse_model_response(
        response, GeneratedAnswer, "Генерация ответа"
    )

    logger.info("Шаг 5/5: проверка результата по исходному тексту")
    response = client.generate(
        SELF_CHECK_SYSTEM_PROMPT,
        build_self_check_user_prompt(text, fields, answer),
        response_schema=SelfCheckResult,
    )
    self_check = parse_model_response(
        response, SelfCheckResult, "Проверка результата"
    )
    result = TextAnalysis(
        **fields.model_dump(), **answer.model_dump(), self_check=self_check
    )
    if self_check.passed:
        logger.info("Проверка пройдена; итоговый результат готов")
    else:
        logger.warning(
            "Проверка не пройдена: противоречия=%s; потерянные детали=%s",
            self_check.contradictions,
            self_check.missing_details,
        )
    return result
