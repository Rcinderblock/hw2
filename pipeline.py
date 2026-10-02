"""Extract meaning -> classify -> build fields -> answer -> self-check."""

import json
import logging
import re
from typing import Protocol, TypeVar

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    DEFAULT_PROMPT_VARIANT,
    SELF_CHECK_SYSTEM_PROMPT,
    build_answer_repair_user_prompt,
    build_answer_system_prompt,
    build_answer_user_prompt,
    build_classification_user_prompt,
    build_fallback_system_prompt,
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
from utils import decode_json

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
MAX_MODEL_RESPONSE_CHARS = 12000


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
        self.partial_result: dict = {}


class PipelineAPIError(OpenAIError):
    """Expose the failed stage and the already validated data."""

    def __init__(self, message: str, partial_result: dict) -> None:
        super().__init__(message)
        self.partial_result = partial_result.copy()


def describe_schema_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors(include_url=False, include_input=False):
        field = ".".join(str(part) for part in error["loc"]) or "ответ"
        context = error.get("ctx", {})
        explanations = {
            "missing": "обязательное поле отсутствует",
            "string_type": "ожидалась строка",
            "list_type": "ожидался список",
            "bool_type": "ожидалось логическое значение true или false",
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
    if len(response) > MAX_MODEL_RESPONSE_CHARS:
        raise InvalidModelResponse(
            f"{stage}: ответ превышает {MAX_MODEL_RESPONSE_CHARS} символов",
            response,
        )
    if not response.strip():
        raise InvalidModelResponse(
            f"{stage}: модель вернула пустой ответ", response
        )

    try:
        payload = decode_json(response)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponse(
            f"{stage}: Некорректный JSON: строка {exc.lineno}, "
            f"столбец {exc.colno}",
            response,
        ) from exc
    except ValueError as exc:
        raise InvalidModelResponse(
            f"{stage}: Некорректный JSON: недопустимые значения "
            "или повторяющиеся ключи",
            response,
        ) from exc
    except RecursionError as exc:
        raise InvalidModelResponse(
            f"{stage}: Некорректный JSON: слишком глубокая вложенность",
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


def request_step(
    client: ModelClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[ResponseModel],
    stage: str,
    partial_result: dict,
    source_text: str,
) -> ResponseModel:
    for attempt in range(2):
        try:
            response = client.generate(
                system_prompt, user_prompt, response_schema=schema
            )
        except OpenAIError as exc:
            logger.error("%s: %s", stage, exc)
            raise PipelineAPIError(f"{stage}: {exc}", partial_result) from exc
        try:
            result = parse_model_response(response, schema, stage)
            check_output_context(result, source_text, response, stage)
            if attempt:
                logger.info("%s: запасной промпт восстановил формат", stage)
            return result
        except InvalidModelResponse as exc:
            logger.warning("%s", exc)
            if attempt:
                exc.partial_result = partial_result.copy()
                logger.error("%s: запасной промпт не помог", stage)
                raise
            logger.info("%s: повтор с запасным промптом", stage)
            system_prompt = build_fallback_system_prompt(
                system_prompt, str(exc)
            )
    raise AssertionError("Unreachable: every step returns or raises")


def check_output_context(
    result: BaseModel, source_text: str, response: str, stage: str
) -> None:
    russian_letters = len(re.findall(r"[А-Яа-яЁё]", source_text))
    latin_letters = len(re.findall(r"[A-Za-z]", source_text))
    if russian_letters >= 10 and russian_letters > latin_letters:
        for field in ("summary", "final_answer"):
            value = getattr(result, field, "")
            if len(re.findall(r"[A-Za-z]", value)) > 30 and not re.search(
                r"[А-Яа-яЁё]", value
            ):
                raise InvalidModelResponse(
                    f"{stage}: поле {field} должно быть на языке исходника, "
                    "по-русски",
                    response,
                )
    answer = getattr(result, "final_answer", "")
    # The assistant gives advice; it cannot act as the service operator.
    if re.search(
        r"\b(мы|просим|попросим|передадим|отменим|вернём|повысим)\b",
        answer,
        re.IGNORECASE,
    ):
        raise InvalidModelResponse(
            f"{stage}: советуй действие пользователю, не отвечай от имени "
            "сервиса; не используй первое лицо множественного числа",
            response,
        )


def process_text(
    text: str,
    client: ModelClient,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> TextAnalysis:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    prompt = get_prompt_variant(prompt_variant)
    logger.info("Шаг 1/5: извлечение смысла, промпт %s", prompt_variant)
    meaning = request_step(
        client,
        prompt.system_prompt,
        build_user_prompt(text, prompt_variant),
        MeaningExtraction,
        "Извлечение смысла",
        {},
        text,
    )

    logger.info("Шаг 2/5: классификация по извлечённому смыслу")
    classification = request_step(
        client,
        CLASSIFICATION_SYSTEM_PROMPT,
        build_classification_user_prompt(text, meaning),
        RequestClassification,
        "Классификация",
        meaning.model_dump(),
        text,
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
    answer = request_step(
        client,
        answer_prompt,
        build_answer_user_prompt(text, fields),
        GeneratedAnswer,
        "Генерация ответа",
        fields.model_dump(),
        text,
    )

    logger.info("Шаг 5/5: проверка результата по исходному тексту")
    self_check = request_step(
        client,
        SELF_CHECK_SYSTEM_PROMPT,
        build_self_check_user_prompt(text, fields, answer),
        SelfCheckResult,
        "Проверка результата",
        {**fields.model_dump(), **answer.model_dump()},
        text,
    )
    if not self_check.passed:
        logger.warning("Проверка отклонила ответ: %s", self_check.model_dump())
        logger.info("Исправление ответа по замечаниям; максимум один повтор")
        answer = request_step(
            client,
            answer_prompt,
            build_answer_repair_user_prompt(
                text, fields, answer, self_check.model_dump()
            ),
            GeneratedAnswer,
            "Исправление ответа",
            {
                **fields.model_dump(),
                **answer.model_dump(),
                "self_check": self_check.model_dump(),
            },
            text,
        )
        logger.info("Повторная проверка исправленного ответа")
        self_check = request_step(
            client,
            SELF_CHECK_SYSTEM_PROMPT,
            build_self_check_user_prompt(text, fields, answer),
            SelfCheckResult,
            "Повторная проверка результата",
            {**fields.model_dump(), **answer.model_dump()},
            text,
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
