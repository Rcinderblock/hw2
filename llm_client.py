"""The only module that communicates with the model API."""

import logging
import math
import os
import random
import time
from email.utils import parsedate_to_datetime

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from pydantic import BaseModel

logger = logging.getLogger(__name__)
API_ATTEMPTS = 3
MAX_RETRY_DELAY = 5.0
QUOTA_CODES = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}


class InvalidAPIResponse(OpenAIError):
    """The server response does not match the SDK response format."""


def is_quota_error(exc: APIStatusError) -> bool:
    # Compatible servers may put an object or list in the code field.
    return (
        isinstance(exc.code, str) and exc.code in QUOTA_CODES
    ) or exc.type == "insufficient_quota"


def is_temporary_error(exc: OpenAIError) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        if is_quota_error(exc):
            return False
        if exc.response.headers.get("x-should-retry") == "false":
            return False
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


def retry_delay(exc: OpenAIError, attempt: int) -> float:
    if isinstance(exc, APIStatusError):
        headers = exc.response.headers
        for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
            value = headers.get(name)
            if value is None:
                continue
            try:
                delay = float(value) / divisor
            except ValueError:
                if name != "retry-after":
                    continue
                try:
                    delay = (
                        parsedate_to_datetime(value).timestamp() - time.time()
                    )
                except (ValueError, TypeError, OverflowError):
                    continue
            if math.isfinite(delay) and delay >= 0:
                return delay
    return min(2 ** (attempt - 1) + random.uniform(0, 0.25), MAX_RETRY_DELAY)


def describe_api_error(exc: OpenAIError) -> str:
    # Server messages can echo input or credentials; only report safe metadata.
    if isinstance(exc, APIStatusError):
        if is_quota_error(exc):
            return "API: закончились средства или достигнут лимит расходов"
        if exc.status_code in (401, 403):
            return (
                f"API: доступ отклонён (HTTP {exc.status_code}); "
                "проверьте ключ и права"
            )
        return f"API: ошибка HTTP {exc.status_code}"
    if isinstance(exc, APIConnectionError):
        return "API: не удалось подключиться или истекло время ожидания"
    if isinstance(exc, InvalidAPIResponse):
        return "API: ответ сервера имеет неправильный формат"
    return "API: запрос не выполнен"


class LLMClient:
    def __init__(self) -> None:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "Set OPENAI_API_KEY in the environment or .env file"
            )

        self.model = (os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
        temperature = (os.getenv("OPENAI_TEMPERATURE") or "").strip()
        self.generation_options = {}
        if temperature:
            try:
                value = float(temperature)
            except ValueError as exc:
                raise ValueError(
                    "OPENAI_TEMPERATURE must be between 0 and 2"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 2:
                raise ValueError("OPENAI_TEMPERATURE must be between 0 and 2")
            self.generation_options["temperature"] = value
        reasoning_effort = (os.getenv("OPENAI_REASONING_EFFORT") or "").strip()
        if reasoning_effort:
            if reasoning_effort not in {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
            }:
                raise ValueError(
                    "OPENAI_REASONING_EFFORT must be none, minimal, low, "
                    "medium, or high"
                )
            self.generation_options["reasoning"] = {"effort": reasoning_effort}
        base_url = os.getenv("OPENAI_BASE_URL") or None
        # Own the retry budget instead of multiplying it by SDK retries.
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str:
        for attempt in range(1, API_ATTEMPTS + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": response_schema.__name__,
                            "strict": True,
                            "schema": response_schema.model_json_schema(),
                        }
                    },
                    # This budget includes reasoning, not only the final JSON.
                    max_output_tokens=4096,
                    store=False,
                    **self.generation_options,
                )
                try:
                    text = response.output_text
                except (AttributeError, TypeError) as exc:
                    raise InvalidAPIResponse() from exc
                if text is not None and not isinstance(text, str):
                    raise InvalidAPIResponse()
                return text or ""
            except OpenAIError as exc:
                message = describe_api_error(exc)
                if not is_temporary_error(exc) or attempt == API_ATTEMPTS:
                    logger.error("%s; попыток: %s", message, attempt)
                    raise OpenAIError(
                        f"{message}; попыток: {attempt}"
                    ) from exc
                delay = retry_delay(exc, attempt)
                if delay > MAX_RETRY_DELAY:
                    # Do not retry sooner than the server's requested delay.
                    message += (
                        f"; повтор разрешён через {delay:.1f} с, "
                        "запустите позже"
                    )
                    logger.error(message)
                    raise OpenAIError(message) from exc
                logger.warning(
                    "%s; попытка %s/%s, повтор через %.2f с",
                    message,
                    attempt,
                    API_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError(
            "Unreachable: every API attempt returns or raises"
        )
