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


def is_temporary_error(exc: OpenAIError) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        if exc.code in QUOTA_CODES or exc.type == "insufficient_quota":
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
        if exc.code in QUOTA_CODES or exc.type == "insufficient_quota":
            return "API: закончились средства или достигнут лимит расходов"
        if exc.status_code in (401, 403):
            return (
                f"API: доступ отклонён (HTTP {exc.status_code}); "
                "проверьте ключ и права"
            )
        return f"API: ошибка HTTP {exc.status_code}"
    if isinstance(exc, APIConnectionError):
        return "API: не удалось подключиться или истекло время ожидания"
    return "API: запрос не выполнен"


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "Set OPENAI_API_KEY in the environment or .env file"
            )

        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
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
                    max_output_tokens=1200,
                    store=False,
                )
                return response.output_text or ""
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
