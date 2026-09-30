"""Explicit fixtures for local checks; these do not classify real text."""

import json

from pydantic import BaseModel


def classification_payload(**changes) -> dict:
    return {
        "summary": "Кратко",
        "category": "general_question",
        "intent": "Составить план действий",
        "sentiment": "neutral",
        "key_points": ["Один", "Два", "Три"],
        **changes,
    }


def result_payload(**changes) -> dict:
    return {
        **classification_payload(),
        "final_answer": "Предлагаю начать с первого шага.",
        **changes,
    }


def success_replies(**changes) -> list[str]:
    answer = changes.pop("final_answer", "Предлагаю начать с первого шага.")
    return [
        json.dumps(classification_payload(**changes), ensure_ascii=False),
        json.dumps({"final_answer": answer}, ensure_ascii=False),
    ]


class ScriptedClient:
    model = "test-fixture"

    def __init__(self, replies: list) -> None:
        self.replies = iter(replies)
        self.calls = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str:
        self.calls.append((system_prompt, user_prompt, response_schema))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
