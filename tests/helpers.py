"""Explicit fixtures; these do not establish model quality."""

import json

from pydantic import BaseModel


def meaning_payload(**changes) -> dict:
    return {
        "summary": "Кратко",
        "key_points": ["Один", "Два", "Три"],
        **changes,
    }


def classification_payload(**changes) -> dict:
    return {
        "category": "general_question",
        "intent": "Составить план действий",
        "sentiment": "neutral",
        **changes,
    }


def check_payload(**changes) -> dict:
    return {
        "passed": True,
        "contradictions": [],
        "missing_details": [],
        **changes,
    }


def result_payload(**changes) -> dict:
    return {
        **meaning_payload(),
        **classification_payload(),
        "final_answer": "Предлагаю начать с первого шага.",
        "self_check": check_payload(),
        **changes,
    }


def success_replies(**changes) -> list[str]:
    answer = changes.pop("final_answer", "Предлагаю начать с первого шага.")
    self_check = changes.pop("self_check", check_payload())
    meaning_changes = {
        field: changes.pop(field)
        for field in ("summary", "key_points")
        if field in changes
    }
    return [
        json.dumps(meaning_payload(**meaning_changes), ensure_ascii=False),
        json.dumps(classification_payload(**changes), ensure_ascii=False),
        json.dumps({"final_answer": answer}, ensure_ascii=False),
        json.dumps(self_check, ensure_ascii=False),
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
