"""Run the real pipeline with five explicit fixtures, without an API key."""

import argparse
import json
import logging
from pathlib import Path

from openai import OpenAIError
from pydantic import BaseModel

from pipeline import process_text
from schemas import (
    GeneratedAnswer,
    MeaningExtraction,
    RequestClassification,
    SelfCheckResult,
)

PROJECT_DIR = Path(__file__).parent
FAILURES = (
    "empty",
    "invalid_json",
    "missing_fields",
    "long_answer",
    "api_unavailable",
)


def load_cases() -> list[dict]:
    return json.loads(
        (PROJECT_DIR / "examples" / "demo_cases.json").read_text(
            encoding="utf-8"
        )
    )


class FixtureClient:
    """Return prepared data; do not infer a category or generate an answer."""

    def __init__(self, result: dict, failure: str | None = None) -> None:
        self.result = result
        self.failure = failure
        self.injected = False
        self.calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str:
        self.calls += 1
        target = {
            "empty": MeaningExtraction,
            "invalid_json": MeaningExtraction,
            "missing_fields": RequestClassification,
            "long_answer": GeneratedAnswer,
            "api_unavailable": SelfCheckResult,
        }.get(self.failure)
        if not self.injected and response_schema is target:
            self.injected = True
            if self.failure == "empty":
                return ""
            if self.failure == "invalid_json":
                return "Ответ текстом вместо JSON"
            if self.failure == "missing_fields":
                return '{"category": "support", "sentiment": "neutral"}'
            if self.failure == "long_answer":
                return json.dumps({"final_answer": "x" * 401})
            raise OpenAIError(
                "API недоступно после исчерпания попыток (имитация)"
            )
        payload = (
            self.result["self_check"]
            if response_schema is SelfCheckResult
            else {
                key: self.result[key] for key in response_schema.model_fields
            }
        )
        return json.dumps(payload, ensure_ascii=False)


def run_demo(scenario: str | None = None, failure: str | None = None) -> dict:
    cases = load_cases()
    results = []
    for case in cases:
        if scenario and case["name"] != scenario:
            continue
        text = (PROJECT_DIR / "sample_inputs" / case["source_file"]).read_text(
            encoding="utf-8"
        )
        client = FixtureClient(case["result"], failure)
        record = {"name": case["name"], "source_text": text.strip()}
        try:
            record.update(process_text(text, client).model_dump())
        except (OpenAIError, ValueError) as exc:
            record["error"] = str(exc)
            if partial := getattr(exc, "partial_result", None):
                record["partial_result"] = partial
        record["model_calls"] = client.calls
        results.append(record)
    return {"mode": "offline_fixture", "failure": failure, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline demo with prepared model replies"
    )
    parser.add_argument(
        "--scenario", choices=[case["name"] for case in load_cases()]
    )
    parser.add_argument("--failure", choices=FAILURES)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    logging.info("Демонстрация с заданными ответами: API не вызывается")
    report = run_demo(args.scenario, args.failure)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = any(
        "error" in result or not result["self_check"]["passed"]
        for result in report["results"]
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
