"""Compare three prompts on the same inputs using the actual pipeline."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAIError

from llm_client import LLMClient
from pipeline import InvalidModelResponse, ModelClient, process_text
from prompts import (
    ANSWER_INSTRUCTIONS,
    ANSWER_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_PROMPT,
    FALLBACK_PROMPT,
    PROMPT_VARIANTS,
    SELF_CHECK_SYSTEM_PROMPT,
)

EXAMPLES_DIR = Path(__file__).parent / "sample_inputs"


def compare_prompts(
    inputs: list[tuple[str, str]], client: ModelClient, repeats: int = 1
) -> dict:
    if (
        repeats < 1
        or not inputs
        or any(not text.strip() for _, text in inputs)
    ):
        raise ValueError("Provide nonempty texts and at least one repeat")

    runs = []
    # Interleave variants so every one sees the same inputs in each round.
    for repeat in range(1, repeats + 1):
        for name, text in inputs:
            for variant in PROMPT_VARIANTS:
                run = {"variant": variant, "input": name, "repeat": repeat}
                try:
                    result = process_text(text, client, variant)
                    run.update(
                        status="ok"
                        if result.self_check.passed
                        else "self_check_failed",
                        result=result.model_dump(),
                    )
                except InvalidModelResponse as exc:
                    run.update(
                        status="invalid_response",
                        error=str(exc),
                        raw_response=exc.response,
                    )
                    if exc.partial_result:
                        run["partial_result"] = exc.partial_result
                except OpenAIError as exc:
                    run.update(status="api_error", error=str(exc))
                    if partial := getattr(exc, "partial_result", None):
                        run["partial_result"] = partial
                runs.append(run)

    statistics = {}
    for variant in PROMPT_VARIANTS:
        attempts = [run for run in runs if run["variant"] == variant]
        valid = [
            run["result"]
            for run in attempts
            if run["status"] in ("ok", "self_check_failed")
        ]
        invalid = sum(run["status"] == "invalid_response" for run in attempts)
        api_errors = sum(run["status"] == "api_error" for run in attempts)
        completed = len(valid) + invalid
        statistics[variant] = {
            "attempts": len(attempts),
            "valid": len(valid),
            "invalid_responses": invalid,
            "api_errors": api_errors,
            "self_check_failures": sum(
                run["status"] == "self_check_failed" for run in attempts
            ),
            "format_success_rate": len(valid) / completed
            if completed
            else None,
            "mean_summary_chars": mean(len(r["summary"]) for r in valid)
            if valid
            else None,
            "mean_response_chars": mean(len(r["final_answer"]) for r in valid)
            if valid
            else None,
        }

    complete = not any(s["api_errors"] for s in statistics.values())
    best = []
    if complete:
        best_rate = max(s["format_success_rate"] for s in statistics.values())
        if best_rate > 0:
            best = [
                name
                for name, stats in statistics.items()
                if stats["format_success_rate"] == best_rate
            ]

    return {
        "model": getattr(client, "model", None),
        "prompts": {
            name: asdict(prompt) for name, prompt in PROMPT_VARIANTS.items()
        },
        "answer_system_prompt": ANSWER_SYSTEM_PROMPT,
        "classification_system_prompt": CLASSIFICATION_SYSTEM_PROMPT,
        "self_check_system_prompt": SELF_CHECK_SYSTEM_PROMPT,
        "fallback_prompt": FALLBACK_PROMPT,
        "answer_instructions": ANSWER_INSTRUCTIONS,
        "repeats": repeats,
        "inputs": [{"name": name, "text": text} for name, text in inputs],
        "comparison_complete": complete,
        "statistics": statistics,
        "best_format_variants": best,
        "note": (
            "These metrics measure format compliance. Review the answers for "
            "factual accuracy and usefulness before selecting a best prompt."
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare prompt formulations")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    load_dotenv()
    try:
        inputs = [
            (path.stem, path.read_text(encoding="utf-8"))
            for path in sorted(EXAMPLES_DIR.glob("*.txt"))
        ]
        client = LLMClient()
        report = compare_prompts(inputs, client, args.repeats)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        logging.error("Cannot complete comparison: %s", exc)
        return 1

    return 0 if report["comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
