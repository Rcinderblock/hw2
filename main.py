"""Command-line entry point for processing one text or demo texts."""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

from llm_client import LLMClient
from pipeline import process_text
from prompts import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS
from schemas import CATEGORIES, SENTIMENTS

EXAMPLES_DIR = Path(__file__).parent / "sample_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and answer a text")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to process")
    source.add_argument("--file", type=Path, help="UTF-8 text file to process")
    source.add_argument(
        "--demo", action="store_true", help="Process sample texts"
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON results to this file"
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=DEFAULT_PROMPT_VARIANT,
        help="Prompt formulation to use",
    )
    parser.add_argument(
        "--category", choices=CATEGORIES, help="Show only this category"
    )
    parser.add_argument(
        "--sentiment", choices=SENTIMENTS, help="Show only this sentiment"
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.demo:
        return [
            (path.stem, path.read_text(encoding="utf-8"))
            for path in sorted(EXAMPLES_DIR.glob("*.txt"))
        ]
    if args.file:
        return [(args.file.stem, args.file.read_text(encoding="utf-8"))]
    return [("input", args.text)]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    load_dotenv()

    try:
        inputs = load_inputs(args)
        client = LLMClient()
    except (OSError, ValueError) as exc:
        logging.error("Cannot start: %s", exc)
        return 1

    results = []
    for name, text in inputs:
        try:
            logging.info("Processing %s", name)
            analysis = process_text(text, client, args.prompt_variant)
            if (
                analysis.self_check.passed
                and args.category
                and analysis.category != args.category
            ):
                continue
            if (
                analysis.self_check.passed
                and args.sentiment
                and analysis.sentiment != args.sentiment
            ):
                continue
            results.append({"name": name, **analysis.model_dump()})
        except (OpenAIError, ValueError) as exc:
            logging.error("Failed to process %s: %s", name, exc)
            results.append({"name": name, "error": str(exc)})

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            logging.error("Cannot write output: %s", exc)
            return 1

    failed = any(
        "error" in result or not result["self_check"]["passed"]
        for result in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
