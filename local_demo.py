"""Validate sample texts with a real local model, without an API key."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAIError

from local_client import (
    DEFAULT_LOCAL_MODEL,
    SAMPLING_SETTINGS,
    LocalLLMClient,
)
from pipeline import process_text
from utils import load_sample_inputs, validate_output_path, write_json_output

PROJECT_DIR = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real local model")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    samples_dir = PROJECT_DIR / "sample_inputs"
    try:
        inputs = load_sample_inputs(samples_dir)
        if args.samples:
            available = dict(inputs)
            if len(set(args.samples)) != len(args.samples):
                raise ValueError("Sample names must not repeat")
            unknown = set(args.samples) - available.keys()
            if unknown:
                raise ValueError(
                    "Unknown samples: " + ", ".join(sorted(unknown))
                )
            inputs = [(name, available[name]) for name in args.samples]
        validate_output_path(args.output, list(samples_dir.glob("*")))
        client = LocalLLMClient(args.model)
    except (OSError, ValueError) as exc:
        logging.error("Cannot start: %s", exc)
        return 1

    report = {
        "mode": "local_model",
        "model": client.model,
        "seed": 42,
        **SAMPLING_SETTINGS,
        "max_tokens": 4096,
        "enable_thinking": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "pipeline.py",
                "prompts.py",
                "schemas.py",
                "local_client.py",
            )
        },
        "results": [],
    }
    for name, text in inputs:
        record = {"name": name, "source_text": text}
        call_start = len(client.calls)
        try:
            record.update(process_text(text, client).model_dump())
        except (OpenAIError, ValueError) as exc:
            logging.error("Failed to process %s: %s", name, exc)
            record["error"] = str(exc)
            if partial := getattr(exc, "partial_result", None):
                record["partial_result"] = partial
        record["model_calls"] = client.calls[call_start:]
        report["results"].append(record)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        try:
            write_json_output(args.output, rendered)
        except OSError as exc:
            logging.error("Cannot write output: %s", exc)
            return 1
    return int(
        any(
            "error" in r or not r["self_check"]["passed"]
            for r in report["results"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
