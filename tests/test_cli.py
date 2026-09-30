"""End-to-end local checks of CLI output using a clearly marked test client."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openai import OpenAIError
from test_comparison import VALID_RESPONSE, ScriptedClient

import compare_prompts
import main


class CLITests(unittest.TestCase):
    def test_demo_uses_selected_prompt_and_saves_five_results(self) -> None:
        client = ScriptedClient([VALID_RESPONSE] * 5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "demo.json"
            argv = [
                "main.py",
                "--demo",
                "--prompt-variant",
                "example",
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with (
                patch.object(main, "LLMClient", return_value=client),
                patch("sys.argv", argv),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main.main(), 0)
            results = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(results), 5)
            self.assertTrue(
                all(set(json.loads(VALID_RESPONSE)) <= set(r) for r in results)
            )
            self.assertEqual(results, json.loads(stdout.getvalue()))
            self.assertTrue(
                all("Example input:" in system for system, _ in client.calls)
            )

    def test_comparison_cli_saves_all_fifteen_attempts(self) -> None:
        client = ScriptedClient([VALID_RESPONSE] * 15)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            argv = [
                "compare_prompts.py",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
            with (
                patch.object(
                    compare_prompts, "LLMClient", return_value=client
                ),
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(compare_prompts.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 15)
            self.assertEqual(report["model"], "test-fixture")

    def run_main(self, replies: list, extra_args: list[str]) -> tuple:
        client = ScriptedClient(replies)
        stdout = io.StringIO()
        with (
            patch.object(main, "LLMClient", return_value=client),
            patch("sys.argv", ["main.py", "--demo", *extra_args]),
            redirect_stdout(stdout),
        ):
            exit_code = main.main()
        return exit_code, json.loads(stdout.getvalue()), client

    def test_category_and_sentiment_control_result_selection(self) -> None:
        categories = (
            ("request", "neutral"),
            ("question", "neutral"),
            ("request", "negative"),
            ("feedback", "positive"),
            ("feedback", "negative"),
        )
        replies = [
            json.dumps(
                {
                    **json.loads(VALID_RESPONSE),
                    "category": cat,
                    "sentiment": sent,
                }
            )
            for cat, sent in categories
        ]
        cases = (
            (
                ["--category", "feedback"],
                ["04_positive_feedback", "05_negative_feedback"],
            ),
            (
                ["--sentiment", "negative"],
                ["03_support", "05_negative_feedback"],
            ),
            (
                ["--category", "feedback", "--sentiment", "negative"],
                ["05_negative_feedback"],
            ),
            (["--category", "question", "--sentiment", "positive"], []),
        )
        for args, expected_names in cases:
            with self.subTest(args=args):
                code, results, client = self.run_main(replies, args)
                self.assertEqual(code, 0)
                self.assertEqual([r["name"] for r in results], expected_names)
                self.assertEqual(len(client.calls), 5)

    def test_broken_json_does_not_stop_remaining_examples(self) -> None:
        code, results, _ = self.run_main(
            ["{bad json", *([VALID_RESPONSE] * 4)], []
        )
        self.assertEqual(code, 1)
        self.assertIn("Некорректный JSON", results[0]["error"])
        self.assertEqual(len(results), 5)
        self.assertTrue(all("final_answer" in r for r in results[1:]))

    def test_missing_fields_and_wrong_types_get_readable_errors(self) -> None:
        payload = json.loads(VALID_RESPONSE)
        del payload["category"]
        payload["final_answer"] = 42
        code, results, _ = self.run_main(
            [json.dumps(payload), *([VALID_RESPONSE] * 4)], []
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "category: обязательное поле отсутствует", results[0]["error"]
        )
        self.assertIn("final_answer: ожидалась строка", results[0]["error"])

    def test_api_errors_remain_visible_when_results_are_filtered(self) -> None:
        code, results, _ = self.run_main(
            [OpenAIError("Unavailable"), *([VALID_RESPONSE] * 4)],
            ["--category", "feedback"],
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["error"], "Unavailable")


if __name__ == "__main__":
    unittest.main()
