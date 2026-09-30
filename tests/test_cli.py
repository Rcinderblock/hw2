"""Local checks of batch output and filtering with scripted model replies."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from helpers import ScriptedClient, result_payload, success_replies
from openai import OpenAIError

import compare_prompts
import main


class CLITests(unittest.TestCase):
    def test_demo_uses_selected_prompt_and_saves_ten_results(self) -> None:
        client = ScriptedClient(success_replies() * 10)
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
            self.assertEqual(len(results), 10)
            self.assertTrue(
                all(set(result_payload()) <= set(r) for r in results)
            )
            self.assertEqual(results, json.loads(stdout.getvalue()))
            self.assertTrue(
                all(
                    "Example input:" in system
                    for system, _, schema in client.calls
                    if schema.__name__ == "TextClassification"
                )
            )
            self.assertEqual(len(client.calls), 20)

    def test_comparison_cli_saves_thirty_attempts(self) -> None:
        client = ScriptedClient(success_replies() * 30)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            argv = ["compare_prompts.py", "--output", str(output)]
            with (
                patch.object(
                    compare_prompts, "LLMClient", return_value=client
                ),
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(compare_prompts.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 30)
            self.assertEqual(report["model"], "test-fixture")
            self.assertEqual(len(client.calls), 60)

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
            ("general_question", "neutral"),
            ("general_question", "neutral"),
            ("support", "negative"),
            ("feedback", "positive"),
            ("feedback", "negative"),
            ("support", "neutral"),
            ("complaint", "negative"),
            ("complaint", "negative"),
            ("sales", "neutral"),
            ("sales", "positive"),
        )
        replies = [
            reply
            for cat, sent in categories
            for reply in success_replies(category=cat, sentiment=sent)
        ]
        cases = (
            (
                ["--category", "feedback"],
                ["04_positive_feedback", "05_negative_feedback"],
            ),
            (
                ["--sentiment", "negative"],
                [
                    "03_support",
                    "05_negative_feedback",
                    "07_refund",
                    "08_billing",
                ],
            ),
            (
                ["--category", "feedback", "--sentiment", "negative"],
                ["05_negative_feedback"],
            ),
            (
                ["--category", "general_question", "--sentiment", "positive"],
                [],
            ),
        )
        for args, expected_names in cases:
            with self.subTest(args=args):
                code, results, client = self.run_main(replies, args)
                self.assertEqual(code, 0)
                self.assertEqual([r["name"] for r in results], expected_names)
                self.assertEqual(len(client.calls), 20)

    def test_broken_json_does_not_stop_remaining_examples(self) -> None:
        code, results, _ = self.run_main(
            ["{bad json", *success_replies() * 9], []
        )
        self.assertEqual(code, 1)
        self.assertIn("Некорректный JSON", results[0]["error"])
        self.assertEqual(len(results), 10)
        self.assertTrue(all("final_answer" in r for r in results[1:]))

    def test_missing_fields_and_wrong_types_get_readable_errors(self) -> None:
        classification = json.loads(success_replies()[0])
        del classification["category"]
        classification["intent"] = 42
        code, results, _ = self.run_main(
            [json.dumps(classification), *success_replies() * 9], []
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "category: обязательное поле отсутствует", results[0]["error"]
        )
        self.assertIn("intent: ожидалась строка", results[0]["error"])

    def test_second_step_failure_does_not_stop_remaining_examples(
        self,
    ) -> None:
        for failure in (
            json.dumps({"final_answer": 42}),
            OpenAIError("Unavailable"),
        ):
            with self.subTest(failure=failure):
                code, results, _ = self.run_main(
                    [success_replies()[0], failure, *success_replies() * 9],
                    [],
                )
                self.assertEqual(code, 1)
                self.assertIn("error", results[0])
                self.assertEqual(len(results), 10)
                self.assertTrue(all("final_answer" in r for r in results[1:]))

    def test_api_errors_remain_visible_when_results_are_filtered(self) -> None:
        code, results, _ = self.run_main(
            [OpenAIError("Unavailable"), *success_replies() * 9],
            ["--category", "feedback"],
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["error"], "Unavailable")


if __name__ == "__main__":
    unittest.main()
