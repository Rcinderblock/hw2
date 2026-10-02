"""Local checks of batch output and filtering with scripted model replies."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from helpers import (
    ScriptedClient,
    check_payload,
    result_payload,
    success_replies,
)
from openai import OpenAIError

import compare_prompts
import main


class CLITests(unittest.TestCase):
    def test_empty_sample_directory_fails_before_creating_a_client(self):
        with tempfile.TemporaryDirectory() as directory:
            for module, argv in (
                (main, ["main.py", "--demo"]),
                (compare_prompts, ["compare_prompts.py"]),
            ):
                with (
                    self.subTest(module=module.__name__),
                    patch.object(module, "EXAMPLES_DIR", Path(directory)),
                    patch.object(module, "LLMClient") as client,
                    patch("sys.argv", argv),
                    self.assertLogs(level="ERROR") as captured,
                ):
                    self.assertEqual(module.main(), 1)
                    client.assert_not_called()
                    self.assertIn(
                        "Нет входных текстов", "\n".join(captured.output)
                    )

    def test_output_cannot_overwrite_the_source_or_demo_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("Важный исходный текст", encoding="utf-8")
            labels = root / "expected_categories.json"
            labels.write_text('{"input": "support"}', encoding="utf-8")
            for module, argv, target in (
                (main, ["main.py", "--file", str(source)], source),
                (main, ["main.py", "--demo"], labels),
                (compare_prompts, ["compare_prompts.py"], labels),
            ):
                with self.subTest(module=module.__name__, target=target.name):
                    original = target.read_bytes()
                    with (
                        patch.object(module, "EXAMPLES_DIR", root),
                        patch.object(module, "LLMClient") as client,
                        patch("sys.argv", [*argv, "--output", str(target)]),
                        self.assertLogs(level="ERROR"),
                    ):
                        self.assertEqual(module.main(), 1)
                        client.assert_not_called()
                    self.assertEqual(target.read_bytes(), original)

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
                    if schema.__name__ == "MeaningExtraction"
                )
            )
            self.assertEqual(len(client.calls), 40)

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
            self.assertEqual(len(client.calls), 120)

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
                self.assertEqual(len(client.calls), 40)

    def test_broken_json_does_not_stop_remaining_examples(self) -> None:
        code, results, _ = self.run_main(
            ["{bad json", "{bad json", *success_replies() * 9], []
        )
        self.assertEqual(code, 1)
        self.assertIn("Некорректный JSON", results[0]["error"])
        self.assertEqual(len(results), 10)
        self.assertTrue(all("final_answer" in r for r in results[1:]))

    def test_missing_fields_and_wrong_types_get_readable_errors(self) -> None:
        classification = json.loads(success_replies()[1])
        del classification["category"]
        classification["intent"] = 42
        code, results, _ = self.run_main(
            [
                success_replies()[0],
                json.dumps(classification),
                json.dumps(classification),
                *success_replies() * 9,
            ],
            [],
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "category: обязательное поле отсутствует", results[0]["error"]
        )
        self.assertIn("intent: ожидалась строка", results[0]["error"])

    def test_later_step_failure_does_not_stop_remaining_examples(
        self,
    ) -> None:
        for index, failure in (
            (2, json.dumps({"final_answer": 42})),
            (3, OpenAIError("Unavailable")),
        ):
            with self.subTest(failure=failure):
                code, results, _ = self.run_main(
                    [
                        *success_replies()[:index],
                        failure,
                        *([] if isinstance(failure, Exception) else [failure]),
                        *success_replies() * 9,
                    ],
                    [],
                )
                self.assertEqual(code, 1)
                self.assertIn("error", results[0])
                partial = results[0]["partial_result"]
                self.assertIn("summary", partial)
                self.assertIn("category", partial)
                self.assertNotIn("self_check", partial)
                self.assertEqual("final_answer" in partial, index == 3)
                self.assertEqual(len(results), 10)
                self.assertTrue(all("final_answer" in r for r in results[1:]))

    def test_failed_self_check_survives_filters_and_batch_continues(
        self,
    ) -> None:
        verdict = check_payload(
            passed=False, missing_details=["Не учтён срок до утра."]
        )
        replies = [
            *success_replies(self_check=verdict),
            *success_replies(self_check=verdict)[2:],
            *success_replies(category="feedback") * 9,
        ]
        code, results, client = self.run_main(
            replies, ["--category", "feedback", "--sentiment", "positive"]
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["self_check"], verdict)
        self.assertIn("final_answer", results[0])
        self.assertEqual(len(client.calls), 42)

    def test_api_errors_remain_visible_when_results_are_filtered(self) -> None:
        code, results, _ = self.run_main(
            [OpenAIError("Unavailable"), *success_replies() * 9],
            ["--category", "feedback"],
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(results), 1)
        self.assertIn("Unavailable", results[0]["error"])


if __name__ == "__main__":
    unittest.main()
