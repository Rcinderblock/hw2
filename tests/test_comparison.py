"""Comparison checks; fixtures do not establish real model quality."""

import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest.mock import patch

from helpers import ScriptedClient, check_payload, success_replies
from openai import OpenAIError

from compare_prompts import compare_prompts, main
from prompts import PROMPT_VARIANTS, build_user_prompt
from schemas import MeaningExtraction

INPUTS = [
    ("first", "Первый текст"),
    ("second", "Второй текст"),
    ("third", "Третий текст"),
]


class ComparisonTests(unittest.TestCase):
    def test_all_variants_receive_identical_inputs_and_repeat_counts(
        self,
    ) -> None:
        client = ScriptedClient(success_replies() * 18)
        report = compare_prompts(INPUTS, client, repeats=2)
        expected = Counter(
            {
                (prompt.system_prompt, build_user_prompt(text, name)): 2
                for name, prompt in PROMPT_VARIANTS.items()
                for _, text in INPUTS
            }
        )
        actual = Counter(
            (system, user)
            for system, user, schema in client.calls
            if schema is MeaningExtraction
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(client.calls), 72)
        self.assertEqual(len(report["runs"]), 18)
        self.assertEqual(report["model"], "test-fixture")
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(report["best_format_variants"], list(PROMPT_VARIANTS))

    def test_invalid_outputs_are_counted_and_preserved(self) -> None:
        client = ScriptedClient(
            [*success_replies(), "not json", "not json", *success_replies()]
            * 3
        )
        report = compare_prompts(INPUTS, client, repeats=1)
        self.assertEqual(
            report["statistics"]["explicit"]["invalid_responses"], 3
        )
        self.assertEqual(
            report["statistics"]["explicit"]["format_success_rate"], 0
        )
        self.assertEqual(
            report["best_format_variants"], ["baseline", "example"]
        )
        invalid = [
            run
            for run in report["runs"]
            if run["status"] == "invalid_response"
        ]
        self.assertEqual(invalid[0]["raw_response"], "not json")

    def test_api_errors_prevent_selecting_a_format_winner(self) -> None:
        client = ScriptedClient(
            [OpenAIError("Unavailable"), *success_replies() * 2] * 3
        )
        report = compare_prompts(INPUTS, client, repeats=1)
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(report["best_format_variants"], [])
        self.assertEqual(report["statistics"]["baseline"]["api_errors"], 3)
        self.assertIsNone(
            report["statistics"]["baseline"]["format_success_rate"]
        )

    def test_later_step_errors_are_preserved_and_comparison_continues(
        self,
    ) -> None:
        for index, failure, status in (
            (2, "not json", "invalid_response"),
            (3, OpenAIError("Unavailable"), "api_error"),
        ):
            with self.subTest(status=status):
                replies = [
                    *success_replies()[:index],
                    failure,
                    *([] if isinstance(failure, Exception) else [failure]),
                    *success_replies() * 8,
                ]
                report = compare_prompts(INPUTS, ScriptedClient(replies))
                self.assertEqual(report["runs"][0]["status"], status)
                self.assertEqual(
                    sum(r["status"] == "ok" for r in report["runs"]), 8
                )

    def test_self_check_failure_is_distinct_from_broken_format(self) -> None:
        verdict = check_payload(
            passed=False,
            contradictions=["Ответ противоречит исходному тексту."],
        )
        replies = [
            *success_replies(self_check=verdict),
            *success_replies(self_check=verdict)[2:],
            *success_replies() * 8,
        ]
        report = compare_prompts(INPUTS, ScriptedClient(replies))
        self.assertEqual(report["runs"][0]["status"], "self_check_failed")
        self.assertEqual(report["runs"][0]["result"]["self_check"], verdict)
        stats = report["statistics"]["baseline"]
        self.assertEqual(stats["self_check_failures"], 1)
        self.assertEqual(stats["format_success_rate"], 1)
        self.assertEqual(stats["invalid_responses"], 0)

    def test_no_winner_when_every_model_output_is_invalid(self) -> None:
        report = compare_prompts(
            INPUTS, ScriptedClient(["bad"] * 18), repeats=1
        )
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(report["best_format_variants"], [])

    def test_invalid_experiment_inputs_do_not_call_model(self) -> None:
        for inputs, repeats in ((INPUTS, 0), ([], 1), ([("empty", " ")], 1)):
            with self.subTest(inputs=inputs, repeats=repeats):
                client = ScriptedClient([])
                with self.assertRaises(ValueError):
                    compare_prompts(inputs, client, repeats)
                self.assertEqual(client.calls, [])

    def test_cli_compares_only_requested_samples(self) -> None:
        client = ScriptedClient(success_replies() * 6)
        output = io.StringIO()
        with (
            patch(
                "sys.argv",
                ["compare_prompts.py", "--samples", "third", "first"],
            ),
            patch("compare_prompts.load_sample_inputs", return_value=INPUTS),
            patch("compare_prompts.LLMClient", return_value=client),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(
            [item["name"] for item in report["inputs"]], ["third", "first"]
        )
        self.assertEqual(len(client.calls), 24)

    def test_cli_rejects_bad_samples_before_api_calls(self) -> None:
        for names in (["absent"], ["first", "first"]):
            with (
                self.subTest(names=names),
                patch("sys.argv", ["compare_prompts.py", "--samples", *names]),
                patch(
                    "compare_prompts.load_sample_inputs", return_value=INPUTS
                ),
                patch("compare_prompts.LLMClient") as create_client,
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)
                create_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
