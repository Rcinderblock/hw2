"""Comparison correctness, without treating fixtures as real model evidence."""

import json
import unittest
from collections import Counter

from openai import OpenAIError

from compare_prompts import compare_prompts
from prompts import PROMPT_VARIANTS, build_user_prompt

VALID_RESPONSE = json.dumps(
    {
        "summary": "Кратко",
        "key_points": ["Один", "Два", "Три"],
        "helpful_response": "Ответ",
    }
)
INPUTS = [
    ("first", "Первый текст"),
    ("second", "Второй текст"),
    ("third", "Третий текст"),
]


class ScriptedClient:
    model = "test-fixture"

    def __init__(self, replies: list) -> None:
        self.replies = iter(replies)
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ComparisonTests(unittest.TestCase):
    def test_all_variants_receive_identical_inputs_and_repeat_counts(
        self,
    ) -> None:
        client = ScriptedClient([VALID_RESPONSE] * 18)
        report = compare_prompts(INPUTS, client, repeats=2)
        expected = Counter(
            {
                (prompt.system_prompt, build_user_prompt(text, name)): 2
                for name, prompt in PROMPT_VARIANTS.items()
                for _, text in INPUTS
            }
        )
        self.assertEqual(Counter(client.calls), expected)
        self.assertEqual(len(report["runs"]), 18)
        self.assertEqual(report["model"], "test-fixture")
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(report["best_format_variants"], list(PROMPT_VARIANTS))

    def test_invalid_outputs_are_counted_and_preserved(self) -> None:
        client = ScriptedClient(
            [VALID_RESPONSE, "not json", VALID_RESPONSE] * 3
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
            [OpenAIError("Unavailable"), VALID_RESPONSE, VALID_RESPONSE] * 3
        )
        report = compare_prompts(INPUTS, client, repeats=1)
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(report["best_format_variants"], [])
        self.assertEqual(report["statistics"]["baseline"]["api_errors"], 3)
        self.assertIsNone(
            report["statistics"]["baseline"]["format_success_rate"]
        )

    def test_no_winner_when_every_model_output_is_invalid(self) -> None:
        report = compare_prompts(
            INPUTS, ScriptedClient(["bad"] * 9), repeats=1
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


if __name__ == "__main__":
    unittest.main()
