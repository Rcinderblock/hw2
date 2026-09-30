"""Checks for selecting and building the application instructions."""

import json
import unittest

from pipeline import process_text
from prompts import EXAMPLE_OUTPUT, PROMPT_VARIANTS
from schemas import RESPONSE_MAX_CHARS, SUMMARY_MAX_CHARS, TextAnalysis


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(
            {
                "summary": "Кратко",
                "key_points": ["Один", "Два", "Три"],
                "helpful_response": "Ответ",
            }
        )


class PromptTests(unittest.TestCase):
    def test_each_variant_reaches_the_client_with_the_source_text(
        self,
    ) -> None:
        text = 'Текст с "кавычками", {скобками}\nи новой строкой.'
        client = RecordingClient()
        for name, prompt in PROMPT_VARIANTS.items():
            with self.subTest(variant=name):
                process_text(text, client, name)
                system, user = client.calls[-1]
                self.assertEqual(system, prompt.system_prompt)
                self.assertIn(json.dumps(text, ensure_ascii=False), user)
                self.assertIn(str(SUMMARY_MAX_CHARS), system)
                self.assertIn(str(RESPONSE_MAX_CHARS), system)
        self.assertEqual(len(client.calls), 3)

    def test_unknown_variant_does_not_call_model(self) -> None:
        client = RecordingClient()
        with self.assertRaisesRegex(ValueError, "Unknown prompt"):
            process_text("Текст", client, "missing")
        self.assertEqual(client.calls, [])

    def test_example_output_satisfies_the_same_schema(self) -> None:
        result = TextAnalysis.model_validate_json(EXAMPLE_OUTPUT)
        self.assertEqual(len(result.key_points), 3)


if __name__ == "__main__":
    unittest.main()
