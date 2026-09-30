"""Checks for selecting and building the meaning extraction instructions."""

import json
import unittest

from helpers import ScriptedClient, success_replies

from pipeline import process_text
from prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    EXAMPLE_OUTPUT,
    PROMPT_VARIANTS,
)
from schemas import (
    INTENT_MAX_CHARS,
    RESPONSE_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    MeaningExtraction,
)


class PromptTests(unittest.TestCase):
    def test_each_variant_reaches_the_client_with_the_source_text(
        self,
    ) -> None:
        text = 'Текст с "кавычками", {скобками}\nи новой строкой.'
        client = ScriptedClient(success_replies() * 3)
        for name, prompt in PROMPT_VARIANTS.items():
            with self.subTest(variant=name):
                process_text(text, client, name)
                system, user, _ = client.calls[-4]
                self.assertEqual(system, prompt.system_prompt)
                self.assertIn(json.dumps(text, ensure_ascii=False), user)
                self.assertIn(str(SUMMARY_MAX_CHARS), system)
                self.assertIn(
                    str(INTENT_MAX_CHARS), CLASSIFICATION_SYSTEM_PROMPT
                )
                self.assertIn(str(RESPONSE_MAX_CHARS), client.calls[-2][0])
        self.assertEqual(len(client.calls), 12)

    def test_unknown_variant_does_not_call_model(self) -> None:
        client = ScriptedClient([])
        with self.assertRaisesRegex(ValueError, "Unknown prompt"):
            process_text("Текст", client, "missing")
        self.assertEqual(client.calls, [])

    def test_example_output_satisfies_the_meaning_schema(self) -> None:
        result = MeaningExtraction.model_validate_json(EXAMPLE_OUTPUT)
        self.assertEqual(len(result.key_points), 3)


if __name__ == "__main__":
    unittest.main()
