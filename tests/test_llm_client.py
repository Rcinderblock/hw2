"""Check each API boundary without making a network request."""

import os
import unittest
from unittest.mock import patch

from llm_client import LLMClient
from schemas import GeneratedAnswer, TextClassification


class LLMClientTests(unittest.TestCase):
    def test_each_step_sends_its_own_json_schema(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch("llm_client.OpenAI") as sdk,
        ):
            sdk.return_value.responses.create.return_value.output_text = "{}"
            client = LLMClient()
            for schema, fields in (
                (
                    TextClassification,
                    {
                        "summary",
                        "category",
                        "intent",
                        "sentiment",
                        "key_points",
                    },
                ),
                (GeneratedAnswer, {"final_answer"}),
            ):
                with self.subTest(schema=schema.__name__):
                    self.assertEqual(
                        client.generate(
                            "Rules", "Text", response_schema=schema
                        ),
                        "{}",
                    )
                    request = (
                        sdk.return_value.responses.create.call_args.kwargs
                    )
                    output_format = request["text"]["format"]
                    self.assertEqual(output_format["type"], "json_schema")
                    self.assertTrue(output_format["strict"])
                    self.assertEqual(
                        set(output_format["schema"]["required"]), fields
                    )
                    self.assertFalse(
                        output_format["schema"]["additionalProperties"]
                    )
                    self.assertEqual(request["instructions"], "Rules")
                    self.assertEqual(request["input"], "Text")
                    self.assertFalse(request["store"])

    def test_missing_key_fails_before_creating_an_api_client(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch("llm_client.OpenAI") as sdk,
        ):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                LLMClient()
            sdk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
