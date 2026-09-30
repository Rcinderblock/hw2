"""Check the API boundary without making a network request."""

import os
import unittest
from unittest.mock import patch

from llm_client import LLMClient


class LLMClientTests(unittest.TestCase):
    def test_json_schema_is_sent_to_the_model(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch("llm_client.OpenAI") as sdk,
        ):
            sdk.return_value.responses.create.return_value.output_text = "{}"
            self.assertEqual(LLMClient().generate("Rules", "Text"), "{}")
            request = sdk.return_value.responses.create.call_args.kwargs
        output_format = request["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertEqual(
            set(output_format["schema"]["required"]),
            {"summary", "category", "sentiment", "key_points", "final_answer"},
        )
        self.assertFalse(output_format["schema"]["additionalProperties"])

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
