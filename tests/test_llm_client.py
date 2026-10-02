"""Check each API boundary without making a network request."""

import os
import unittest
from unittest.mock import patch

from llm_client import LLMClient
from schemas import (
    GeneratedAnswer,
    MeaningExtraction,
    RequestClassification,
    SelfCheckResult,
)


class LLMClientTests(unittest.TestCase):
    def test_reasoning_setting_is_optional_and_invalid_values_fail_early(self):
        for value in ("", "none", "low", "high", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-only-key",
                        "OPENAI_REASONING_EFFORT": value,
                    },
                ),
                patch("llm_client.OpenAI") as sdk,
            ):
                if value == "invalid":
                    with self.assertRaisesRegex(
                        ValueError, "OPENAI_REASONING_EFFORT"
                    ):
                        LLMClient()
                    sdk.assert_not_called()
                    continue
                sdk.return_value.responses.create.return_value.output_text = (
                    "{}"
                )
                LLMClient().generate(
                    "Rules", "Text", response_schema=MeaningExtraction
                )
                request = sdk.return_value.responses.create.call_args.kwargs
                self.assertEqual(
                    request.get("reasoning"),
                    {"effort": value} if value else None,
                )

    def test_invalid_temperature_fails_before_creating_a_client(self):
        for value in ("not a number", "nan", "inf", "-0.1", "2.1"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-only-key",
                        "OPENAI_TEMPERATURE": value,
                    },
                ),
                patch("llm_client.OpenAI") as sdk,
            ):
                with self.assertRaisesRegex(ValueError, "OPENAI_TEMPERATURE"):
                    LLMClient()
                sdk.assert_not_called()

    def test_optional_temperature_is_sent_only_when_configured(self):
        for value, expected in (
            ("", None),
            ("0", 0.0),
            ("0.5", 0.5),
            ("2", 2.0),
        ):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-only-key",
                        "OPENAI_TEMPERATURE": value,
                    },
                ),
                patch("llm_client.OpenAI") as sdk,
            ):
                sdk.return_value.responses.create.return_value.output_text = (
                    "{}"
                )
                LLMClient().generate(
                    "Rules", "Text", response_schema=MeaningExtraction
                )
                options = sdk.return_value.responses.create.call_args.kwargs
                self.assertEqual(options.get("temperature"), expected)
                if expected is None:
                    self.assertNotIn("temperature", options)

    def test_each_step_sends_its_own_json_schema(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
            patch("llm_client.OpenAI") as sdk,
        ):
            sdk.return_value.responses.create.return_value.output_text = "{}"
            client = LLMClient()
            for schema, fields in (
                (
                    MeaningExtraction,
                    {"summary", "key_points"},
                ),
                (
                    RequestClassification,
                    {
                        "category",
                        "intent",
                        "sentiment",
                    },
                ),
                (GeneratedAnswer, {"final_answer"}),
                (
                    SelfCheckResult,
                    {"passed", "contradictions", "missing_details"},
                ),
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
