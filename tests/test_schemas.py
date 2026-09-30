"""Output limits must hold even when the model ignores its instructions."""

import unittest

from helpers import result_payload
from pydantic import ValidationError

from schemas import (
    INTENT_MAX_CHARS,
    RESPONSE_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    GeneratedAnswer,
    TextAnalysis,
)


def valid_result() -> dict:
    return result_payload()


class SchemaTests(unittest.TestCase):
    def test_exact_length_limits_are_accepted(self) -> None:
        payload = valid_result()
        payload.update(
            summary="я" * SUMMARY_MAX_CHARS,
            intent="я" * INTENT_MAX_CHARS,
            final_answer="я" * RESPONSE_MAX_CHARS,
        )
        result = TextAnalysis.model_validate(payload)
        self.assertEqual(len(result.summary), SUMMARY_MAX_CHARS)
        self.assertEqual(len(result.intent), INTENT_MAX_CHARS)
        self.assertEqual(len(result.final_answer), RESPONSE_MAX_CHARS)

    def test_overlong_outputs_are_rejected_without_truncation(self) -> None:
        for field, limit in (
            ("summary", SUMMARY_MAX_CHARS),
            ("intent", INTENT_MAX_CHARS),
            ("final_answer", RESPONSE_MAX_CHARS),
        ):
            with self.subTest(field=field):
                payload = valid_result()
                payload[field] = "я" * (limit + 1)
                with self.assertRaises(ValidationError):
                    TextAnalysis.model_validate(payload)

    def test_whitespace_is_removed_before_checking_length(self) -> None:
        payload = valid_result()
        payload["summary"] = "  " + "я" * SUMMARY_MAX_CHARS + "  "
        self.assertEqual(
            len(TextAnalysis.model_validate(payload).summary),
            SUMMARY_MAX_CHARS,
        )

    def test_blank_fields_and_duplicate_points_are_rejected(self) -> None:
        cases = (
            {"summary": "   "},
            {"intent": "\n"},
            {"final_answer": "\n"},
            {"key_points": ["Один", "Два", " "]},
            {"key_points": ["Один", " один ", "Три"]},
            {"key_points": ["Один", "Два", "Три", "Четыре"]},
            {"extra": "unexpected"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                payload = {**valid_result(), **changes}
                with self.assertRaises(ValidationError):
                    TextAnalysis.model_validate(payload)

    def test_unknown_category_and_sentiment_are_rejected(self) -> None:
        for field in ("category", "sentiment"):
            with self.subTest(field=field):
                payload = {**valid_result(), field: "unknown"}
                with self.assertRaises(ValidationError):
                    TextAnalysis.model_validate(payload)

    def test_non_string_key_points_are_rejected(self) -> None:
        payload = {**valid_result(), "key_points": ["Один", 2, "Три"]}
        with self.assertRaises(ValidationError):
            TextAnalysis.model_validate(payload)

    def test_answer_cannot_replace_the_classification(self) -> None:
        with self.assertRaises(ValidationError):
            GeneratedAnswer.model_validate(
                {"final_answer": "Ответ", "category": "sales"}
            )


if __name__ == "__main__":
    unittest.main()
