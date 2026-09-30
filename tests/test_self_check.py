"""Validate the reviewer's verdict and its specific issues."""

import unittest

from helpers import check_payload
from pydantic import ValidationError

from schemas import SelfCheckResult


class SelfCheckTests(unittest.TestCase):
    def test_success_requires_no_issues(self) -> None:
        result = SelfCheckResult.model_validate(check_payload())
        self.assertTrue(result.passed)
        for changes in (
            {"contradictions": ["Выдуманный факт"]},
            {"missing_details": ["Потерян срок"]},
            {"passed": False},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValidationError, "passed"):
                    SelfCheckResult.model_validate(check_payload(**changes))

    def test_either_kind_of_issue_can_fail_the_check(self) -> None:
        for field in ("contradictions", "missing_details"):
            with self.subTest(field=field):
                result = SelfCheckResult.model_validate(
                    check_payload(
                        passed=False, **{field: ["  Конкретное замечание  "]}
                    )
                )
                self.assertFalse(result.passed)
                self.assertEqual(
                    getattr(result, field), ["Конкретное замечание"]
                )

    def test_blank_issues_and_incorrect_types_are_rejected(self) -> None:
        for changes in (
            {"passed": "true"},
            {"passed": 1},
            {"contradictions": [42]},
            {"missing_details": "Потерян срок"},
            {"passed": False, "missing_details": [" "]},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    SelfCheckResult.model_validate(check_payload(**changes))


if __name__ == "__main__":
    unittest.main()
