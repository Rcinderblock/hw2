"""The offline examples exercise the same chain and format recovery."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import demo
from schemas import CATEGORIES, TextAnalysis


class DemoTests(unittest.TestCase):
    def test_five_prepared_scenarios_finish_and_match_stored_outputs(self):
        report = demo.run_demo()
        self.assertEqual(report["mode"], "offline_fixture")
        self.assertEqual(len(report["results"]), 5)
        self.assertEqual(
            {r["category"] for r in report["results"]}, set(CATEGORIES)
        )
        for case, result in zip(
            demo.load_cases(), report["results"], strict=True
        ):
            actual = {key: result[key] for key in TextAnalysis.model_fields}
            self.assertEqual(actual, case["result"])
            TextAnalysis.model_validate(actual)
            self.assertTrue(result["source_text"].strip())
            self.assertEqual(result["model_calls"], 4)

    def test_four_bad_formats_recover_with_one_additional_call(self):
        for failure in (
            "empty",
            "invalid_json",
            "missing_fields",
            "long_answer",
        ):
            with self.subTest(failure=failure):
                result = demo.run_demo("support", failure)["results"][0]
                self.assertTrue(result["self_check"]["passed"])
                self.assertEqual(result["model_calls"], 5)
                self.assertNotIn("error", result)

    def test_unavailable_checker_preserves_results_for_all_scenarios(self):
        report = demo.run_demo(failure="api_unavailable")
        self.assertEqual(len(report["results"]), 5)
        for result in report["results"]:
            self.assertIn("Проверка результата", result["error"])
            self.assertIn("final_answer", result["partial_result"])
            self.assertNotIn("self_check", result["partial_result"])

    def test_cli_is_runnable_without_creating_a_real_api_client(self):
        for failure, expected_code in (
            (None, 0),
            ("invalid_json", 0),
            ("api_unavailable", 1),
        ):
            with self.subTest(failure=failure):
                argv = ["demo.py", "--scenario", "support"]
                if failure:
                    argv += ["--failure", failure]
                stdout = io.StringIO()
                with (
                    patch("sys.argv", argv),
                    patch("llm_client.OpenAI") as sdk,
                    redirect_stdout(stdout),
                ):
                    self.assertEqual(demo.main(), expected_code)
                sdk.assert_not_called()
                self.assertEqual(
                    json.loads(stdout.getvalue())["mode"], "offline_fixture"
                )


if __name__ == "__main__":
    unittest.main()
