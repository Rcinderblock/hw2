"""Deliberately fail the API and model output without network access."""

import json
import os
import unittest
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from helpers import ScriptedClient, success_replies
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
)

from llm_client import QUOTA_CODES, LLMClient
from pipeline import InvalidModelResponse, PipelineAPIError, process_text
from schemas import MeaningExtraction


def status_error(status: int, **kwargs) -> APIStatusError:
    return APIStatusError(
        "Do not print this server message: secret-placeholder",
        response=httpx.Response(
            status,
            request=httpx.Request("POST", "https://api.example/test"),
            headers=kwargs.get("headers", {}),
        ),
        body={"code": kwargs.get("code"), "type": kwargs.get("error_type")},
    )


class APIReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        sdk_patch = patch("llm_client.OpenAI")
        self.sdk = sdk_patch.start()
        self.addCleanup(sdk_patch.stop)
        sleep_patch = patch("llm_client.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        jitter_patch = patch("llm_client.random.uniform", return_value=0)
        jitter_patch.start()
        self.addCleanup(jitter_patch.stop)
        self.client = LLMClient()
        self.create = self.sdk.return_value.responses.create

    def generate(self) -> str:
        return self.client.generate(
            "Rules", "Text", response_schema=MeaningExtraction
        )

    def test_temporary_errors_retry_and_recover_with_increasing_delay(self):
        request = httpx.Request("POST", "https://api.example/test")
        errors = [
            APIConnectionError(request=request),
            APITimeoutError(request=request),
            *(status_error(s) for s in (408, 409, 429, 500, 503)),
        ]
        for error in errors:
            with self.subTest(
                error=type(error), status=getattr(error, "status_code", None)
            ):
                self.create.reset_mock()
                self.sleep.reset_mock()
                self.create.side_effect = [
                    error,
                    error,
                    SimpleNamespace(output_text="{}"),
                ]
                self.assertEqual(self.generate(), "{}")
                self.assertEqual(self.create.call_count, 3)
                self.assertEqual(
                    [call.args[0] for call in self.sleep.call_args_list],
                    [1, 2],
                )
        self.assertEqual(self.sdk.call_args.kwargs["max_retries"], 0)

    def test_permanent_errors_do_not_retry_or_expose_server_message(self):
        errors = [
            *(status_error(s) for s in (400, 401, 403, 404, 422)),
            *(status_error(429, code=code) for code in QUOTA_CODES),
            status_error(429, error_type="insufficient_quota"),
            status_error(503, headers={"x-should-retry": "false"}),
        ]
        for error in errors:
            with self.subTest(error=error):
                self.create.reset_mock()
                self.sleep.reset_mock()
                self.create.side_effect = error
                with self.assertLogs("llm_client", "ERROR") as captured:
                    with self.assertRaises(OpenAIError) as caught:
                        self.generate()
                self.assertEqual(self.create.call_count, 1)
                self.sleep.assert_not_called()
                self.assertNotIn("secret-placeholder", str(caught.exception))
                self.assertNotIn(
                    "secret-placeholder", "\n".join(captured.output)
                )

    def test_attempt_budget_stops_an_unavailable_api(self):
        self.create.side_effect = status_error(503)
        with self.assertRaisesRegex(OpenAIError, "попыток: 3"):
            self.generate()
        self.assertEqual(self.create.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_retry_after_is_respected_in_seconds_milliseconds_and_date(self):
        for headers in (
            {"retry-after": "3"},
            {"retry-after-ms": "3000"},
            {
                "retry-after": format_datetime(
                    datetime.fromtimestamp(1003, timezone.utc)
                )
            },
        ):
            with (
                self.subTest(headers=headers),
                patch("llm_client.time.time", return_value=1000),
            ):
                self.sleep.reset_mock()
                self.create.side_effect = [
                    status_error(429, headers=headers),
                    SimpleNamespace(output_text="{}"),
                ]
                self.generate()
                self.sleep.assert_called_once_with(3)

    def test_long_server_delay_stops_instead_of_retrying_too_soon(self):
        self.create.side_effect = status_error(
            429, headers={"retry-after": "60"}
        )
        with self.assertRaisesRegex(OpenAIError, "запустите позже"):
            self.generate()
        self.assertEqual(self.create.call_count, 1)
        self.sleep.assert_not_called()

    def test_invalid_retry_after_uses_local_backoff(self):
        for value in ("nonsense", "nan", "inf", "-1"):
            with self.subTest(value=value):
                self.sleep.reset_mock()
                self.create.side_effect = [
                    status_error(503, headers={"retry-after": value}),
                    SimpleNamespace(output_text="{}"),
                ]
                self.generate()
                self.sleep.assert_called_once_with(1)

    def test_non_string_api_error_codes_do_not_break_error_handling(self):
        for code in ({"unexpected": "object"}, ["unknown"], 42):
            with self.subTest(code=code):
                self.create.reset_mock()
                self.create.side_effect = [
                    status_error(503, code=code),
                    SimpleNamespace(output_text="{}"),
                ]
                self.assertEqual(self.generate(), "{}")
                self.assertEqual(self.create.call_count, 2)


class FormatRecoveryTests(unittest.TestCase):
    def test_language_and_service_role_errors_use_fallback(self):
        cases = (
            (
                0,
                json.dumps(
                    {
                        "summary": "Customer needs help with the application.",
                        "key_points": ["A", "B", "C"],
                    }
                ),
            ),
            (
                2,
                json.dumps(
                    {"final_answer": "Мы передадим ваш отзыв команде."}
                ),
            ),
        )
        for index, bad in cases:
            with self.subTest(index=index):
                replies = success_replies()
                client = ScriptedClient(
                    [*replies[:index], bad, *replies[index:]]
                )
                result = process_text(
                    "Приложение не сохраняет данные, нужен совет.", client
                )
                self.assertTrue(result.self_check.passed)
                self.assertEqual(len(client.calls), 5)
                self.assertIn(
                    "Validation problem:", client.calls[index + 1][0]
                )

    def test_english_input_and_technical_names_are_not_language_errors(self):
        for source, summary in (
            (
                "I need help with the application.",
                "The customer needs help with the application.",
            ),
            (
                "Мне нужна помощь с Python и настройкой приложения.",
                "Автор просит помочь с Python.",
            ),
        ):
            with self.subTest(source=source):
                replies = success_replies()
                extraction = json.loads(replies[0])
                extraction["summary"] = summary
                replies[0] = json.dumps(extraction)
                client = ScriptedClient(replies)
                process_text(source, client)
                self.assertEqual(len(client.calls), 4)

    def test_ambiguous_or_nonstandard_json_uses_fallback(self):
        cases = (
            '{"summary":"first","summary":"second",'
            '"key_points":["A","B","C"]}',
            '{"summary":NaN,"key_points":["A","B","C"]}',
            '{"summary":Infinity,"key_points":["A","B","C"]}',
            '{"summary":-Infinity,"key_points":["A","B","C"]}',
            '{"summary":' + "9" * 5000 + ',"key_points":["A","B","C"]}',
        )
        for response in cases:
            with self.subTest(response=response[:60]):
                client = ScriptedClient([response, *success_replies()])
                result = process_text("Текст", client)
                self.assertTrue(result.self_check.passed)
                self.assertEqual(len(client.calls), 5)

    def test_invalid_output_recovers_without_losing_source_or_stage_data(self):
        for index in range(4):
            for bad in ("", "not JSON", "{}", "[]"):
                with self.subTest(index=index, bad=bad):
                    replies = success_replies()
                    client = ScriptedClient(
                        [*replies[:index], bad, *replies[index:]]
                    )
                    with self.assertLogs("pipeline", "INFO") as captured:
                        result = process_text("Срок — утро", client)
                    self.assertTrue(result.self_check.passed)
                    self.assertEqual(len(client.calls), 5)
                    first, fallback = client.calls[index : index + 2]
                    self.assertEqual(first[1:], fallback[1:])
                    self.assertTrue(fallback[0].startswith(first[0]))
                    self.assertIn("Validation problem:", fallback[0])
                    self.assertIn(
                        "восстановил формат", "\n".join(captured.output)
                    )

    def test_overlong_fields_and_raw_output_use_the_same_fallback(self):
        for index, bad in (
            (0, "x" * 12001),
            (0, "[" * 2000 + "0" + "]" * 2000),
            (
                0,
                json.dumps(
                    {"summary": "x" * 251, "key_points": ["A", "B", "C"]}
                ),
            ),
            (2, json.dumps({"final_answer": "x" * 401})),
            (2, json.dumps({"final_answer": "x" * 400})),
        ):
            with self.subTest(index=index):
                replies = success_replies()
                client = ScriptedClient(
                    [*replies[:index], bad, *replies[index:]]
                )
                result = process_text("Текст", client)
                self.assertTrue(result.self_check.passed)
                self.assertEqual(len(client.calls), 5)

    def test_exhausted_fallback_preserves_only_validated_partial_fields(self):
        replies = success_replies()
        client = ScriptedClient([*replies[:2], "bad", "bad"])
        with self.assertRaises(InvalidModelResponse) as caught:
            process_text("Текст", client)
        partial = caught.exception.partial_result
        self.assertEqual(
            partial, {**json.loads(replies[0]), **json.loads(replies[1])}
        )
        self.assertNotIn("final_answer", partial)

    def test_unavailable_checker_keeps_answer_but_never_claims_it_passed(self):
        replies = success_replies()
        client = ScriptedClient([*replies[:3], OpenAIError("Unavailable")])
        with self.assertRaises(PipelineAPIError) as caught:
            process_text("Текст", client)
        partial = caught.exception.partial_result
        self.assertEqual(
            partial["final_answer"], json.loads(replies[2])["final_answer"]
        )
        self.assertNotIn("self_check", partial)
        self.assertIn("Проверка результата", str(caught.exception))

    def test_failure_during_content_repair_never_reuses_an_old_verdict(self):
        replies = success_replies()
        rejected = {
            "passed": False,
            "contradictions": ["Возврат ещё не выполнен."],
            "missing_details": [],
        }
        for failure_at in (4, 5):
            with self.subTest(failure_at=failure_at):
                sequence = [*replies[:3], json.dumps(rejected)]
                if failure_at == 5:
                    sequence.append(
                        json.dumps(
                            {"final_answer": "Запросите возврат в поддержке."}
                        )
                    )
                client = ScriptedClient(
                    [*sequence, OpenAIError("Unavailable")]
                )
                with self.assertRaises(PipelineAPIError) as caught:
                    process_text("Хочу вернуть деньги.", client)
                partial = caught.exception.partial_result
                if failure_at == 4:
                    self.assertEqual(partial["self_check"], rejected)
                else:
                    self.assertEqual(
                        partial["final_answer"],
                        "Запросите возврат в поддержке.",
                    )
                    self.assertNotIn("self_check", partial)
                self.assertEqual(len(client.calls), failure_at + 1)


if __name__ == "__main__":
    unittest.main()
