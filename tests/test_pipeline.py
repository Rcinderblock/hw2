"""Checks for data flowing through all five stages and rejected responses."""

import json
import unittest
from argparse import Namespace

from helpers import (
    ScriptedClient,
    check_payload,
    meaning_payload,
    result_payload,
    success_replies,
)
from openai import OpenAIError

from main import load_inputs
from pipeline import InvalidModelResponse, process_text
from schemas import (
    GeneratedAnswer,
    MeaningExtraction,
    RequestClassification,
    SelfCheckResult,
)

STAGES = (
    "Извлечение смысла",
    "Классификация",
    "Генерация ответа",
    "Проверка результата",
)


class PipelineTests(unittest.TestCase):
    def test_four_model_calls_produce_one_checked_result(self) -> None:
        client = ScriptedClient(success_replies())
        result = process_text("Нужно подготовить план.", client)
        self.assertEqual(result.model_dump(), result_payload())
        self.assertEqual(
            [schema for _, _, schema in client.calls],
            [
                MeaningExtraction,
                RequestClassification,
                GeneratedAnswer,
                SelfCheckResult,
            ],
        )

    def test_each_next_step_uses_previous_results_and_original_source(
        self,
    ) -> None:
        source = 'Текст с "кавычками"\nи важным сроком.'
        replies = success_replies(
            summary="Сохранить заметки до утра",
            key_points=[
                "Экспорт сломан",
                "Перезапуск не помог",
                "Срок — утро",
            ],
            category="support",
            intent="Сохранить заметки до утра",
            final_answer="Скопируйте важные заметки вручную до утра.",
        )
        client = ScriptedClient(replies)
        result = process_text(source, client)
        contexts = [
            json.loads(user.split("\n", 1)[1])
            for _, user, _ in client.calls[1:]
        ]
        self.assertTrue(
            all(context["source_text"] == source for context in contexts)
        )
        self.assertEqual(contexts[0]["meaning"], json.loads(replies[0]))
        expected_fields = {**json.loads(replies[0]), **json.loads(replies[1])}
        self.assertEqual(contexts[1]["classification"], expected_fields)
        self.assertEqual(
            contexts[2]["candidate_result"],
            {**expected_fields, **json.loads(replies[2])},
        )
        self.assertEqual(
            result.final_answer, json.loads(replies[2])["final_answer"]
        )

    def test_invalid_json_and_empty_responses_identify_any_failed_stage(
        self,
    ) -> None:
        for response, message in (
            ("not json", "Некорректный JSON"),
            (" ", "пустой ответ"),
        ):
            for index, stage in enumerate(STAGES):
                with self.subTest(response=response, stage=stage):
                    client = ScriptedClient(
                        [*success_replies()[:index], response, response]
                    )
                    with self.assertRaises(InvalidModelResponse) as caught:
                        process_text("Текст", client)
                    self.assertIn(stage, str(caught.exception))
                    self.assertIn(message, str(caught.exception))
                    self.assertEqual(caught.exception.response, response)
                    self.assertEqual(len(client.calls), index + 2)

    def test_wrong_number_of_points_stops_before_classification(self) -> None:
        response = json.dumps(meaning_payload(key_points=["Один"]))
        client = ScriptedClient([response, response])
        with self.assertRaisesRegex(InvalidModelResponse, "key_points"):
            process_text("Текст", client)
        self.assertEqual(len(client.calls), 2)

    def test_empty_input_does_not_call_model(self) -> None:
        client = ScriptedClient([])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            process_text("   ", client)
        self.assertEqual(client.calls, [])

    def test_ten_samples_complete_the_whole_chain_with_fixture_responses(
        self,
    ) -> None:
        inputs = load_inputs(Namespace(demo=True, file=None, text=None))
        self.assertEqual(len(inputs), 10)
        for _, text in inputs:
            client = ScriptedClient(success_replies())
            result = process_text(text, client)
            self.assertEqual(len(result.key_points), 3)
            self.assertTrue(result.self_check.passed)
            self.assertEqual(len(client.calls), 4)

    def test_each_missing_field_has_a_readable_error(self) -> None:
        valid_replies = success_replies()
        for index, response in enumerate(valid_replies):
            for field in json.loads(response):
                with self.subTest(stage=STAGES[index], field=field):
                    payload = json.loads(response)
                    del payload[field]
                    replies = [
                        *valid_replies[:index],
                        json.dumps(payload),
                        json.dumps(payload),
                    ]
                    with self.assertRaisesRegex(
                        InvalidModelResponse, f"{field}: обязательное поле"
                    ):
                        process_text("Текст", ScriptedClient(replies))

    def test_wrong_types_are_not_silently_converted(self) -> None:
        cases = (
            (0, "summary", 123),
            (0, "key_points", "Один, Два, Три"),
            (1, "category", []),
            (1, "intent", 42),
            (1, "sentiment", None),
            (2, "final_answer", False),
            (3, "passed", "true"),
            (3, "contradictions", "нет"),
            (3, "missing_details", None),
        )
        for index, field, value in cases:
            with self.subTest(stage=STAGES[index], field=field):
                replies = success_replies()
                payload = json.loads(replies[index])
                payload[field] = value
                with self.assertRaisesRegex(InvalidModelResponse, field):
                    process_text(
                        "Текст",
                        ScriptedClient(
                            [
                                *replies[:index],
                                json.dumps(payload),
                                json.dumps(payload),
                            ]
                        ),
                    )

    def test_non_object_json_has_a_schema_error_in_each_model_step(
        self,
    ) -> None:
        for index in range(4):
            with self.subTest(stage=STAGES[index]):
                with self.assertRaisesRegex(
                    InvalidModelResponse, "JSON-объект"
                ):
                    process_text(
                        "Текст",
                        ScriptedClient(
                            [*success_replies()[:index], "[]", "[]"]
                        ),
                    )

    def test_unknown_category_stops_before_answer_generation(self) -> None:
        replies = success_replies(category="unknown")[:2]
        client = ScriptedClient([*replies, replies[-1]])
        with self.assertRaisesRegex(InvalidModelResponse, "category"):
            process_text("Текст", client)
        self.assertEqual(len(client.calls), 3)

    def test_api_error_in_any_model_step_is_propagated(self) -> None:
        for index in range(4):
            with self.subTest(stage=STAGES[index]):
                client = ScriptedClient(
                    [*success_replies()[:index], OpenAIError("Unavailable")]
                )
                with self.assertRaisesRegex(OpenAIError, "Unavailable"):
                    process_text("Текст", client)
                self.assertEqual(len(client.calls), index + 1)

    def test_failed_self_check_preserves_answer_and_issues(self) -> None:
        verdict = check_payload(
            passed=False,
            contradictions=["Ответ утверждает, что возврат уже выполнен."],
            missing_details=["Не учтён срок до утра."],
        )
        client = ScriptedClient(success_replies(self_check=verdict))
        result = process_text("Текст", client)
        self.assertEqual(result.self_check.model_dump(), verdict)
        self.assertEqual(result.final_answer, result_payload()["final_answer"])
        self.assertEqual(len(client.calls), 4)

    def test_logs_show_all_five_stages_and_selected_category(self) -> None:
        client = ScriptedClient(
            success_replies(category="support", intent="Сохранить заметки")
        )
        with self.assertLogs("pipeline", level="INFO") as captured:
            process_text("Текст", client)
        messages = "\n".join(captured.output)
        for step in range(1, 6):
            self.assertIn(f"Шаг {step}/5", messages)
        self.assertIn("Категория: support; цель: Сохранить заметки", messages)
        self.assertIn("Проверка пройдена", messages)


if __name__ == "__main__":
    unittest.main()
