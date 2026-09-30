"""Checks for both model calls and the boundary around their responses."""

import json
import unittest
from argparse import Namespace

from helpers import (
    ScriptedClient,
    classification_payload,
    result_payload,
    success_replies,
)
from openai import OpenAIError

from main import load_inputs
from pipeline import InvalidModelResponse, process_text
from schemas import GeneratedAnswer, TextClassification


class PipelineTests(unittest.TestCase):
    def test_two_valid_steps_produce_one_result(self) -> None:
        client = ScriptedClient(success_replies())
        result = process_text("Нужно подготовить план.", client)
        self.assertEqual(result.model_dump(), result_payload())
        self.assertEqual(len(client.calls), 2)
        self.assertIs(client.calls[0][2], TextClassification)
        self.assertIs(client.calls[1][2], GeneratedAnswer)

    def test_invalid_json_and_empty_responses_identify_the_stage(self) -> None:
        for response, message in (
            ("not json", "Некорректный JSON"),
            (" ", "пустой ответ"),
        ):
            for stage, prefix in (
                ("Классификация", []),
                ("Генерация ответа", success_replies()[:1]),
            ):
                with self.subTest(response=response, stage=stage):
                    client = ScriptedClient([*prefix, response])
                    with self.assertRaises(InvalidModelResponse) as caught:
                        process_text("Текст", client)
                    self.assertIn(stage, str(caught.exception))
                    self.assertIn(message, str(caught.exception))
                    self.assertEqual(caught.exception.response, response)
                    self.assertEqual(len(client.calls), len(prefix) + 1)

    def test_wrong_number_of_points_stops_before_answer_generation(
        self,
    ) -> None:
        response = json.dumps(classification_payload(key_points=["Один"]))
        client = ScriptedClient([response])
        with self.assertRaisesRegex(InvalidModelResponse, "key_points"):
            process_text("Текст", client)
        self.assertEqual(len(client.calls), 1)

    def test_empty_input_does_not_call_model(self) -> None:
        client = ScriptedClient([])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            process_text("   ", client)
        self.assertEqual(client.calls, [])

    def test_demo_contains_ten_processable_texts(self) -> None:
        inputs = load_inputs(Namespace(demo=True, file=None, text=None))
        self.assertEqual(len(inputs), 10)
        for _, text in inputs:
            client = ScriptedClient(success_replies())
            self.assertEqual(len(process_text(text, client).key_points), 3)
            self.assertEqual(len(client.calls), 2)

    def test_each_missing_field_has_a_readable_error(self) -> None:
        for field in result_payload():
            with self.subTest(field=field):
                if field == "final_answer":
                    replies = [success_replies()[0], "{}"]
                else:
                    payload = classification_payload()
                    del payload[field]
                    replies = [json.dumps(payload)]
                with self.assertRaisesRegex(
                    InvalidModelResponse, f"{field}: обязательное поле"
                ):
                    process_text("Текст", ScriptedClient(replies))

    def test_wrong_types_are_not_silently_converted(self) -> None:
        cases = {
            "summary": 123,
            "category": [],
            "intent": 42,
            "sentiment": None,
            "key_points": "Один, Два, Три",
            "final_answer": False,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                if field == "final_answer":
                    replies = [
                        success_replies()[0],
                        json.dumps({field: value}),
                    ]
                else:
                    replies = [
                        json.dumps(classification_payload(**{field: value}))
                    ]
                with self.assertRaisesRegex(InvalidModelResponse, field):
                    process_text("Текст", ScriptedClient(replies))

    def test_non_object_json_has_a_schema_error_in_both_steps(self) -> None:
        for prefix in ([], success_replies()[:1]):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(
                    InvalidModelResponse, "JSON-объект"
                ):
                    process_text("Текст", ScriptedClient([*prefix, "[]"]))

    def test_unknown_category_stops_before_selecting_instructions(
        self,
    ) -> None:
        client = ScriptedClient(
            [json.dumps(classification_payload(category="unknown"))]
        )
        with self.assertRaisesRegex(InvalidModelResponse, "category"):
            process_text("Текст", client)
        self.assertEqual(len(client.calls), 1)

    def test_api_error_in_either_step_is_propagated(self) -> None:
        for prefix in ([], success_replies()[:1]):
            with self.subTest(prefix=prefix):
                client = ScriptedClient([*prefix, OpenAIError("Unavailable")])
                with self.assertRaisesRegex(OpenAIError, "Unavailable"):
                    process_text("Текст", client)
                self.assertEqual(len(client.calls), len(prefix) + 1)

    def test_key_stages_are_logged(self) -> None:
        with self.assertLogs("pipeline", level="INFO") as captured:
            process_text("Текст", ScriptedClient(success_replies()))
        messages = "\n".join(captured.output)
        self.assertIn("Classifying input", messages)
        self.assertIn("Classification validated", messages)
        self.assertIn("Generating answer for category", messages)
        self.assertIn("Final result validated", messages)


if __name__ == "__main__":
    unittest.main()
