"""Tests for the boundary between model output and validated result."""

import json
import unittest
from argparse import Namespace

from main import load_inputs
from pipeline import InvalidModelResponse, process_text


def valid_payload() -> dict:
    return {
        "summary": "Кратко",
        "category": "request",
        "sentiment": "neutral",
        "key_points": ["Один", "Два", "Три"],
        "final_answer": "Предлагаю начать с первого шага.",
    }


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


class PipelineTests(unittest.TestCase):
    def test_valid_response_has_three_points(self) -> None:
        client = FakeClient(json.dumps(valid_payload()))
        result = process_text("Нужно подготовить план.", client)
        self.assertEqual(result.summary, "Кратко")
        self.assertEqual(len(result.key_points), 3)
        self.assertEqual(client.calls, 1)

    def test_invalid_json_is_reported(self) -> None:
        with self.assertRaisesRegex(InvalidModelResponse, "Некорректный JSON"):
            process_text("Текст", FakeClient("not json"))

    def test_wrong_number_of_points_is_reported(self) -> None:
        payload = valid_payload()
        payload["key_points"] = ["Один"]
        client = FakeClient(json.dumps(payload))
        with self.assertRaisesRegex(InvalidModelResponse, "key_points"):
            process_text("Текст", client)

    def test_empty_input_does_not_call_model(self) -> None:
        client = FakeClient("{}")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            process_text("   ", client)
        self.assertEqual(client.calls, 0)

    def test_empty_model_response_is_reported(self) -> None:
        with self.assertRaisesRegex(InvalidModelResponse, "пустой ответ"):
            process_text("Текст", FakeClient(" "))

    def test_demo_contains_at_least_five_processable_texts(self) -> None:
        inputs = load_inputs(Namespace(demo=True, file=None, text=None))
        self.assertGreaterEqual(len(inputs), 5)
        for _, text in inputs:
            client = FakeClient(json.dumps(valid_payload()))
            self.assertEqual(len(process_text(text, client).key_points), 3)
            self.assertEqual(client.calls, 1)

    def test_each_missing_field_has_a_readable_error(self) -> None:
        for field in valid_payload():
            with self.subTest(field=field):
                payload = valid_payload()
                del payload[field]
                with self.assertRaisesRegex(
                    InvalidModelResponse, f"{field}: обязательное поле"
                ):
                    process_text("Текст", FakeClient(json.dumps(payload)))

    def test_wrong_types_are_not_silently_converted(self) -> None:
        cases = {
            "summary": 123,
            "category": [],
            "sentiment": None,
            "key_points": "Один, Два, Три",
            "final_answer": False,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                payload = {**valid_payload(), field: value}
                with self.assertRaisesRegex(InvalidModelResponse, field):
                    process_text("Текст", FakeClient(json.dumps(payload)))

    def test_non_object_json_has_a_schema_error(self) -> None:
        with self.assertRaisesRegex(InvalidModelResponse, "JSON-объект"):
            process_text("Текст", FakeClient("[]"))


if __name__ == "__main__":
    unittest.main()
