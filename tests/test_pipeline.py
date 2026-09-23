"""Tests for the boundary between model output and validated result."""

import unittest
from argparse import Namespace

from main import load_inputs
from pipeline import process_text


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


class PipelineTests(unittest.TestCase):
    def test_valid_response_has_three_points(self) -> None:
        client = FakeClient(
            '{"summary":"Кратко", "key_points":["Один", "Два", "Три"], '
            '"helpful_response":"Предлагаю начать с первого шага."}'
        )
        result = process_text("Нужно подготовить план.", client)
        self.assertEqual(result.summary, "Кратко")
        self.assertEqual(len(result.key_points), 3)
        self.assertEqual(client.calls, 1)

    def test_invalid_json_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            process_text("Текст", FakeClient("not json"))

    def test_wrong_number_of_points_is_reported(self) -> None:
        client = FakeClient(
            '{"summary":"Кратко", "key_points":["Один"], '
            '"helpful_response":"Ответ"}'
        )
        with self.assertRaisesRegex(ValueError, "invalid JSON or fields"):
            process_text("Текст", client)

    def test_empty_input_does_not_call_model(self) -> None:
        client = FakeClient("{}")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            process_text("   ", client)
        self.assertEqual(client.calls, 0)

    def test_empty_model_response_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty response"):
            process_text("Текст", FakeClient(" "))

    def test_demo_contains_three_processable_texts(self) -> None:
        inputs = load_inputs(Namespace(demo=True, file=None, text=None))
        self.assertEqual(len(inputs), 3)
        for _, text in inputs:
            client = FakeClient(
                '{"summary":"Кратко", "key_points":["Один", "Два", "Три"], '
                '"helpful_response":"Ответ"}'
            )
            self.assertEqual(len(process_text(text, client).key_points), 3)
            self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
