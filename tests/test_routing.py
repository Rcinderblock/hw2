"""Check category selection in code using explicit classification fixtures."""

import json
import unittest
from collections import Counter

from helpers import ScriptedClient, success_replies

from main import EXAMPLES_DIR
from pipeline import process_text
from schemas import CATEGORIES, GeneratedAnswer, RequestClassification


class RoutingTests(unittest.TestCase):
    def test_category_changes_instructions_for_the_same_source(self) -> None:
        source = "Помогите разобраться с моим обращением."
        expected_instructions = {
            "support": "numbered troubleshooting steps",
            "feedback": "Thank the author for feedback",
            "complaint": "empathetic response",
            "sales": "purchase-oriented answer",
            "general_question": "Answer the question directly",
        }
        classification_prompts = set()
        answer_prompts = set()
        for category, instruction in expected_instructions.items():
            with self.subTest(category=category):
                client = ScriptedClient(success_replies(category=category))
                result = process_text(source, client)
                first_system, _, first_schema = client.calls[1]
                second_system, second_user, second_schema = client.calls[2]
                classification_prompts.add(first_system)
                answer_prompts.add(second_system)
                self.assertIs(first_schema, RequestClassification)
                self.assertIs(second_schema, GeneratedAnswer)
                self.assertIn(instruction, second_system)
                context = json.loads(second_user.split("\n", 1)[1])
                self.assertEqual(context["source_text"], source)
                self.assertEqual(
                    context["classification"]["category"], category
                )
                self.assertEqual(result.category, category)
                self.assertEqual(
                    result.final_answer, "Предлагаю начать с первого шага."
                )
        self.assertEqual(len(classification_prompts), 1)
        self.assertEqual(len(answer_prompts), 5)

    def test_intent_is_passed_to_answer_generation(self) -> None:
        intent = "Вернуть лишнее списание за подписку"
        client = ScriptedClient(
            success_replies(category="complaint", intent=intent)
        )
        result = process_text("Прошу разобраться с оплатой.", client)
        context = json.loads(client.calls[2][1].split("\n", 1)[1])
        self.assertEqual(context["classification"]["intent"], intent)
        self.assertEqual(result.intent, intent)

    def test_sentiment_does_not_override_the_selected_category(self) -> None:
        answer_prompts = set()
        for sentiment in ("positive", "neutral", "negative"):
            client = ScriptedClient(
                success_replies(category="feedback", sentiment=sentiment)
            )
            process_text("Отзыв о редакторе", client)
            answer_prompts.add(client.calls[2][0])
        self.assertEqual(len(answer_prompts), 1)

    def test_ten_examples_cover_each_category_twice(self) -> None:
        expected = json.loads(
            (EXAMPLES_DIR / "expected_categories.json").read_text(
                encoding="utf-8"
            )
        )
        samples = {path.stem: path for path in EXAMPLES_DIR.glob("*.txt")}
        self.assertEqual(set(samples), set(expected))
        self.assertEqual(
            Counter(expected.values()),
            Counter({category: 2 for category in CATEGORIES}),
        )
        # Labels supply fixture responses; this is not model accuracy evidence.
        for name, category in expected.items():
            with self.subTest(sample=name):
                source = samples[name].read_text(encoding="utf-8")
                client = ScriptedClient(success_replies(category=category))
                result = process_text(source, client)
                context = json.loads(client.calls[2][1].split("\n", 1)[1])
                self.assertEqual(context["source_text"], source)
                self.assertEqual(result.category, category)
                self.assertEqual(len(client.calls), 4)


if __name__ == "__main__":
    unittest.main()
