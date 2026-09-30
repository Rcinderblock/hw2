"""Exercise the installed SDK and full chain with a local HTTP transport."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from helpers import result_payload, success_replies
from openai import OpenAI, OpenAIError

from llm_client import LLMClient
from pipeline import process_text
from schemas import MeaningExtraction


def response_body(text: str) -> dict:
    return {
        "id": "resp_local",
        "object": "response",
        "created_at": 0,
        "model": "test-fixture",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "id": "msg_local",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ],
            }
        ],
    }


class SDKFlowTests(unittest.TestCase):
    def test_invalid_server_output_type_is_reported_without_a_traceback(self):
        body = response_body("placeholder")
        body["output"][0]["content"][0]["text"] = 42
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=body)

        with httpx.Client(
            transport=httpx.MockTransport(handle)
        ) as http_client:
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
                patch(
                    "llm_client.OpenAI",
                    side_effect=lambda **kwargs: OpenAI(
                        **kwargs, http_client=http_client
                    ),
                ),
            ):
                client = LLMClient()
                with self.assertRaisesRegex(
                    OpenAIError, "неправильный формат"
                ):
                    client.generate(
                        "Rules", "Source", response_schema=MeaningExtraction
                    )
        self.assertEqual(len(requests), 1)

    def test_http_retry_then_format_fallback_finish_the_full_chain(self):
        replies = iter(["not JSON", *success_replies()])
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "code": "server_is_overloaded",
                            "message": "temporary failure",
                        }
                    },
                )
            return httpx.Response(200, json=response_body(next(replies)))

        with httpx.Client(
            transport=httpx.MockTransport(handle)
        ) as http_client:

            def create_sdk(**kwargs):
                return OpenAI(**kwargs, http_client=http_client)

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}),
                patch("llm_client.OpenAI", side_effect=create_sdk),
                patch("llm_client.time.sleep") as sleep,
            ):
                result = process_text("Исходный текст", LLMClient())
        self.assertEqual(result.model_dump(), result_payload())
        self.assertEqual(len(requests), 6)
        sleep.assert_called_once()
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[1]["input"], requests[2]["input"])
        self.assertIn("Validation problem:", requests[2]["instructions"])
        self.assertEqual(
            [r["text"]["format"]["name"] for r in requests],
            ["MeaningExtraction"] * 3
            + ["RequestClassification", "GeneratedAnswer", "SelfCheckResult"],
        )
        self.assertTrue(all(r["store"] is False for r in requests))


if __name__ == "__main__":
    unittest.main()
