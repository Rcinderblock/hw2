"""The only module that communicates with the model API."""

import os

from openai import OpenAI
from pydantic import BaseModel


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "Set OPENAI_API_KEY in the environment or .env file"
            )

        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                }
            },
            store=False,
        )
        return response.output_text or ""
