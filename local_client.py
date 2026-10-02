"""Optional local model client for the same pipeline on Apple Silicon."""

import json
import time

from httpx import HTTPError
from openai import OpenAIError
from pydantic import BaseModel

from prompts import LOCAL_SYSTEM_TEMPLATE

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3.5-9B-4bit"
SAMPLING_SETTINGS = {
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "presence_context_size": 20,
}


class LocalLLMClient:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import (
                make_logits_processors,
                make_sampler,
            )
        except ImportError as exc:
            raise ValueError(
                "Install requirements-local.txt on an Apple Silicon Mac"
            ) from exc

        self.model = model
        self.calls: list[dict] = []
        try:
            self.loaded_model, self.tokenizer = load(model)
        except (OSError, ValueError, RuntimeError, HTTPError) as exc:
            raise ValueError(
                f"Cannot load the local model ({type(exc).__name__})"
            ) from exc
        mx.random.seed(42)
        self.generate_text = generate
        self.sampler = make_sampler(
            temp=SAMPLING_SETTINGS["temperature"],
            top_p=SAMPLING_SETTINGS["top_p"],
            top_k=SAMPLING_SETTINGS["top_k"],
        )
        self.logits_processors = make_logits_processors(
            presence_penalty=SAMPLING_SETTINGS["presence_penalty"],
            presence_context_size=SAMPLING_SETTINGS["presence_context_size"],
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: type[BaseModel],
    ) -> str:
        schema = response_schema.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": LOCAL_SYSTEM_TEMPLATE.format(
                    instructions=system_prompt,
                    schema=json.dumps(schema, ensure_ascii=False),
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        started = time.monotonic()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            response = self.generate_text(
                self.loaded_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=4096,
                sampler=self.sampler,
                logits_processors=self.logits_processors,
                verbose=False,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise OpenAIError(
                f"Local model generation failed ({type(exc).__name__})"
            ) from exc
        final_text = response.strip()
        self.calls.append(
            {
                "step_schema": response_schema.__name__,
                "system_prompt": messages[0]["content"],
                "user_prompt": user_prompt,
                "schema": schema,
                "raw_generation": response,
                "raw_output": final_text,
                "seconds": time.monotonic() - started,
            }
        )
        return final_text
