"""The expected structured result of processing one text."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUMMARY_MAX_CHARS = 250
RESPONSE_MAX_CHARS = 400
Category = Literal["question", "request", "feedback", "other"]
Sentiment = Literal["positive", "neutral", "negative"]
CATEGORIES = get_args(Category)
SENTIMENTS = get_args(Sentiment)


class TextAnalysis(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True
    )

    summary: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
    category: Category
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str = Field(min_length=1, max_length=RESPONSE_MAX_CHARS)

    @field_validator("key_points")
    @classmethod
    def valid_points(cls, points: list[str]) -> list[str]:
        stripped = [point.strip() for point in points]
        if any(not point for point in stripped):
            raise ValueError("ключевые мысли не должны быть пустыми")
        if len({point.casefold() for point in stripped}) != len(stripped):
            raise ValueError("ключевые мысли не должны повторяться")
        return stripped
