"""The expected structured result of processing one text."""

from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SUMMARY_MAX_CHARS = 250
RESPONSE_MAX_CHARS = 400
INTENT_MAX_CHARS = 160
Category = Literal[
    "support", "feedback", "complaint", "sales", "general_question"
]
Sentiment = Literal["positive", "neutral", "negative"]
CATEGORIES = get_args(Category)
SENTIMENTS = get_args(Sentiment)


class StructuredOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True
    )


class MeaningExtraction(StructuredOutput):
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
    key_points: list[str] = Field(min_length=3, max_length=3)

    @field_validator("key_points")
    @classmethod
    def valid_points(cls, points: list[str]) -> list[str]:
        stripped = [point.strip() for point in points]
        if any(not point for point in stripped):
            raise ValueError("ключевые мысли не должны быть пустыми")
        if len({point.casefold() for point in stripped}) != len(stripped):
            raise ValueError("ключевые мысли не должны повторяться")
        return stripped


class RequestClassification(StructuredOutput):
    category: Category
    intent: str = Field(min_length=1, max_length=INTENT_MAX_CHARS)
    sentiment: Sentiment


class TextClassification(MeaningExtraction):
    category: Category
    intent: str = Field(min_length=1, max_length=INTENT_MAX_CHARS)
    sentiment: Sentiment


class GeneratedAnswer(StructuredOutput):
    final_answer: str = Field(min_length=1, max_length=RESPONSE_MAX_CHARS)


class SelfCheckResult(StructuredOutput):
    passed: bool
    contradictions: list[str]
    missing_details: list[str]

    @field_validator("contradictions", "missing_details")
    @classmethod
    def nonempty_issues(cls, issues: list[str]) -> list[str]:
        stripped = [issue.strip() for issue in issues]
        if any(not issue for issue in stripped):
            raise ValueError("замечания проверки не должны быть пустыми")
        return stripped

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> "SelfCheckResult":
        expected = not (self.contradictions or self.missing_details)
        if self.passed != expected:
            raise ValueError("passed должен соответствовать наличию замечаний")
        return self


class TextAnalysis(TextClassification):
    final_answer: str = Field(min_length=1, max_length=RESPONSE_MAX_CHARS)
    self_check: SelfCheckResult
