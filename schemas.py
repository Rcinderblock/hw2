"""The expected structured result of processing one text."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    key_points: list[str] = Field(min_length=3, max_length=3)
    helpful_response: str

    @field_validator("summary", "helpful_response")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("key_points")
    @classmethod
    def valid_points(cls, points: list[str]) -> list[str]:
        stripped = [point.strip() for point in points]
        if any(not point for point in stripped):
            raise ValueError("key points must not be blank")
        if len({point.casefold() for point in stripped}) != len(stripped):
            raise ValueError("key points must be distinct")
        return stripped
