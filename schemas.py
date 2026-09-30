"""The expected structured result of processing one text."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUMMARY_MAX_CHARS = 250
RESPONSE_MAX_CHARS = 400


class TextAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
    key_points: list[str] = Field(min_length=3, max_length=3)
    helpful_response: str = Field(min_length=1, max_length=RESPONSE_MAX_CHARS)

    @field_validator("key_points")
    @classmethod
    def valid_points(cls, points: list[str]) -> list[str]:
        stripped = [point.strip() for point in points]
        if any(not point for point in stripped):
            raise ValueError("key points must not be blank")
        if len({point.casefold() for point in stripped}) != len(stripped):
            raise ValueError("key points must be distinct")
        return stripped
