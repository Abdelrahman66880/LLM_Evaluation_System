"""Typed application response models."""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """A generated answer together with the evidence shown to the model."""

    answer: str = Field(description="Answer grounded in the retrieved PDF passages")
    context: str = Field(description="Retrieved passages used to produce the answer")
