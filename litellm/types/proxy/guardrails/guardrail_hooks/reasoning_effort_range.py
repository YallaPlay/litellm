"""Configuration model for the reasoning-effort range guardrail."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from .base import GuardrailConfigModel

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_RANK = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
}


class ReasoningEffortRangeGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    """Model and allowed reasoning-effort range for one guardrail instance."""

    model: str = Field(
        min_length=1,
        description="Exact public model name this guardrail is allowed to protect.",
    )
    min_effort: ReasoningEffort = Field(
        default="none",
        description="Minimum accepted reasoning effort.",
    )
    max_effort: ReasoningEffort = Field(
        description="Maximum accepted reasoning effort.",
    )
    default_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Effort injected when omitted. Defaults to max_effort.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if EFFORT_RANK[self.min_effort] > EFFORT_RANK[self.max_effort]:
            raise ValueError("min_effort must not exceed max_effort")
        default_effort = self.default_effort or self.max_effort
        if not EFFORT_RANK[self.min_effort] <= EFFORT_RANK[default_effort] <= EFFORT_RANK[self.max_effort]:
            raise ValueError("default_effort must be inside the configured range")
        return self

    @staticmethod
    def ui_friendly_name() -> str:
        return "Reasoning Effort Range"
