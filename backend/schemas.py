"""
backend/schemas.py – Pydantic Input Validation Models
SkillGenome X

Validates all incoming request payloads before they reach ML inference.
Returns structured 422 errors on bad input.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# ── Full list of Indian states + UTs (28 states + 8 UTs) ─────────────────────
VALID_STATES = {
    # States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    # Union Territories
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
}

VALID_DOMAINS = {
    "Agriculture", "Agriculture & Allied", "Technology",
    "Manufacturing", "Skilled Trades", "Construction & Skilled Trades",
    "Healthcare", "Education", "Finance", "Business", "Trade",
    "Retail & Sales", "Service Industry", "Logistics & Delivery",
    "Entrepreneurship", "Creative & Media", "Business & Administration",
    "Manufacturing & Operations", "General",
}

VALID_OPPORTUNITY = {"High", "Moderate", "Low"}
VALID_DIGITAL_ACCESS = {"Regular", "Limited", "Occasional"}
VALID_INFRASTRUCTURE = {"High", "Limited", "Minimal"}


class SignalInput(BaseModel):
    """Behavioral signal vector — all values must be 0-100."""
    creation_output:          float = Field(default=50, ge=0, le=100)
    learning_behavior:        float = Field(default=50, ge=0, le=100)
    experience_consistency:   float = Field(default=50, ge=0, le=100)
    economic_activity:        float = Field(default=50, ge=0, le=100)
    innovation_problem_solving: float = Field(default=50, ge=0, le=100)
    collaboration_community:  float = Field(default=50, ge=0, le=100)
    offline_capability:       float = Field(default=50, ge=0, le=100)
    digital_presence:         float = Field(default=50, ge=0, le=100)
    learning_hours:           float = Field(default=10, ge=0, le=168)
    projects:                 int   = Field(default=5,  ge=0, le=500)
    github_repos:             int   = Field(default=0,  ge=0, le=1000)
    hackathons:               int   = Field(default=0,  ge=0, le=100)


class ContextInput(BaseModel):
    """Contextual / demographic inputs."""
    state: str = Field(default="Maharashtra")
    area_type: Literal["Rural", "Urban", "Semi-Urban"] = Field(default="Urban")
    domain: str = Field(default="General")
    opportunity_level: Literal["High", "Moderate", "Low"] = Field(default="Moderate")
    digital_access: Literal["Regular", "Limited", "Occasional"] = Field(default="Regular")
    infrastructure_access: Optional[Literal["High", "Limited", "Minimal"]] = Field(default="High")

    @field_validator("state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        if v not in VALID_STATES:
            raise ValueError(
                f"'{v}' is not a recognised Indian state or UT. "
                f"Valid options: {sorted(VALID_STATES)}"
            )
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        if v not in VALID_DOMAINS:
            # Accept unknown domains rather than rejecting — default to "General"
            return "General"
        return v


class PredictRequest(BaseModel):
    """Top-level payload for POST /api/predict."""
    signals: SignalInput = Field(default_factory=SignalInput)
    context: ContextInput = Field(default_factory=ContextInput)


class PolicySimulationRequest(BaseModel):
    """Payload for POST /api/simulate."""
    policies: list[str] = Field(default_factory=list, min_length=1)
    state: Optional[str] = Field(default=None)
    target_population: Optional[int] = Field(default=None, ge=0)

    @field_validator("state")
    @classmethod
    def validate_state(cls, v):
        if v is not None and v not in VALID_STATES:
            raise ValueError(f"'{v}' is not a recognised Indian state or UT.")
        return v
