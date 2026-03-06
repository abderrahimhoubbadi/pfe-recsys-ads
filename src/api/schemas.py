"""
Pydantic schemas for the Recommendation API.
"""

from pydantic import BaseModel, Field
from typing import List


class AdInfo(BaseModel):
    """Description of an available ad."""

    ad_id: int
    title: str
    description: str
    category: str = ""


class RecommendationRequest(BaseModel):
    """Request to get a recommendation for a user."""

    user_id: int
    user_text: str = Field(
        ...,
        description="User profile text (e.g. 'homme 28 ans passionné de tech')",
    )
    available_ads: List[AdInfo] = Field(
        ...,
        description="List of available ads to choose from",
    )


class RecommendationResponse(BaseModel):
    """Response with the selected ad and prediction scores."""

    user_id: int
    selected_ad_id: int
    engagement_score: float
    revenue_score: float
    latency_ms: float
    model_type: str = "H-DeepBandit"
    policy: str = "epsilon-constraint"


class FeedbackRequest(BaseModel):
    """Feedback after an ad was shown to a user."""

    user_id: int
    ad_id: int
    user_text: str = Field(
        ...,
        description="User profile text (needed for model update)",
    )
    ad_text: str = Field(
        default="",
        description="Ad description text (optional, for logging)",
    )
    feedback_type: str = Field(
        default="full",
        description=(
            "Type of feedback: "
            "'click' = immediate click signal (revenue unknown yet), "
            "'conversion' = delayed conversion with final revenue, "
            "'full' = both click and revenue available at once"
        ),
    )
    click: bool = False
    conversion: bool = False
    revenue: float = 0.0


class FeedbackResponse(BaseModel):
    """Confirmation of feedback processing."""

    status: str = "ok"
    agent_updated: bool = True


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_type: str = "H-DeepBandit"
    policy: str = "epsilon-constraint"
    n_arms: int = 0
    redis_connected: bool = False
    model_version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Current performance metrics."""

    total_requests: int = 0
    total_feedbacks: int = 0
    pending_conversions: int = 0
    avg_ctr: float = 0.0
    avg_revenue: float = 0.0
    avg_latency_ms: float = 0.0
