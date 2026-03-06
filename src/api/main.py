"""
FastAPI Application — Real-Time Recommendation API.

Endpoints:
- POST /recommend   → Select the best ad for a user
- POST /feedback    → Process click/conversion feedback (closed loop)
- GET  /health      → Health check
- GET  /metrics     → Current performance metrics
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
)
from src.api.recommendation_service import RecommendationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global service instance ──
service: RecommendationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the recommendation service on startup."""
    global service
    logger.info("Starting Recommendation Service...")
    service = RecommendationService()
    logger.info("✅ Service ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Real-Time Ad Recommendation API",
    description=(
        "Closed-Loop Multi-Objective Recommender System "
        "using H-DeepBandit × ε-Constraint policy. "
        "Powered by semantic embeddings (all-MiniLM-L6-v2) "
        "for zero-shot cold-start transfer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Select the best ad for a given user context.

    The system encodes the user profile and each ad via SentenceTransformer,
    then uses H-DeepBandit with ε-Constraint policy to select the arm
    that maximizes engagement subject to revenue ≥ ε.
    """
    if not request.available_ads:
        raise HTTPException(status_code=400, detail="No ads provided")

    ad_id, eng, rev, latency = service.recommend(
        user_text=request.user_text,
        available_ads=request.available_ads,
    )

    return RecommendationResponse(
        user_id=request.user_id,
        selected_ad_id=ad_id,
        engagement_score=round(eng, 4),
        revenue_score=round(rev, 4),
        latency_ms=round(latency, 2),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Process user feedback (click, conversion, revenue).

    This closes the loop: the agent updates its model based on the
    observed reward, improving future recommendations.
    """
    updated = service.process_feedback(
        user_text=request.user_text,
        ad_id=request.ad_id,
        click=request.click,
        conversion=request.conversion,
        revenue=request.revenue,
    )

    return FeedbackResponse(
        status="ok" if updated else "skipped",
        agent_updated=updated,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and model status."""
    return HealthResponse(
        status="healthy",
        model_type="H-DeepBandit",
        policy="epsilon-constraint",
        n_arms=service.agent.n_arms if service else 0,
        redis_connected=False,  # Will be True when Redis is integrated
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Get current performance metrics (sliding window)."""
    if service is None:
        return MetricsResponse()
    m = service.get_metrics()
    return MetricsResponse(**m)
