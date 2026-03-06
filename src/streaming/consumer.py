"""
Kafka Streaming Consumer — Processes impressions and feedback in real-time.

This is the streaming counterpart to the REST API.
Instead of HTTP requests, events flow through Kafka topics.
"""

import json
import time
import logging
import signal
import sys

from src.infra.factory import create_producer, create_consumer
from src.api.recommendation_service import RecommendationService
from src.api.schemas import AdInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingPipeline:
    """
    Closed-loop streaming pipeline:
    1. Consume from 'impressions' topic
    2. Produce decisions to 'decisions' topic
    3. Consume from 'feedback' topic
    4. Update the agent model (closed loop)
    """

    def __init__(self, queue_backend: str = "kafka"):
        self.service = RecommendationService()

        # Producers
        self.decision_producer = create_producer(backend=queue_backend)

        # Consumers
        self.impression_consumer = create_consumer(
            topic="impressions",
            group_id="recsys-impression-group",
            backend=queue_backend,
        )
        self.feedback_consumer = create_consumer(
            topic="feedback", group_id="recsys-feedback-group", backend=queue_backend
        )

        self.running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info(f"StreamingPipeline initialized (backend={queue_backend})")

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def process_impression(self, message: dict) -> None:
        """
        Process an impression event.

        Expected message format:
        {
            "user_id": 42,
            "user_text": "homme 28 ans tech",
            "available_ads": [
                {"ad_id": 1, "title": "...", "description": "...", "category": "..."},
                ...
            ]
        }
        """
        try:
            user_id = message["user_id"]
            user_text = message["user_text"]
            ads = [AdInfo(**ad) for ad in message["available_ads"]]

            ad_id, eng, rev, latency = self.service.recommend(
                user_text=user_text,
                available_ads=ads,
            )

            decision = {
                "user_id": user_id,
                "selected_ad_id": ad_id,
                "engagement_score": round(eng, 4),
                "revenue_score": round(rev, 4),
                "latency_ms": round(latency, 2),
                "timestamp": time.time(),
            }

            self.decision_producer.send("decisions", decision)
            logger.info(
                f"Decision: user={user_id} → ad={ad_id} "
                f"(eng={eng:.4f}, rev={rev:.4f}, {latency:.1f}ms)"
            )

        except (KeyError, ValueError) as e:
            logger.error(f"Invalid impression message: {e}")

    def process_feedback(self, message: dict) -> None:
        """
        Process a feedback event (closed loop).

        Expected message format:
        {
            "user_id": 42,
            "ad_id": 7,
            "user_text": "homme 28 ans tech",
            "ad_text": "Latest smartphone deals",
            "click": true,
            "conversion": false,
            "revenue": 0.0
        }
        """
        try:
            self.service.process_feedback(
                user_text=message["user_text"],
                ad_id=message["ad_id"],
                click=message.get("click", False),
                conversion=message.get("conversion", False),
                revenue=message.get("revenue", 0.0),
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid feedback message: {e}")

    def run(self) -> None:
        """
        Main event loop: alternates between consuming impressions and feedback.
        """
        logger.info("🚀 StreamingPipeline started — listening for events...")

        while self.running:
            # Process impressions (non-blocking)
            impression = self.impression_consumer.consume_one(timeout=0.1)
            if impression:
                self.process_impression(impression)

            # Process feedback (non-blocking)
            feedback = self.feedback_consumer.consume_one(timeout=0.1)
            if feedback:
                self.process_feedback(feedback)

            # Flush decisions
            self.decision_producer.flush()

        # Cleanup
        self.impression_consumer.close()
        self.feedback_consumer.close()
        self.decision_producer.flush()
        logger.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    import os

    backend = os.getenv("QUEUE_BACKEND", "kafka")
    pipeline = StreamingPipeline(queue_backend=backend)
    pipeline.run()
