"""
Google Cloud Pub/Sub Adapters — Wraps the google-cloud-pubsub SDK
into the factory interfaces.

Used in Option B (GCP Production) deployment.
"""

import json
import logging
from typing import Dict, Any, Optional
from collections import deque

from src.infra.factory import MessageProducer, MessageConsumer

logger = logging.getLogger(__name__)


class PubSubProducerAdapter(MessageProducer):
    """Google Cloud Pub/Sub producer implementing the MessageProducer interface."""

    def __init__(self, project_id: str):
        from google.cloud import pubsub_v1

        self.publisher = pubsub_v1.PublisherClient()
        self.project_id = project_id
        self._futures = []
        logger.info(f"PubSubProducer initialized for project '{project_id}'")

    def send(self, topic: str, data: Dict[str, Any]) -> None:
        topic_path = self.publisher.topic_path(self.project_id, topic)
        payload = json.dumps(data).encode("utf-8")
        future = self.publisher.publish(topic_path, payload)
        self._futures.append(future)

    def flush(self) -> None:
        for future in self._futures:
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Pub/Sub publish failed: {e}")
        self._futures.clear()


class PubSubConsumerAdapter(MessageConsumer):
    """
    Google Cloud Pub/Sub consumer implementing the MessageConsumer interface.

    Uses synchronous pull for compatibility with the consume_one() pattern.
    """

    def __init__(self, project_id: str, subscription: str):
        from google.cloud import pubsub_v1

        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            project_id, subscription
        )
        self._buffer: deque = deque()
        logger.info(
            f"PubSubConsumer initialized for subscription '{subscription}' "
            f"in project '{project_id}'"
        )

    def consume_one(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        # Return buffered message if available
        if self._buffer:
            return self._buffer.popleft()

        # Synchronous pull
        from google.cloud.pubsub_v1.types import PullRequest

        try:
            response = self.subscriber.pull(
                request=PullRequest(
                    subscription=self.subscription_path,
                    max_messages=10,
                ),
                timeout=timeout,
            )
        except Exception as e:
            logger.debug(f"Pub/Sub pull timeout or error: {e}")
            return None

        ack_ids = []
        for msg in response.received_messages:
            ack_ids.append(msg.ack_id)
            try:
                data = json.loads(msg.message.data.decode("utf-8"))
                self._buffer.append(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode Pub/Sub message: {msg.message.data}")

        if ack_ids:
            self.subscriber.acknowledge(
                subscription=self.subscription_path, ack_ids=ack_ids
            )

        return self._buffer.popleft() if self._buffer else None

    def close(self) -> None:
        self.subscriber.close()
