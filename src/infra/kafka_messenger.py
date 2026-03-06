"""
Kafka Message Adapters — Wraps confluent_kafka into the factory interfaces.

Used in Option A (Docker Compose) deployment.
"""

import json
import logging
from typing import Dict, Any, Optional

from src.infra.factory import MessageProducer, MessageConsumer

logger = logging.getLogger(__name__)


class KafkaProducerAdapter(MessageProducer):
    """Kafka producer implementing the MessageProducer interface."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        from confluent_kafka import Producer

        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "socket.timeout.ms": 1000,
            }
        )
        logger.info(f"KafkaProducer connected to {bootstrap_servers}")

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Kafka delivery failed: {err}")

    def send(self, topic: str, data: Dict[str, Any]) -> None:
        self.producer.poll(0)
        try:
            payload = json.dumps(data).encode("utf-8")
            self.producer.produce(topic, payload, callback=self._delivery_report)
        except BufferError:
            logger.error("Kafka producer queue is full")

    def flush(self) -> None:
        self.producer.flush()


class KafkaConsumerAdapter(MessageConsumer):
    """Kafka consumer implementing the MessageConsumer interface."""

    def __init__(
        self,
        topic: str,
        group_id: str = "recsys-group",
        bootstrap_servers: str = "localhost:9092",
    ):
        from confluent_kafka import Consumer

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        self.consumer.subscribe([topic])
        logger.info(f"KafkaConsumer subscribed to '{topic}' (group={group_id})")

    def consume_one(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        from confluent_kafka import KafkaError

        msg = self.consumer.poll(timeout)
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                return None
            logger.error(f"Kafka error: {msg.error()}")
            return None
        try:
            return json.loads(msg.value().decode("utf-8"))
        except json.JSONDecodeError:
            logger.error(f"Failed to decode Kafka message: {msg.value()}")
            return None

    def close(self) -> None:
        self.consumer.close()
