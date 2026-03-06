"""
Infrastructure Factory — Abstract interfaces and backend selection.

Enables switching between Docker (Kafka + Redis) and GCP (Pub/Sub + Memorystore)
without changing any business logic code.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Abstract Interfaces
# ════════════════════════════════════════════════════════════════


class MessageProducer(ABC):
    """Abstract message producer (Kafka / Pub/Sub)."""

    @abstractmethod
    def send(self, topic: str, data: Dict[str, Any]) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class MessageConsumer(ABC):
    """Abstract message consumer (Kafka / Pub/Sub)."""

    @abstractmethod
    def consume_one(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def close(self) -> None: ...


class StateStore(ABC):
    """Abstract state store for model persistence (Redis / Memorystore)."""

    @abstractmethod
    def save_model(self, agent_id: str, state: Dict[str, bytes]) -> None: ...

    @abstractmethod
    def load_model(self, agent_id: str) -> Optional[Dict[str, bytes]]: ...

    @abstractmethod
    def clear(self, agent_id: str) -> None: ...


# ════════════════════════════════════════════════════════════════
# Factory Functions
# ════════════════════════════════════════════════════════════════


def create_producer(backend: str = None) -> MessageProducer:
    """
    Create a message producer.

    Args:
        backend: "kafka" or "pubsub". If None, reads from QUEUE_BACKEND env var.
    """
    backend = backend or os.getenv("QUEUE_BACKEND", "kafka")

    if backend == "kafka":
        from src.infra.kafka_messenger import KafkaProducerAdapter

        return KafkaProducerAdapter(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        )
    elif backend == "pubsub":
        from src.infra.pubsub_client import PubSubProducerAdapter

        return PubSubProducerAdapter(project_id=os.getenv("GCP_PROJECT_ID", ""))
    else:
        raise ValueError(f"Unknown queue backend: {backend}")


def create_consumer(
    topic: str,
    group_id: str = "recsys-group",
    backend: str = None,
) -> MessageConsumer:
    """
    Create a message consumer.

    Args:
        topic: Topic/subscription to consume from.
        group_id: Consumer group ID (Kafka only).
        backend: "kafka" or "pubsub". If None, reads from QUEUE_BACKEND env var.
    """
    backend = backend or os.getenv("QUEUE_BACKEND", "kafka")

    if backend == "kafka":
        from src.infra.kafka_messenger import KafkaConsumerAdapter

        return KafkaConsumerAdapter(
            topic=topic,
            group_id=group_id,
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        )
    elif backend == "pubsub":
        from src.infra.pubsub_client import PubSubConsumerAdapter

        return PubSubConsumerAdapter(
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            subscription=f"{topic}-sub",
        )
    else:
        raise ValueError(f"Unknown queue backend: {backend}")


def create_state_store(backend: str = None) -> StateStore:
    """
    Create a state store for model persistence.

    Args:
        backend: "redis" (works for both local Redis and GCP Memorystore).
    """
    backend = backend or os.getenv("STATE_BACKEND", "redis")

    if backend == "redis":
        from src.infra.redis_client import RedisStateStore

        return RedisStateStore(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )
    else:
        raise ValueError(f"Unknown state backend: {backend}")
