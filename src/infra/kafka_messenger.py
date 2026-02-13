import json
import logging
from typing import Dict, Any, Optional, Callable
from confluent_kafka import Producer, Consumer, KafkaError

class KafkaProducerWrapper:
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'socket.timeout.ms': 1000,
        })
        self.logger = logging.getLogger(__name__)

    def delivery_report(self, err, msg):
        """ Called once for each message produced to indicate delivery result. """
        if err is not None:
            self.logger.error(f'Message delivery failed: {err}')
        # else:
            # self.logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send(self, topic: str, data: Dict[str, Any]):
        """ Asynchronously send a dictionary as JSON. """
        # Trigger any available delivery report callbacks from previous produce() calls
        self.producer.poll(0)

        try:
            json_data = json.dumps(data).encode('utf-8')
            self.producer.produce(topic, json_data, callback=self.delivery_report)
        except BufferError:
            self.logger.error("Local producer queue is full")

    def flush(self):
        self.producer.flush()


class KafkaConsumerWrapper:
    def __init__(self, topic: str, group_id: str, bootstrap_servers: str = 'localhost:9092'):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest' # Start from beginning if no offset
        })
        self.consumer.subscribe([topic])
        self.running = True
        self.logger = logging.getLogger(__name__)

    def consume_one(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """ Consume a single message. Returns parsed JSON or None. """
        msg = self.consumer.poll(timeout)

        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                return None # End of partition event
            else:
                self.logger.error(msg.error())
                return None

        try:
            return json.loads(msg.value().decode('utf-8'))
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode message: {msg.value()}")
            return None

    def close(self):
        self.consumer.close()
