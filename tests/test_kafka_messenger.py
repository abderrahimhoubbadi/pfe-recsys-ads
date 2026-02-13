import pytest
import time
from src.infra.kafka_messenger import KafkaProducerWrapper, KafkaConsumerWrapper

# Check usage of real Kafka or Mock? 
# For integration tests, we want real Kafka if available.
# But CI might not have it.
# We'll use the same logic as Redis tests.

import socket

def is_kafka_ready(host='localhost', port=9092):
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return True
    except:
        return False

@pytest.mark.skipif(not is_kafka_ready(), reason="Kafka is not reachable")
class TestKafkaIntegration:
    def setup_method(self):
        self.test_topic = f"test_topic_{int(time.time())}"
        self.group_id = f"test_group_{int(time.time())}"
        self.producer = KafkaProducerWrapper()
        # Consumer needs to exist before we send if we want to be sure (though Kafka buffers)
        # But 'earliest' offset should handle it.
        self.consumer = KafkaConsumerWrapper(self.test_topic, self.group_id)

    def teardown_method(self):
        self.producer.flush()
        self.consumer.close()

    def test_send_receive(self):
        """Test sending a message and receiving it"""
        data = {"event_id": 123, "context": [0.1, 0.2, 0.3]}
        
        self.producer.send(self.test_topic, data)
        self.producer.flush()
        
        # Give it a moment
        time.sleep(1)
        
        # Consume
        received = self.consumer.consume_one(timeout=5.0)
        
        assert received is not None
        assert received['event_id'] == 123
        assert received['context'] == [0.1, 0.2, 0.3]

    def test_consume_timeout(self):
        """Test that consume returns None when empty"""
        # Ensure topic is empty or we read past end
        # Since it's a new topic/group, it should be empty after the first test consumption
        # Or we use a new topic
        
        # Let's trust setup_method creates fresh topics/groups by timestamp
        # But wait, timestamp resolution is second. If tests run fast, collision.
        # Add random suffix
        import random
        suffix = random.randint(0, 10000)
        topic = f"empty_topic_{suffix}"
        gp = f"empty_group_{suffix}"
        
        consumer = KafkaConsumerWrapper(topic, gp)
        msg = consumer.consume_one(timeout=1.0)
        consumer.close()
        
        assert msg is None
