import sys
import os
import json
import time
import numpy as np
import threading
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import (
    KAFKA_TOPIC, KAFKA_GROUP_ID, N_ITERATIONS,
    DIMENSION, N_ARMS, ALPHA
)
from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator
from src.agents.redis_linucb_agent import RedisLinUCBAgent
from src.infra.redis_client import RedisClient
from src.infra.kafka_messenger import KafkaProducerWrapper, KafkaConsumerWrapper

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTimeSim")

# Agent ID (unique per instance)
AGENT_ID = "rt_agent_001"

def traffic_generator(producer, n_events):
    """
    Simulates user traffic arriving at the system.
    Generates contexts and sends them to Kafka.
    """
    ctx_gen = ContextGenerator(dimension=DIMENSION)
    logger.info(f"[Generator] Starting to generate {n_events} events...")
    
    for i in range(n_events):
        context = ctx_gen.get_context()
        event = {
            "event_id": f"evt_{i}",
            "context": context.tolist(),
            "timestamp": time.time()
        }
        producer.send(KAFKA_TOPIC, event)
        # Simulate variable traffic
        # time.sleep(0.001) 
        
        if (i+1) % 100 == 0:
            logger.info(f"[Generator] Produced {i+1} events")
            producer.flush()
            
    producer.flush()
    logger.info("[Generator] Finished producing.")

def agent_consumer(consumer, redis_client, n_events):
    """
    Simulates the Agent service.
    Consumes requests, predicts, simulates reward (Oracle), and updates.
    """
    # Initialize Agent
    agent = RedisLinUCBAgent(
        agent_id=AGENT_ID,
        n_arms=N_ARMS,
        dimension=DIMENSION,
        redis_client=redis_client,
        alpha=ALPHA
    )
    
    # Initialize Environment (Oracle for rewards)
    # In a real system, this would be decoupled (user clicks later)
    env = RewardSimulator(dimension=DIMENSION, n_arms=N_ARMS)
    
    clicks = 0
    total_revenue = 0.0
    start_time = time.time()
    
    logger.info("[Agent] Listening for events...")
    
    count = 0
    while count < n_events:
        msg = consumer.consume_one(timeout=2.0)
        
        if msg is None:
            # Check if we are done or just waiting
            # For this script, we assume if we processed n_events we are done
            continue
            
        context = np.array(msg['context'])
        
        # 1. Prediction (Select Arm)
        chosen_arm = agent.select_arm(context)
        
        # 2. Simulation (User Reaction)
        # In real life, we wouldn't have the reward immediately. 
        # We would probably join this later. 
        # Here we simulate immediate feedback for 'Closed Loop' testing.
        rewards = env.get_reward(context, chosen_arm)
        # Extract click reward for single-objective agent (backward compatible)
        reward_click = rewards['click'] if isinstance(rewards, dict) else rewards
        
        # 3. Learning (Update)
        agent.update(context, chosen_arm, reward_click)
        
        clicks += reward_click
        if isinstance(rewards, dict):
            total_revenue += rewards['revenue']
        count += 1
        
        if count % 100 == 0:
            ctr = clicks / count
            avg_rev = total_revenue / count
            logger.info(f"[Agent] Processed {count}/{n_events} | CTR: {ctr:.4f} | Revenue: {avg_rev:.4f}")

    duration = time.time() - start_time
    logger.info(f"[Agent] Processed {count} events in {duration:.2f}s ({count/duration:.1f} req/s)")
    logger.info(f"[Agent] Final CTR: {clicks/count:.4f}")

def run_realtime_simulation():
    # Setup Infrastructure
    redis_client = RedisClient()
    
    # Ensure Redis is clean for this run
    redis_client.clear_agent(AGENT_ID) # Warning: clears base keys, but check if correct for arm keys
    # Helper to clear arm keys manually as in test
    for i in range(N_ARMS):
        redis_client.client.delete(f"{AGENT_ID}:arm:{i}:A_inv", f"{AGENT_ID}:arm:{i}:b")

    producer = KafkaProducerWrapper()
    consumer = KafkaConsumerWrapper(KAFKA_TOPIC, KAFKA_GROUP_ID)
    
    # Threads
    gen_thread = threading.Thread(target=traffic_generator, args=(producer, N_ITERATIONS))
    agent_thread = threading.Thread(target=agent_consumer, args=(consumer, redis_client, N_ITERATIONS))
    
    logger.info("Starting threads...")
    agent_thread.start()
    time.sleep(1) # Give consumer a sec to join group
    gen_thread.start()
    
    gen_thread.join()
    agent_thread.join()
    
    consumer.close()
    logger.info("Simulation Complete.")

if __name__ == "__main__":
    run_realtime_simulation()
