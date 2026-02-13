import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.ips_evaluator import IPSEvaluator
from src.agents.linucb_agent import LinUCBAgent
from src.agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Simple random agent for baseline"""
    def __init__(self, n_arms, dimension):
        super().__init__(n_arms, dimension)
        self.rng = np.random.default_rng(42)
        
    def select_arm(self, context):
        return self.rng.integers(0, self.n_arms)
        
    def update(self, context, arm, reward):
        pass
        
    def get_model_params(self):
        return {}

def load_logs(filepath):
    print(f"Loading logs from {filepath}...")
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def run_plotting():
    LOG_FILE = "data/logs_10k.jsonl"
    OUTPUT_IMG = "metrics/ips_convergence.png"
    DIMENSION = 5
    N_ARMS = 5
    M_CAP = 10.0
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    
    # 1. Load Data
    dataset = load_logs(LOG_FILE)
    
    # 2. Random Policy
    print("Evaluating Random Policy...")
    random_agent = RandomAgent(n_arms=N_ARMS, dimension=DIMENSION)
    ips_rnd = IPSEvaluator(random_agent, cap_M=M_CAP)
    _, metrics_rnd = ips_rnd.evaluate(dataset)
    
    # 3. Logging Policy Baseline (Approximate as constant line)
    mean_reward = np.mean([d['reward'] for d in dataset])
    
    # 4. LinUCB
    print("Evaluating LinUCB...")
    linucb = LinUCBAgent(n_arms=N_ARMS, dimension=DIMENSION, alpha=0.5)
    ips_linucb = IPSEvaluator(linucb, cap_M=M_CAP)
    _, metrics_linucb = ips_linucb.evaluate(dataset)
    
    # 5. Plotting
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # X axis: Number of events processed
    # Note: history has same length as dataset because we append on Match AND Reject (using previous val)
    x = range(len(dataset))
    
    # Plot LinUCB
    plt.plot(metrics_linucb['history'], label='LinUCB (Online Learning)', color='blue', linewidth=2)
    
    # Plot Random
    plt.plot(metrics_rnd['history'], label='Random Policy', color='red', linestyle='--', alpha=0.7)
    
    # Plot Logging Policy (Constant)
    plt.axhline(y=mean_reward, color='green', linestyle=':', label='Logging Policy (Historical Average)', linewidth=2)
    
    plt.title(f"Convergence du Score IPS (Clipped M={M_CAP})")
    plt.xlabel("Nombre d'événements (Replay)")
    plt.ylabel("Score IPS Cumulatif Moyen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"Plot saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    run_plotting()
