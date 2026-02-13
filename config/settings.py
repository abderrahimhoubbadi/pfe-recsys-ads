"""
Centralized Configuration for pfe-recsys-ads.

Import these settings in your scripts:
    from config.settings import KAFKA_TOPIC, DIMENSION
"""

# =============================================================================
# Model Parameters
# =============================================================================
DIMENSION = 5           # Feature dimension for context vectors
N_ARMS = 5              # Number of ads/arms
ALPHA = 0.2             # Exploration parameter for LinUCB
LAMBDA_REG = 1.0        # Regularization for Ridge Regression

# =============================================================================
# Multi-Objective Parameters
# =============================================================================
OBJECTIVES = ['click', 'revenue']
EPSILON_THRESHOLD = 0.3  # Minimum revenue constraint for ε-constraint policy

# =============================================================================
# Kafka Configuration
# =============================================================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "ad_requests"
KAFKA_GROUP_ID = "agent_group_v1"

# =============================================================================
# Redis Configuration
# =============================================================================
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# =============================================================================
# Experiment Parameters
# =============================================================================
N_ITERATIONS = 2000     # Default number of iterations for experiments
RANDOM_SEED = 42        # For reproducibility

# =============================================================================
# Neural Agent Parameters
# =============================================================================
HIDDEN_DIM = 32         # Hidden layer size for neural agents
LEARNING_RATE = 0.01    # Learning rate for neural optimizers
N_ENSEMBLE = 5          # Number of ensemble members (Deep Bandits)
TS_VARIANCE = 0.2       # Thompson Sampling exploration variance
NEURAL_TS_SIGMA = 0.05  # NeuralTS parameter perturbation scale
PESSIMISM_DECAY = 0.995  # Offline-to-Online pessimism decay rate
DELAY_WINDOW = 50        # Delayed feedback pending buffer size

