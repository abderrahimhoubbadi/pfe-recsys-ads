"""
Recommendation Service — Business logic layer.

Manages the H-DeepBandit agent lifecycle:
- Initialization with SentenceTransformer embeddings
- Arm selection via ε-Constraint MOO policy
- Model updates on feedback
- State persistence via Redis
"""

import time
import logging
import numpy as np
import torch
import io
from typing import Dict, List, Tuple
from collections import deque

from src.agents.global_semantic_neural import GlobalSemanticDeepBandit
from src.policy.moo_policies import epsilon_constraint_policy
from src.llm.sentence_transformer_client import SentenceTransformerClient
from src.api.schemas import AdInfo

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Encapsulates the H-DeepBandit agent for real-time recommendations.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 1.0,
        n_ensemble: int = 5,
        hidden_dim: int = 32,
        epsilon: float = 0.3,
        redis_host: str = None,
        redis_port: int = 6379,
        save_every_n: int = 50,
    ):
        # ── Embedding encoder ──
        self.encoder = SentenceTransformerClient(embedding_model)
        self.emb_dim = self.encoder.get_dimension()

        # ── H-DeepBandit agent ──
        self.agent = GlobalSemanticDeepBandit(
            user_dim=self.emb_dim,
            ad_dim=self.emb_dim,
            alpha=alpha,
            n_ensemble=n_ensemble,
            hidden_dim=hidden_dim,
        )

        # ── ε-Constraint MOO policy ──
        self.policy = epsilon_constraint_policy(
            primary_objective="click",
            constraint_objective="revenue",
            epsilon=epsilon,
        )

        # ── Ad registry ──
        self.ad_texts: Dict[int, str] = {}  # ad_id → text
        self.ad_embeddings: Dict[int, np.ndarray] = {}  # ad_id → embedding
        self.ad_id_to_arm: Dict[int, int] = {}  # ad_id → arm_index
        self.arm_to_ad_id: Dict[int, int] = {}  # arm_index → ad_id

        # ── Metrics tracking ──
        self.total_requests = 0
        self.total_feedbacks = 0
        self._ctr_window: deque = deque(maxlen=1000)
        self._rev_window: deque = deque(maxlen=1000)
        self._latency_window: deque = deque(maxlen=1000)

        # ── Redis state persistence ──
        self.redis_client = None
        self.save_every_n = save_every_n
        self._feedback_since_save = 0

        if redis_host:
            try:
                from src.infra.redis_client import RedisStateStore

                self.redis_client = RedisStateStore(host=redis_host, port=redis_port)
                if self.redis_client.ping():
                    logger.info(f"Redis connected at {redis_host}:{redis_port}")
                    self._load_state()
                else:
                    logger.warning("Redis not reachable, running without persistence")
                    self.redis_client = None
            except Exception as e:
                logger.warning(f"Redis init failed: {e}, running without persistence")
                self.redis_client = None

        logger.info(
            f"RecommendationService initialized: "
            f"model=H-DeepBandit, dim={self.emb_dim}, "
            f"ensemble={n_ensemble}, epsilon={epsilon}, "
            f"redis={'connected' if self.redis_client else 'disabled'}"
        )

    # ════════════════════════════════════════════════════════════
    # Ad Management
    # ════════════════════════════════════════════════════════════

    def register_ads(self, ads: List[AdInfo]) -> int:
        """
        Register ads and compute their embeddings.
        Returns the number of newly registered ads.
        """
        new_count = 0
        new_embeddings = {}

        for ad in ads:
            if ad.ad_id not in self.ad_embeddings:
                text = f"{ad.title}. {ad.description}. {ad.category}".strip()
                emb = self.encoder.get_embedding(text)
                arm_idx = len(self.ad_embeddings)

                self.ad_texts[ad.ad_id] = text
                self.ad_embeddings[ad.ad_id] = emb
                self.ad_id_to_arm[ad.ad_id] = arm_idx
                self.arm_to_ad_id[arm_idx] = ad.ad_id
                new_embeddings[arm_idx] = emb
                new_count += 1

        if new_embeddings:
            self.agent.expand_arms(new_embeddings)
            # Also set full embeddings on first call
            if self.agent.n_arms == len(new_embeddings):
                self.agent.set_ad_embeddings(
                    {
                        self.ad_id_to_arm[aid]: emb
                        for aid, emb in self.ad_embeddings.items()
                    }
                )
            logger.info(
                f"Registered {new_count} new ads (total active: {self.agent.n_arms})"
            )

        return new_count

    # ════════════════════════════════════════════════════════════
    # Recommendation
    # ════════════════════════════════════════════════════════════

    def recommend(
        self, user_text: str, available_ads: List[AdInfo]
    ) -> Tuple[int, float, float, float]:
        """
        Select the best ad for a user.

        Returns:
            (ad_id, engagement_score, revenue_score, latency_ms)
        """
        t0 = time.perf_counter()

        # 1. Register any new ads
        self.register_ads(available_ads)

        # 2. Encode user
        user_emb = self.encoder.get_embedding(user_text)

        # 3. Select arm via H-DeepBandit + ε-Constraint
        arm_idx = self.agent.select_arm(user_emb, policy=self.policy)
        ad_id = self.arm_to_ad_id.get(arm_idx, available_ads[0].ad_id)

        # 4. Get scores for the selected arm
        ctx = self.agent._build_context(user_emb, arm_idx)
        x = torch.FloatTensor(ctx)
        arm_pred = {}
        for obj in self.agent.objectives:
            preds = []
            with torch.no_grad():
                for net in self.agent.ensembles[obj]:
                    net.eval()
                    preds.append(net(x).item())
            arm_pred[obj] = float(np.mean(preds))

        latency_ms = (time.perf_counter() - t0) * 1000

        # 5. Track metrics
        self.total_requests += 1
        self._latency_window.append(latency_ms)

        logger.debug(
            f"Recommend: user='{user_text[:30]}...' → ad_id={ad_id}, "
            f"eng={arm_pred.get('click', 0):.4f}, "
            f"rev={arm_pred.get('revenue', 0):.4f}, "
            f"latency={latency_ms:.1f}ms"
        )

        return (
            ad_id,
            arm_pred.get("click", 0.0),
            arm_pred.get("revenue", 0.0),
            latency_ms,
        )

    # ════════════════════════════════════════════════════════════
    # Feedback (Closed Loop)
    # ════════════════════════════════════════════════════════════

    def process_feedback(
        self,
        user_text: str,
        ad_id: int,
        click: bool,
        conversion: bool,
        revenue: float,
    ) -> bool:
        """
        Process user feedback and update the model.

        Returns True if the model was updated.
        """
        if ad_id not in self.ad_id_to_arm:
            logger.warning(f"Feedback for unknown ad_id={ad_id}, skipping")
            return False

        arm_idx = self.ad_id_to_arm[ad_id]
        user_emb = self.encoder.get_embedding(user_text)

        rewards = {
            "click": 1.0 if click else 0.0,
            "revenue": revenue,
        }

        self.agent.update(user_emb, arm_idx, rewards)

        # Track metrics
        self.total_feedbacks += 1
        self._ctr_window.append(1.0 if click else 0.0)
        self._rev_window.append(revenue)

        # Auto-save to Redis every N feedbacks
        self._feedback_since_save += 1
        if self.redis_client and self._feedback_since_save >= self.save_every_n:
            self._save_state()
            self._feedback_since_save = 0

        logger.debug(
            f"Feedback: ad_id={ad_id}, click={click}, "
            f"conversion={conversion}, revenue={revenue:.4f}"
        )

        return True

    # ════════════════════════════════════════════════════════════
    # Metrics
    # ════════════════════════════════════════════════════════════

    def get_metrics(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "total_feedbacks": self.total_feedbacks,
            "avg_ctr": float(np.mean(self._ctr_window)) if self._ctr_window else 0.0,
            "avg_revenue": float(np.mean(self._rev_window))
            if self._rev_window
            else 0.0,
            "avg_latency_ms": float(np.mean(self._latency_window))
            if self._latency_window
            else 0.0,
        }

    # ════════════════════════════════════════════════════════════
    # State Persistence (Redis)
    # ════════════════════════════════════════════════════════════

    def _save_state(self) -> None:
        """Save agent state to Redis."""
        if not self.redis_client:
            return
        try:
            agent_id = "h-deepbandit-main"
            # Save ensemble weights
            state = {}
            for obj in self.agent.objectives:
                for i, net in enumerate(self.agent.ensembles[obj]):
                    buf = io.BytesIO()
                    torch.save(net.state_dict(), buf)
                    state[f"ensemble_{obj}_{i}"] = buf.getvalue()

            self.redis_client.save_model(agent_id, state)

            # Save ad registry as metadata
            import json

            meta = {
                "ad_id_to_arm": {str(k): v for k, v in self.ad_id_to_arm.items()},
                "arm_to_ad_id": {str(k): v for k, v in self.arm_to_ad_id.items()},
                "ad_texts": {str(k): v for k, v in self.ad_texts.items()},
                "total_requests": self.total_requests,
                "total_feedbacks": self.total_feedbacks,
            }
            self.redis_client.save_meta(agent_id, meta)

            # Save ad embeddings
            for ad_id, emb in self.ad_embeddings.items():
                self.redis_client.save_numpy(agent_id, f"ad_emb_{ad_id}", emb)

            logger.info(f"State saved to Redis (feedbacks={self.total_feedbacks})")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load agent state from Redis (if available)."""
        if not self.redis_client:
            return
        try:
            agent_id = "h-deepbandit-main"
            meta = self.redis_client.load_meta(agent_id)
            if meta is None:
                logger.info("No saved state found in Redis, starting fresh")
                return

            # Restore ad registry
            self.ad_id_to_arm = {int(k): v for k, v in meta["ad_id_to_arm"].items()}
            self.arm_to_ad_id = {int(k): v for k, v in meta["arm_to_ad_id"].items()}
            self.ad_texts = {int(k): v for k, v in meta["ad_texts"].items()}
            self.total_requests = meta.get("total_requests", 0)
            self.total_feedbacks = meta.get("total_feedbacks", 0)

            # Restore ad embeddings
            for ad_id_str in self.ad_texts:
                emb = self.redis_client.load_numpy(agent_id, f"ad_emb_{ad_id_str}")
                if emb is not None:
                    self.ad_embeddings[ad_id_str] = emb

            # Set embeddings on agent
            arm_embs = {
                self.ad_id_to_arm[aid]: emb for aid, emb in self.ad_embeddings.items()
            }
            self.agent.set_ad_embeddings(arm_embs)

            # Restore ensemble weights
            state = self.redis_client.load_model(agent_id)
            if state:
                for obj in self.agent.objectives:
                    for i, net in enumerate(self.agent.ensembles[obj]):
                        key = f"ensemble_{obj}_{i}"
                        if key in state and state[key]:
                            buf = io.BytesIO(state[key])
                            net.load_state_dict(torch.load(buf, weights_only=True))

            logger.info(
                f"State restored from Redis: "
                f"{len(self.ad_embeddings)} ads, "
                f"{self.total_feedbacks} feedbacks"
            )
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, starting fresh")
