"""
Semantic Reward Simulator — Deterministic, noiseless rewards based on
semantic similarity between user profiles and ad descriptions.

The core mechanism:
1. Pre-compute embeddings for all User Profiles and all Ads using SentenceTransformer.
2. Engagement Score (CTR) = sigmoid(scale * cosine_sim(emb_u, emb_a)), continuous in [0, 1].
3. Conversion Rate (CVR) per ad: inversely correlated with price.
   Cheap popular ads convert easily (CVR ~15-20%), expensive niche ads rarely (CVR ~1-3%).
4. Revenue = CTR * CVR * Price (Full Conversion Funnel / eCPM model).

Deterministic mode: No coin-flip sampling. The exact scores are returned
as rewards, yielding perfectly smooth learning curves ideal for comparing agents.

Key feature: Supports dynamic arm expansion (Cold-Start) mid-simulation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from src.llm.sentence_transformer_client import SentenceTransformerClient
from src.env.semantic_env.text_dataset_loader import TextDatasetLoader

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticRewardSimulator:
    """
    Deterministic reward engine based on semantic matching.

    Engagement score for user u and ad a:
        engagement(u, a) = sigmoid(scale * cosine_sim(emb_u, emb_a))

    This produces a continuous value in (0, 1) where:
         - Semantically close (user, ad) pairs → high engagement (~0.8-0.9).
        - Unrelated pairs → low engagement (~0.3-0.4).
        - Revenue = CTR * CVR * price (full conversion funnel).
        - CVR inversely correlated with price (expensive ads convert less).
    """

    def __init__(
        self,
        dataset: TextDatasetLoader,
        embedding_model: str = "all-MiniLM-L6-v2",
        scale: float = 5.0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.scale = scale

        # Load the fast SentenceTransformer
        self.encoder = SentenceTransformerClient(embedding_model)

        # Pre-compute user embeddings
        self.user_embeddings = {}
        for user in dataset.user_profiles:
            text = dataset.get_user_text(user)
            self.user_embeddings[user["id"]] = self.encoder.get_embedding(text)

        # Pre-compute ad embeddings (known + hidden)
        self.ad_embeddings = {}
        for ad in dataset.all_ads:
            text = f"{ad['title']}. {ad['desc']}"
            self.ad_embeddings[ad["id"]] = self.encoder.get_embedding(text)

        # Current active ads (initially only known ones)
        self._active_ads: List[Dict] = list(dataset.get_known_ads())
        self._active_ad_ids: List[int] = [ad["id"] for ad in self._active_ads]

        # Assign prices inversely proportional to average similarity
        # (high-CTR ads get low prices → multi-objective conflict)
        self._assign_prices()

        # Assign Conversion Rates (CVR) inversely proportional to price
        # (expensive niche ads convert less even after click)
        self._assign_cvr()

        logger.info(
            f"SemanticRewardSimulator initialized (DETERMINISTIC mode): "
            f"{len(self._active_ads)} active arms"
        )

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _assign_prices(self):
        """
        Assign a price to each ad using EXPONENTIAL inverse CTR-CPC pricing.

        Real-world insight: in ad marketplaces (Google Ads, Meta Ads),
        popular/generic keywords cost ~$0.10/click while niche/high-intent
        keywords (insurance, SaaS) can cost ~$15/click.

        This creates a sharp multi-objective conflict:
          - High-CTR ads  → cheap  (clickbait, mass appeal)
          - Low-CTR ads   → expensive (niche, high commercial intent)

        Formula: price = MIN_CPC * (MAX_CPC / MIN_CPC) ^ (rank / N)
        """
        MIN_CPC = 0.10  # cheapest ad (most popular / highest CTR)
        MAX_CPC = 15.00  # most expensive ad (niche / lowest CTR)

        # Compute average cos-sim across all users for each ad
        avg_sims = {}
        for ad in self.dataset.all_ads:
            sims = []
            for uid, u_emb in self.user_embeddings.items():
                sim = _cosine_similarity(u_emb, self.ad_embeddings[ad["id"]])
                sims.append(sim)
            avg_sims[ad["id"]] = np.mean(sims)

        # Map similarity rank to price (exponential inverse relationship)
        sorted_ids = sorted(avg_sims.keys(), key=lambda x: avg_sims[x], reverse=True)
        n = len(sorted_ids)
        self.ad_prices = {}
        for rank, ad_id in enumerate(sorted_ids):
            # rank 0 (most popular) → MIN_CPC ($0.10)
            # rank n-1 (least popular) → MAX_CPC ($15.00)
            t = rank / max(n - 1, 1)
            self.ad_prices[ad_id] = MIN_CPC * (MAX_CPC / MIN_CPC) ** t

    def _assign_cvr(self):
        """
        Assign a Conversion Rate (CVR) to each ad.

        Real-world insight (Conversion Funnel):
          - A click does NOT guarantee revenue. The user may click an ad
            but leave without purchasing/subscribing.
          - Cheap popular ads (clickbait) convert easily: CVR ~15-20%.
          - Expensive niche ads (SaaS, insurance) convert rarely: CVR ~1-3%.

        This models the full funnel: Impression → Click (CTR) → Conversion (CVR) → Revenue.

        Formula: CVR = MAX_CVR * (MIN_CVR / MAX_CVR) ^ (price_rank / N)
        """
        MIN_CVR = 0.01  # 1% conversion for the most expensive ad
        MAX_CVR = 0.20  # 20% conversion for the cheapest ad

        # Sort ads by price (cheapest first → highest CVR first)
        sorted_ids = sorted(self.ad_prices.keys(), key=lambda x: self.ad_prices[x])
        n = len(sorted_ids)
        self.ad_cvr = {}
        for rank, ad_id in enumerate(sorted_ids):
            # rank 0 (cheapest / most popular) → MAX_CVR (20%)
            # rank n-1 (most expensive / niche) → MIN_CVR (1%)
            t = rank / max(n - 1, 1)
            self.ad_cvr[ad_id] = MAX_CVR * (MIN_CVR / MAX_CVR) ** t

    def get_n_arms(self) -> int:
        """Current number of active arms."""
        return len(self._active_ads)

    def get_active_ads(self) -> List[Dict]:
        return self._active_ads

    def inject_cold_start_ads(self):
        """
        Inject the hidden ads into the active pool.
        This simulates the 'Cold-Start Shock' event.
        """
        hidden = self.dataset.get_hidden_ads()
        self._active_ads.extend(hidden)
        self._active_ad_ids = [ad["id"] for ad in self._active_ads]
        logger.info(
            f"🔥 Cold-Start Shock: injected {len(hidden)} new ads. "
            f"Total active arms: {len(self._active_ads)}"
        )

    def get_engagement_score(self, user_id: int, arm_index: int) -> float:
        """
        Compute the deterministic engagement score for a user-ad pair.
        Returns a continuous value in (0, 1) via sigmoid(scale * cosine_sim).
        """
        ad_id = self._active_ad_ids[arm_index]
        u_emb = self.user_embeddings[user_id]
        a_emb = self.ad_embeddings[ad_id]

        cos_sim = _cosine_similarity(u_emb, a_emb)
        return self._sigmoid(self.scale * cos_sim)

    def get_reward(self, user_id: int, arm_index: int) -> Dict[str, float]:
        """
        Deterministic reward using the full Conversion Funnel (eCPM model).

        Returns:
            click: engagement_score (CTR probability, continuous 0-1)
            revenue: CTR * CVR * Price (expected revenue per impression)

        The conversion funnel:
            Impression → Click (CTR) → Conversion (CVR) → Revenue
            E[Revenue] = p(Click) × p(Achat|Click) × Prix
        """
        engagement = self.get_engagement_score(user_id, arm_index)
        ad_id = self._active_ad_ids[arm_index]
        price = self.ad_prices[ad_id]
        cvr = self.ad_cvr[ad_id]

        # Full funnel: Revenue = CTR * CVR * Price
        revenue = engagement * cvr * price

        return {"click": float(engagement), "revenue": float(revenue)}

    def get_expected_reward(self, user_id: int, arm_index: int) -> Dict[str, float]:
        """Expected reward (identical to get_reward in deterministic mode)."""
        return self.get_reward(user_id, arm_index)

    def get_user_context_embedding(self, user_id: int) -> np.ndarray:
        """
        Get the pre-computed embedding for a user.
        This is the 'context vector' that agents receive.
        """
        return self.user_embeddings[user_id].copy()

    def get_ad_embedding(self, arm_index: int) -> np.ndarray:
        """Get the pre-computed embedding for an ad by its arm index."""
        ad_id = self._active_ad_ids[arm_index]
        return self.ad_embeddings[ad_id].copy()
