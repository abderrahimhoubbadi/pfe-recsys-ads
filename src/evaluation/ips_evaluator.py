"""
IPS / SNIPS Offline Policy Evaluator.

Implements Inverse Propensity Scoring (IPS) and Self-Normalized IPS (SNIPS)
for off-policy evaluation of bandit agents on logged interaction data.

Theory:
-------
Given logged data D = {(x_t, a_t, r_t, p_t)}_{t=1}^{n} from behavior policy π₀,
we want to estimate the value of a new policy π (our agent):

    V̂_IPS(π)  = (1/n) Σ r_t · 𝟙[π(x_t) = a_t] / p_t

    V̂_SNIPS(π) = [Σ r_t · w_t] / [Σ w_t]   where w_t = 𝟙[...] / p_t

IPS is unbiased but high-variance. SNIPS is biased but lower-variance.
We use SNIPS as the primary metric (standard in the OPE literature).

Clipped IPS: w_t = min(1/p_t, w_max) — caps extreme weights.

References:
-----------
- Horvitz & Thompson (1952)
- Strehl et al. (2010), "Learning from Logged Implicit Exploration Data"
- Saito et al. (2020), "Open Bandit Dataset and Pipeline"
"""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IPSEvaluator:
    """
    Off-Policy Evaluation via Inverse Propensity Scoring.

    Usage:
        evaluator = IPSEvaluator(w_max=100.0)
        result = evaluator.evaluate(agent, policy_fn, data_loader)
    """

    def __init__(self, w_max: float = 100.0):
        """
        Args:
            w_max: Maximum importance weight (clipping threshold).
                   Standard value in OPE literature: 100.
        """
        self.w_max = w_max

    def evaluate(
        self,
        agent,
        policy_fn: Optional[Callable],
        data_loader,
        update_agent: bool = True,
        verbose: bool = True,
        log_interval: int = 1000,
    ) -> dict:
        """
        Evaluate agent+policy on logged OBD data via IPS/SNIPS.

        Algorithm:
            For each event t:
              1. Agent predicts â_t = agent.select_arm(x_t, policy_fn)
              2. If â_t == a_t (logged action):
                   w_t = min(1/p_t, w_max)     [clipped IPS weight]
                   ips_reward_t = r_t * w_t
                   If update_agent: update agent with (x_t, â_t, r_t)
              3. If â_t ≠ a_t:
                   ips_reward_t = 0             [unobserved counterfactual]

        Args:
            agent       : Agent exposing .select_arm(context, policy_fn) -> int
            policy_fn   : MOO policy callable (None = agent's default)
            data_loader : OBDDataLoader instance
            update_agent: If True, agent learns online on matched events
            verbose     : Print progress
            log_interval: Progress log frequency

        Returns:
            dict with IPS/SNIPS estimates and diagnostic statistics
        """
        n_total = len(data_loader)

        # Accumulators
        ips_rewards = []  # full IPS reward series (0 for non-matched)
        snips_numerator = 0.0
        snips_denominator = 0.0
        n_matched = 0
        clipped_count = 0

        for t, event in enumerate(data_loader.iterate()):
            x_t = event["context"]  # (d,)
            a_t = event["logged_action"]  # int
            r_t = event["reward"]  # float 0.0 or 1.0
            p_t = event["pscore"]  # float

            # Agent action selection
            try:
                a_hat = agent.select_arm(x_t, policy_fn)
            except Exception as e:
                logger.warning(f"select_arm failed at t={t}: {e}. Skipping.")
                ips_rewards.append(0.0)
                continue

            if a_hat == a_t:
                # Clipped importance weight
                raw_w = 1.0 / max(p_t, 1e-9)
                if raw_w > self.w_max:
                    clipped_count += 1
                w_t = min(raw_w, self.w_max)

                ips_reward = r_t * w_t
                ips_rewards.append(ips_reward)
                snips_numerator += r_t * w_t
                snips_denominator += w_t
                n_matched += 1

                # Online learning on matched events
                if update_agent:
                    try:
                        agent.update(x_t, a_hat, {"click": r_t, "revenue": r_t})
                    except Exception as e:
                        logger.debug(f"update failed at t={t}: {e}")
            else:
                ips_rewards.append(0.0)

            if verbose and (t + 1) % log_interval == 0:
                pct = (t + 1) / n_total * 100
                match_rate = n_matched / (t + 1) * 100
                logger.info(
                    f"  t={t + 1:>6}/{n_total} ({pct:.1f}%)  matched={match_rate:.1f}%"
                )

        # Compute estimators
        ips_value = float(np.mean(ips_rewards)) if ips_rewards else 0.0
        snips_value = (
            snips_numerator / snips_denominator if snips_denominator > 0 else 0.0
        )
        agreement_rate = n_matched / n_total if n_total > 0 else 0.0

        return {
            "ips_value": ips_value,
            "snips_value": snips_value,
            "agreement_rate": agreement_rate,
            "n_matched": n_matched,
            "n_total": n_total,
            "clipped_count": clipped_count,
            "baseline_ctr": data_loader.baseline_ctr,
        }
