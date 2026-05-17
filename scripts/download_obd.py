"""
Download and prepare the Open Bandit Dataset (ZOZO / ZOZOTOWN).

The OBD contains real interaction logs from an A/B test conducted on
ZOZOTOWN (Japan's largest fashion e-commerce platform) in November 2019.
Two behavior policies were deployed:
  - Bernoulli Thompson Sampling (BTS) : ZOZO's production recommender
  - Uniform Random                    : logging policy for OPE (used here)

We use the 'random' policy logs because its propensity scores p(a|x) = 1/K
are known exactly, which makes IPS estimation unbiased.

Note: This script applies a monkey-patch to fix a known OBP 0.4.1 bug
where pd.concat() is called with a positional axis argument (broken on
pandas >= 2.0).

Usage:
    uv run python scripts/download_obd.py
    uv run python scripts/download_obd.py --campaign all --n_events 10000

Output:
    data/obd/obd_bandit_feedback.npz
    data/obd/obd_stats.json
"""

import argparse
import json
import os
import sys

import numpy as np


def _patch_obp():
    """
    Monkey-patch OpenBanditDataset.pre_process() to fix the pandas >=2.0
    compatibility bug in OBP 0.4.1:

        pd.concat([a, b], 1)  →  TypeError: takes 1 positional argument but 2 were given
    """
    import pandas as pd
    import obp.dataset.real as _obp_real
    from sklearn.preprocessing import LabelEncoder

    def _fixed_pre_process(self):
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        item_feature_0 = self.item_context["item_feature_0"]
        item_feature_cat = self.item_context.drop("item_feature_0", axis=1).apply(
            LabelEncoder().fit_transform
        )
        # FIX: use axis=1 keyword instead of positional argument
        self.action_context = pd.concat(
            [item_feature_cat, item_feature_0], axis=1
        ).values

    _obp_real.OpenBanditDataset.pre_process = _fixed_pre_process


def download_obd(campaign: str = "all", n_events: int = None, seed: int = 42):
    try:
        _patch_obp()
        from obp.dataset import OpenBanditDataset
    except ImportError:
        print("ERROR: 'obp' package not installed. Run: uv add obp")
        sys.exit(1)

    os.makedirs("data/obd", exist_ok=True)

    print(
        f"[OBD] Loading from local path 'data/obd' (campaign='{campaign}', behavior_policy='bts')..."
    )
    from pathlib import Path

    dataset = OpenBanditDataset(
        behavior_policy="bts", campaign=campaign, data_path=Path("data/obd")
    )

    print("[OBD] Loading bandit feedback...")
    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    context = bandit_feedback["context"]  # (n, d)
    action = bandit_feedback["action"]  # (n,)
    reward = bandit_feedback["reward"]  # (n,)
    pscore = bandit_feedback["pscore"]  # (n,)
    n_actions = int(bandit_feedback["n_actions"])
    n_rounds = len(reward)

    # action_context may not exist in all OBD versions
    action_context = bandit_feedback.get("action_context", None)

    # Optional subsample
    if n_events is not None and n_events < n_rounds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_rounds, size=n_events, replace=False)
        idx.sort()
        context = context[idx]
        action = action[idx]
        reward = reward[idx]
        pscore = pscore[idx]
        n_rounds = n_events
        print(f"[OBD] Subsampled to {n_rounds} events.")

    # Validate propensity scores
    assert pscore.min() > 0, "Propensity scores must be strictly positive!"
    expected_pscore = 1.0 / n_actions
    print(
        f"[OBD] pscore: min={pscore.min():.6f}  max={pscore.max():.6f}  expected=1/K={expected_pscore:.6f}"
    )

    # Save
    save_path = "data/obd/obd_bts_feedback.npz"
    save_dict = dict(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
        n_actions=np.array([n_actions]),
    )
    if action_context is not None:
        save_dict["action_context"] = action_context
        print(f"[OBD] action_context shape: {action_context.shape}")

    np.savez(save_path, **save_dict)

    baseline_ctr = float(reward.mean())
    stats = {
        "n_events": n_rounds,
        "n_actions": n_actions,
        "context_dim": int(context.shape[1]),
        "baseline_ctr": baseline_ctr,
        "campaign": campaign,
        "behavior_policy": "bts",
        "has_action_context": action_context is not None,
        "action_context_dim": int(action_context.shape[1])
        if action_context is not None
        else 0,
    }
    with open("data/obd/obd_bts_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n[OBD] ✅ Dataset ready:")
    print(f"  Path      : {save_path}")
    print(f"  Events    : {n_rounds:,}")
    print(f"  Items (K) : {n_actions}")
    print(f"  Features  : {context.shape[1]}")
    print(f"  CTR base  : {baseline_ctr:.4f} ({baseline_ctr * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Open Bandit Dataset (ZOZO)")
    parser.add_argument(
        "--campaign", default="all", help="Campaign: 'all', 'men', 'women'"
    )
    parser.add_argument(
        "--n_events",
        type=int,
        default=None,
        help="Subsample to N events (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download_obd(campaign=args.campaign, n_events=args.n_events, seed=args.seed)
