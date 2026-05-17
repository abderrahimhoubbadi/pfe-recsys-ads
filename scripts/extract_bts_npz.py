"""
Extract the BTS logs into data/obd/obd_bts_feedback.npz.

Memory-efficient: processes the CSV in chunks to avoid OOM on WSL.
The BTS CSV has variable propensity scores (unlike random's uniform 1/K),
which is exactly what makes IPS meaningful.

Usage:
    uv run python scripts/extract_bts_npz.py
"""

import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# CSV lives on Windows host (outside WSL workspace to avoid file watcher crash)
BTS_CSV = Path("/mnt/c/Users/LEGION/OneDrive - EMI University/Desktop/bts_all.csv")
OUT_NPZ  = Path("data/obd/obd_bts_feedback.npz")
OUT_JSON = Path("data/obd/obd_bts_stats.json")
CHUNK_SIZE = 200_000  # rows per chunk — keeps memory under ~2GB


def extract_bts():
    print(f"[BTS] Reading {BTS_CSV}")
    print(f"[BTS] Processing in chunks of {CHUNK_SIZE:,} rows ...")

    # ------------------------------------------------------------------
    # Pass 1: scan all item_ids to build a consistent LabelEncoder,
    #          and identify user_feature / affinity columns
    # ------------------------------------------------------------------
    print("[BTS] Pass 1 — scanning item_ids and columns ...")
    all_item_ids = set()
    cols = None
    n_rows = 0
    for chunk in pd.read_csv(BTS_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        all_item_ids.update(chunk["item_id"].unique())
        n_rows += len(chunk)
        if cols is None:
            cols = chunk.columns.tolist()
        print(f"  scan: {n_rows:,} rows", end="\r")
    print(f"\n[BTS] Total rows: {n_rows:,}")

    # Build label encoder for item_id
    item_le = LabelEncoder()
    item_le.fit(sorted(all_item_ids))
    n_actions = len(all_item_ids)
    print(f"[BTS] n_actions (unique items): {n_actions}")

    # Identify feature columns
    user_feat_cols = sorted([c for c in cols if c.startswith("user_feature_")])
    affinity_cols  = sorted([c for c in cols if c.startswith("user-item_affinity_")])
    print(f"[BTS] user_feature cols: {len(user_feat_cols)}, affinity cols: {len(affinity_cols)}")

    # ------------------------------------------------------------------
    # Pass 1b: fit OneHotEncoder on user_feature columns (categorical)
    # ------------------------------------------------------------------
    print("[BTS] Pass 1b — fitting OneHotEncoder on user features ...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", max_categories=50)
    # Collect unique values from a sample (first 500K rows is enough)
    sample_chunks = []
    sampled = 0
    for chunk in pd.read_csv(BTS_CSV, chunksize=CHUNK_SIZE, usecols=user_feat_cols, low_memory=False):
        sample_chunks.append(chunk)
        sampled += len(chunk)
        if sampled >= 500_000:
            break
    ohe.fit(pd.concat(sample_chunks, ignore_index=True))
    context_dim_user = sum(len(cats) for cats in ohe.categories_)
    context_dim = context_dim_user + len(affinity_cols)
    print(f"[BTS] context_dim: {context_dim} ({context_dim_user} OHE + {len(affinity_cols)} affinity)")
    del sample_chunks
    gc.collect()

    # ------------------------------------------------------------------
    # Pass 2: process chunks and build arrays
    # ------------------------------------------------------------------
    print("[BTS] Pass 2 — extracting features ...")

    # Pre-allocate arrays
    all_context  = np.empty((n_rows, context_dim), dtype=np.float32)
    all_action   = np.empty(n_rows, dtype=np.int32)
    all_reward   = np.empty(n_rows, dtype=np.float32)
    all_pscore   = np.empty(n_rows, dtype=np.float32)

    offset = 0
    for chunk in pd.read_csv(BTS_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        n = len(chunk)

        # action
        all_action[offset:offset+n] = item_le.transform(chunk["item_id"].values)

        # reward (click)
        all_reward[offset:offset+n] = chunk["click"].values.astype(np.float32)

        # pscore
        all_pscore[offset:offset+n] = chunk["propensity_score"].values.astype(np.float32)

        # context: OHE(user_features) + affinity
        user_ohe = ohe.transform(chunk[user_feat_cols]).astype(np.float32)
        affinity = chunk[affinity_cols].fillna(0).values.astype(np.float32)
        all_context[offset:offset+n] = np.concatenate([user_ohe, affinity], axis=1)

        offset += n
        pct = offset / n_rows * 100
        print(f"  extracted: {offset:,}/{n_rows:,} ({pct:.0f}%)", end="\r")
        gc.collect()

    print(f"\n[BTS] Extraction complete.")
    print(f"[BTS] pscore range: {all_pscore.min():.6f} – {all_pscore.max():.6f}")
    print(f"[BTS] baseline CTR: {all_reward.mean():.4f} ({all_reward.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # action_context (item features) from the OBP toy dataset
    # ------------------------------------------------------------------
    item_ctx_path = Path(".venv/lib/python3.10/site-packages/obp/dataset/obd/item_context.csv")
    if item_ctx_path.exists():
        ic = pd.read_csv(item_ctx_path, index_col=0)
        ic = ic.iloc[:n_actions]
        item_feature_0 = ic["item_feature_0"]
        other = ic.drop("item_feature_0", axis=1).apply(LabelEncoder().fit_transform)
        action_context = pd.concat([other, item_feature_0], axis=1).values.astype(np.float32)
    else:
        print("[WARN] item_context.csv not found, using identity")
        action_context = np.eye(n_actions, dtype=np.float32)
    print(f"[BTS] action_context: {action_context.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"[BTS] Saving to {OUT_NPZ} ...")
    np.savez(
        OUT_NPZ,
        context=all_context,
        action=all_action,
        reward=all_reward,
        pscore=all_pscore,
        n_actions=np.array([n_actions]),
        action_context=action_context,
    )
    print(f"[BTS] Saved NPZ ({OUT_NPZ.stat().st_size / 1e9:.2f} GB)")

    stats = {
        "n_events": n_rows,
        "n_actions": n_actions,
        "context_dim": context_dim,
        "baseline_ctr": float(all_reward.mean()),
        "campaign": "all",
        "behavior_policy": "bts",
        "has_action_context": True,
        "action_context_dim": int(action_context.shape[1]),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[BTS] Stats → {OUT_JSON}")
    print(json.dumps(stats, indent=2))
    print("\n[BTS] ✅ Done!")


if __name__ == "__main__":
    extract_bts()
