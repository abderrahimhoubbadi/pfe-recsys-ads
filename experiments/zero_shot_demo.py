"""
Zero-Shot Transfer Demonstration Plot.

Runs LinUCB (disjoint) vs H-LinUCB (global semantic) with SAME policy (Scalar)
to clearly isolate the architectural difference.

Shows:
- Before shock: both learn at similar rates
- At shock (t=2500): 12 new ads injected
- After shock: H-LinUCB immediately generalizes to new ads (zero-shot),
  while LinUCB must explore from scratch
"""

import sys, os, json, time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, logging

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.semantic_env.text_dataset_loader import TextDatasetLoader
from src.env.semantic_env.semantic_reward_simulator import SemanticRewardSimulator
from src.llm.sentence_transformer_client import SentenceTransformerClient
from experiments.mega_semantic_comparison import (
    create_agent,
    run_combination,
)
from src.policy.moo_policies import linear_scalarization_policy

os.makedirs("metrics", exist_ok=True)

# ── Setup ──
print("Loading environment...")
dataset = TextDatasetLoader(cold_start_ratio=0.2, seed=42)
encoder = SentenceTransformerClient("all-MiniLM-L6-v2")
emb_dim = encoder.get_dimension()
n_arms = dataset.get_n_known_arms()

user_embeddings = {}
for user in dataset.user_profiles:
    text = dataset.get_user_text(user)
    user_embeddings[user["id"]] = encoder.get_embedding(text)

n_iter = 5000
shock_at = 2500
scalar = linear_scalarization_policy({"click": 0.5, "revenue": 0.5})

# ── Run the two agents ──
results = {}
for name in ["LinUCB", "H-LinUCB"]:
    print(f"  Running {name} × Scalar...")
    env = SemanticRewardSimulator(dataset, embedding_model="all-MiniLM-L6-v2", seed=42)
    agent, is_sem, is_llm = create_agent(name, n_arms, emb_dim, env)
    result = run_combination(
        agent,
        is_sem,
        is_llm,
        scalar,
        env,
        dataset,
        user_embeddings,
        n_iter,
        shock_at,
        emb_dim,
        track_trajectory=True,
    )
    results[name] = result
    print(f"    Eng={result['engagement']:.4f}  Rev={result['revenue']:.4f}")

# ═══════════════════════════════════════════════════════════════
# PLOT: Clean Zero-Shot Transfer Demonstration
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

# ── Top panel: Moving average engagement ──
ax = axes[0]
window = 150

for name, color, ls, lw in [
    ("LinUCB", "#2196F3", "-", 2.5),
    ("H-LinUCB", "#4CAF50", "-", 2.5),
]:
    eng_arr = np.array(results[name]["eng_trajectory"])
    ma = np.convolve(eng_arr, np.ones(window) / window, mode="valid")
    x = np.arange(len(ma))
    label_text = (
        f"{name} ({'Disjoint per-arm' if name == 'LinUCB' else 'Global Semantic'})"
    )
    ax.plot(x, ma, color=color, linewidth=lw, linestyle=ls, label=label_text)

# Shock line
ax.axvline(x=shock_at, color="red", linewidth=2, linestyle="--", alpha=0.8)

# Annotate the shock zone
ax.axvspan(shock_at, shock_at + 500, color="red", alpha=0.06)

# Get engagement values right after shock for annotation
eng_linucb = np.array(results["LinUCB"]["eng_trajectory"])
eng_hlinucb = np.array(results["H-LinUCB"]["eng_trajectory"])
ma_linucb = np.convolve(eng_linucb, np.ones(window) / window, mode="valid")
ma_hlinucb = np.convolve(eng_hlinucb, np.ones(window) / window, mode="valid")

# Find post-shock values (at t=2700 i.e. 200 iters after shock)
t_post = shock_at + 200
if t_post < len(ma_linucb):
    val_c = ma_linucb[t_post]
    val_h = ma_hlinucb[t_post]
    gap = val_h - val_c

    # Draw the gap annotation
    ax.annotate(
        "",
        xy=(t_post, val_h),
        xytext=(t_post, val_c),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )
    ax.text(
        t_post + 50,
        (val_c + val_h) / 2,
        f"Zero-Shot Gap\nΔ = {gap:.3f}",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
    )

# Cold-Start annotation
ax.annotate(
    "12 new ads injected\n(Cold-Start Shock)",
    xy=(shock_at, ax.get_ylim()[1] * 0.95),
    xytext=(shock_at - 600, ax.get_ylim()[1] * 0.98),
    fontsize=10,
    color="red",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
)

# Explanation box in the post-shock zone
ax.text(
    shock_at + 600,
    ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15,
    "H-LinUCB uses semantic similarity\n"
    "to estimate new arm rewards\n"
    "WITHOUT any exploration needed\n"
    "(zero-shot transfer)",
    fontsize=9,
    color="#2E7D32",
    fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.8),
)

ax.text(
    shock_at + 600,
    ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
    "LinUCB has ZERO info about new arms\n"
    "→ must explore from scratch\n"
    "→ engagement stays flat",
    fontsize=9,
    color="#1565C0",
    fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8),
)

ax.set_ylabel("Engagement (Moving Avg, w=150)", fontsize=12)
ax.set_title(
    "Zero-Shot Transfer Demonstration\nLinUCB (Disjoint) vs H-LinUCB (Global Semantic) — Same Scalar Policy",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)

# ── Bottom panel: Instantaneous difference (H-LinUCB - LinUCB) ──
ax2 = axes[1]
diff = ma_hlinucb - ma_linucb
positive = np.where(diff >= 0, diff, 0)
negative = np.where(diff < 0, diff, 0)

ax2.fill_between(
    np.arange(len(diff)),
    positive,
    0,
    color="#4CAF50",
    alpha=0.5,
    label="H-LinUCB > LinUCB",
)
ax2.fill_between(
    np.arange(len(diff)),
    negative,
    0,
    color="#F44336",
    alpha=0.5,
    label="LinUCB > H-LinUCB",
)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.axvline(x=shock_at, color="red", linewidth=2, linestyle="--", alpha=0.8)

# Shade the post-shock zone
ax2.axvspan(shock_at, n_iter, color="yellow", alpha=0.1)
ax2.text(
    shock_at + 100,
    ax2.get_ylim()[1] * 0.7 if ax2.get_ylim()[1] > 0 else 0.05,
    "Post-Shock: Hybrid advantage amplified",
    fontsize=9,
    fontweight="bold",
    color="#2E7D32",
)

ax2.set_xlabel("Iteration", fontsize=12)
ax2.set_ylabel("Δ Engagement\n(H-LinUCB − LinUCB)", fontsize=10)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("metrics/zero_shot_transfer_demo.png", dpi=150)
plt.close()
print("\n✅ metrics/zero_shot_transfer_demo.png")

print("\n✅ Done!")
