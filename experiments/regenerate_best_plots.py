"""
Regenerate all 4 key plots using the BEST policy per agent.

For trajectory plots: re-runs each agent with its own best-engagement policy.
For Pareto plots: uses best (eng, rev) points from the saved CSV data.
All plots annotate the policy used for each agent.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, logging

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.semantic_env.text_dataset_loader import TextDatasetLoader
from src.env.semantic_env.semantic_reward_simulator import SemanticRewardSimulator
from src.llm.sentence_transformer_client import SentenceTransformerClient
from experiments.mega_semantic_comparison import (
    create_agent,
    get_policies,
    run_combination,
    expand_classical,
    PURE_LLM_AGENTS,
)

os.makedirs("metrics", exist_ok=True)

# ── Load saved matrices ──
eng = pd.read_csv("metrics/eng_matrix.csv", index_col=0)
rev = pd.read_csv("metrics/rev_matrix.csv", index_col=0)
time_df = pd.read_csv("metrics/time_matrix.csv", index_col=0)

agents = eng.index.tolist()
policies_names = eng.columns.tolist()

classical = agents[:7]
hybrid = agents[7:14]

# ── Determine best ENGAGEMENT policy per agent ──
best_eng_policy = {}
for ag in agents:
    if ag in PURE_LLM_AGENTS:
        best_eng_policy[ag] = "Scalar"
    else:
        best_eng_policy[ag] = eng.loc[ag].idxmax()

print("Best Engagement Policy per Agent:")
for ag, pol in best_eng_policy.items():
    if ag not in PURE_LLM_AGENTS:
        print(f"  {ag:<16} → {pol:<12} (Eng={eng.loc[ag, pol]:.4f})")

# ── Setup Environment ──
print("\nLoading environment...")
dataset = TextDatasetLoader(cold_start_ratio=0.2, seed=42)
encoder = SentenceTransformerClient("all-MiniLM-L6-v2")
emb_dim = encoder.get_dimension()
n_arms_initial = dataset.get_n_known_arms()

user_embeddings = {}
for user in dataset.user_profiles:
    text = dataset.get_user_text(user)
    user_embeddings[user["id"]] = encoder.get_embedding(text)

n_iter = 5000
shock_at = 2500

# ── Re-run trajectories with best policy per agent ──
trajectory_agents = [
    "LinUCB",
    "Thompson",
    "NeuralUCB",
    "DeepBandit",
    "H-LinUCB",
    "H-Thompson",
    "H-NeuralUCB",
    "H-DeepBandit",
]

all_policies = get_policies()
trajectories = {}

print(f"\nRunning trajectories with BEST policy per agent...")
for ag_name in trajectory_agents:
    pol_name = best_eng_policy[ag_name]
    policy_fn = all_policies[pol_name]
    print(f"  {ag_name:<14} × {pol_name:<12}", end=" ", flush=True)
    try:
        env = SemanticRewardSimulator(
            dataset, embedding_model="all-MiniLM-L6-v2", seed=42
        )
        agent, is_sem, is_llm = create_agent(ag_name, n_arms_initial, emb_dim, env)
        result = run_combination(
            agent,
            is_sem,
            is_llm,
            policy_fn,
            env,
            dataset,
            user_embeddings,
            n_iter,
            shock_at,
            emb_dim,
            track_trajectory=True,
        )
        trajectories[ag_name] = {**result, "policy": pol_name}
        print(f"Eng={result['engagement']:.4f}  Rev={result['revenue']:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")

# Save updated trajectories
with open("metrics/trajectories_best_policy.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in trajectories.items()}, f)

# ═══════════════════════════════════════════════════════════════
# Style definitions
# ═══════════════════════════════════════════════════════════════
traj_colors = {
    "LinUCB": "#1f77b4",
    "Thompson": "#ff7f0e",
    "NeuralUCB": "#2ca02c",
    "DeepBandit": "#d62728",
    "H-LinUCB": "#1f77b4",
    "H-Thompson": "#ff7f0e",
    "H-NeuralUCB": "#2ca02c",
    "H-DeepBandit": "#d62728",
}
classical_styles = {"linestyle": "-", "alpha": 0.7}
hybrid_styles = {"linestyle": "--", "linewidth": 2.5, "alpha": 0.95}

# ═══════════════════════════════════════════════════════════════
# PLOT 1: Cumulative Engagement Trajectory (best policy per agent)
# ═══════════════════════════════════════════════════════════════
print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(14, 7))
for ag_name, traj in trajectories.items():
    cum_eng = np.cumsum(traj["eng_trajectory"])
    style = hybrid_styles if ag_name.startswith("H-") else classical_styles
    pol = traj["policy"]
    ax.plot(cum_eng, color=traj_colors[ag_name], label=f"{ag_name} ({pol})", **style)

ax.axvline(
    x=shock_at,
    color="red",
    linestyle=":",
    linewidth=2,
    label=f"Cold-Start Shock (t={shock_at})",
)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cumulative Engagement", fontsize=12)
ax.set_title(
    "Cumulative Engagement — Best Policy per Agent\nClassical (solid) vs Hybrid (dashed)",
    fontsize=13,
)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metrics/trajectory_cumulative_engagement.png", dpi=150)
plt.close()
print("  ✅ trajectory_cumulative_engagement.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 2: Post-Shock Recovery (best policy per agent)
# ═══════════════════════════════════════════════════════════════
window = 100
zoom_start = max(0, shock_at - 200)
zoom_end = min(n_iter, shock_at + 800)
fig, ax = plt.subplots(figsize=(14, 7))
for ag_name, traj in trajectories.items():
    eng_arr = np.array(traj["eng_trajectory"])
    ma = np.convolve(eng_arr, np.ones(window) / window, mode="valid")
    x_range = np.arange(len(ma))
    mask = (x_range >= zoom_start) & (x_range < zoom_end)
    if np.any(mask):
        style = hybrid_styles if ag_name.startswith("H-") else classical_styles
        pol = traj["policy"]
        ax.plot(
            x_range[mask],
            ma[mask],
            color=traj_colors[ag_name],
            label=f"{ag_name} ({pol})",
            **style,
        )

ax.axvline(
    x=shock_at,
    color="red",
    linestyle=":",
    linewidth=2,
    label=f"Cold-Start Shock (t={shock_at})",
)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel(f"Engagement (Moving Avg, w={window})", fontsize=12)
ax.set_title(
    "Post-Shock Recovery — Best Policy per Agent\nClassical (solid) vs Hybrid (dashed)",
    fontsize=13,
)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metrics/trajectory_post_shock_recovery.png", dpi=150)
plt.close()
print("  ✅ trajectory_post_shock_recovery.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 3: Mega Pareto — Best point per agent (annotated)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))

# For each agent, pick its best (eng+rev) point
for i, ag in enumerate(agents):
    if ag in PURE_LLM_AGENTS:
        continue  # skip LLM for clarity
    combined = eng.loc[ag] + rev.loc[ag]
    best_j = combined.idxmax()
    e = eng.loc[ag, best_j]
    r = rev.loc[ag, best_j]

    if ag in classical:
        color, marker, ms = "#2196F3", "o", 140
    elif ag in hybrid:
        color, marker, ms = "#4CAF50", "s", 140
    else:
        color, marker, ms = "#F44336", "*", 200

    ax.scatter(
        e, r, c=color, marker=marker, s=ms, edgecolors="black", linewidths=0.8, zorder=5
    )
    ax.annotate(
        f"{ag}\n({best_j})",
        (e, r),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=8,
        alpha=0.9,
    )

# legend patches
from matplotlib.lines import Line2D

legend_els = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#2196F3",
        markersize=10,
        label="Classical (Disjoint)",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="#4CAF50",
        markersize=10,
        label="Hybrid (Global Semantic)",
    ),
]
ax.legend(handles=legend_els, fontsize=11, loc="upper left")
ax.set_xlabel("Engagement Score", fontsize=12)
ax.set_ylabel("Revenue", fontsize=12)
ax.set_title("Pareto Front — Best Point per Agent (with Policy)", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metrics/mega_pareto.png", dpi=150)
plt.close()
print("  ✅ mega_pareto.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 4: Best-of-Class Pareto with envelopes + policy labels
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 9))

for idx_list, color, marker, family_label in [
    (classical, "#2196F3", "o", "Classical (Disjoint)"),
    (hybrid, "#4CAF50", "s", "Hybrid (Global Semantic)"),
]:
    points_eng, points_rev, labels = [], [], []
    for ag in idx_list:
        combined = eng.loc[ag] + rev.loc[ag]
        best_j = combined.idxmax()
        e = eng.loc[ag, best_j]
        r = rev.loc[ag, best_j]
        points_eng.append(e)
        points_rev.append(r)
        labels.append(f"{ag}\n({best_j})")

    ax.scatter(
        points_eng,
        points_rev,
        c=color,
        marker=marker,
        s=200,
        label=family_label,
        edgecolors="black",
        linewidths=1,
        zorder=5,
    )
    for pe, pr, lbl in zip(points_eng, points_rev, labels):
        ax.annotate(
            lbl,
            (pe, pr),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
            alpha=0.9,
        )

    # Pareto envelope
    if len(points_eng) > 1:
        pts = sorted(zip(points_eng, points_rev), key=lambda p: p[0])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            alpha=0.4,
            linewidth=2,
            linestyle="--",
        )

ax.set_xlabel("Best Engagement (across all policies)", fontsize=12)
ax.set_ylabel("Best Revenue (across all policies)", fontsize=12)
ax.set_title(
    "Best-of-Class Pareto — Which Family Reaches Higher?\n(annotated with best policy)",
    fontsize=13,
)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metrics/best_of_class_pareto.png", dpi=150)
plt.close()
print("  ✅ best_of_class_pareto.png")

print("\n✅ All 4 plots updated!")
