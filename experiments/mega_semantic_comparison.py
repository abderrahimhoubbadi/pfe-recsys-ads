"""
Mega Semantic Comparison — 16 Agents × 10 MOO Policies.

7 Classical (Disjoint, per-arm) + 7 Hybrid (Global Semantic) + 2 Pure LLM.
Deterministic, noiseless rewards. Cold-Start Shock at midpoint.

Usage:
    python experiments/mega_semantic_comparison.py
    python experiments/mega_semantic_comparison.py --n 5000 --shock-at 2500
"""

import sys
import os
import time
import argparse
import warnings
import logging
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import (
    ALPHA,
    EPSILON_THRESHOLD,
    HIDDEN_DIM,
    LEARNING_RATE,
    N_ENSEMBLE,
    TS_VARIANCE,
    NEURAL_TS_SIGMA,
    PESSIMISM_DECAY,
    DELAY_WINDOW,
)
from src.env.semantic_env.text_dataset_loader import TextDatasetLoader
from src.env.semantic_env.semantic_reward_simulator import SemanticRewardSimulator
from src.llm.sentence_transformer_client import SentenceTransformerClient

# Classical Agents (per-arm)
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent
from src.agents.thompson_sampling_agent import ThompsonSamplingAgent
from src.agents.neural_ucb_agent import NeuralUCBAgent
from src.agents.neural_ts_agent import NeuralTSAgent
from src.agents.deep_bandit_agent import DeepBanditAgent
from src.agents.offline_online_agent import OfflineOnlineAgent
from src.agents.delayed_feedback_agent import DelayedFeedbackAgent

# Hybrid Global Semantic Agents
from src.agents.global_semantic_linucb import GlobalSemanticLinUCB
from src.agents.global_semantic_neural import (
    GlobalSemanticNeuralUCB,
    GlobalSemanticNeuralTS,
    GlobalSemanticDeepBandit,
)
from src.agents.global_semantic_others import (
    GlobalSemanticThompson,
    GlobalSemanticOfflineOnline,
    GlobalSemanticDelayedFeedback,
)

# Pure LLM Agents
from src.agents.llm_agents.deep_think_agent import GeminiDeepThinkAgent
from src.agents.llm_agents.pro_agent import GeminiProAgent

# MOO Policies
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy,
)
from src.policy.exact_moo import (
    mobb_policy,
    two_phase_policy,
    oss_policy,
    modp_policy,
    moa_star_policy,
)
from src.policy.metaheuristics import nsga2_policy, moead_policy

os.makedirs("metrics", exist_ok=True)

PURE_LLM_AGENTS = ["LlamaReasoning", "LlamaInstruct"]


def get_policies():
    return {
        "Scalar": linear_scalarization_policy({"click": 0.5, "revenue": 0.5}),
        "ε-Constr": epsilon_constraint_policy(
            "click", "revenue", epsilon=EPSILON_THRESHOLD
        ),
        "Pareto-Ch": pareto_frontier_policy(),
        "MOBB": mobb_policy(),
        "TwoPhase": two_phase_policy(),
        "OSS": oss_policy(),
        "MODP": modp_policy(),
        "MOA*": moa_star_policy(),
        "NSGA-II": nsga2_policy(),
        "MOEA/D": moead_policy(),
    }


def _load_ad_embeddings(env, n_arms):
    """Load semantic ad embeddings for hybrid agents."""
    return {idx: env.get_ad_embedding(idx) for idx in range(n_arms)}


def create_agent(name, n_arms, emb_dim, env):
    """
    Create a fresh agent.
    Returns: (agent, is_semantic, is_pure_llm)
    """
    # ── Classical (Disjoint, per-arm) ──
    if name == "LinUCB":
        return (
            MultiObjectiveLinUCBAgent(n_arms=n_arms, dimension=emb_dim, alpha=ALPHA),
            False,
            False,
        )
    elif name == "Thompson":
        return (
            ThompsonSamplingAgent(n_arms=n_arms, dimension=emb_dim, v=TS_VARIANCE),
            False,
            False,
        )
    elif name == "NeuralUCB":
        return (
            NeuralUCBAgent(
                n_arms=n_arms,
                dimension=emb_dim,
                alpha=ALPHA,
                hidden_dim=HIDDEN_DIM,
                lr=LEARNING_RATE,
            ),
            False,
            False,
        )
    elif name == "NeuralTS":
        return (
            NeuralTSAgent(
                n_arms=n_arms,
                dimension=emb_dim,
                sigma=NEURAL_TS_SIGMA,
                hidden_dim=HIDDEN_DIM,
                lr=LEARNING_RATE,
            ),
            False,
            False,
        )
    elif name == "DeepBandit":
        return (
            DeepBanditAgent(
                n_arms=n_arms,
                dimension=emb_dim,
                alpha=ALPHA,
                n_ensemble=N_ENSEMBLE,
                hidden_dim=HIDDEN_DIM,
                lr=LEARNING_RATE,
            ),
            False,
            False,
        )
    elif name == "Offline2On":
        return (
            OfflineOnlineAgent(
                n_arms=n_arms,
                dimension=emb_dim,
                alpha=ALPHA,
                pessimism_decay=PESSIMISM_DECAY,
            ),
            False,
            False,
        )
    elif name == "DelayedFB":
        return (
            DelayedFeedbackAgent(
                n_arms=n_arms, dimension=emb_dim, alpha=ALPHA, delay_window=DELAY_WINDOW
            ),
            False,
            False,
        )

    # ── Hybrid (Global Semantic, single model) ──
    elif name == "H-LinUCB":
        agent = GlobalSemanticLinUCB(user_dim=emb_dim, ad_dim=emb_dim, alpha=ALPHA)
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-Thompson":
        agent = GlobalSemanticThompson(user_dim=emb_dim, ad_dim=emb_dim, v=TS_VARIANCE)
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-NeuralUCB":
        agent = GlobalSemanticNeuralUCB(
            user_dim=emb_dim,
            ad_dim=emb_dim,
            alpha=ALPHA,
            hidden_dim=HIDDEN_DIM,
            lr=LEARNING_RATE,
        )
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-NeuralTS":
        agent = GlobalSemanticNeuralTS(
            user_dim=emb_dim,
            ad_dim=emb_dim,
            sigma=NEURAL_TS_SIGMA,
            hidden_dim=HIDDEN_DIM,
            lr=LEARNING_RATE,
        )
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-DeepBandit":
        agent = GlobalSemanticDeepBandit(
            user_dim=emb_dim,
            ad_dim=emb_dim,
            alpha=ALPHA,
            n_ensemble=N_ENSEMBLE,
            hidden_dim=HIDDEN_DIM,
            lr=LEARNING_RATE,
        )
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-Offline2On":
        agent = GlobalSemanticOfflineOnline(
            user_dim=emb_dim,
            ad_dim=emb_dim,
            alpha=ALPHA,
            pessimism_decay=PESSIMISM_DECAY,
        )
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False
    elif name == "H-DelayedFB":
        agent = GlobalSemanticDelayedFeedback(
            user_dim=emb_dim, ad_dim=emb_dim, alpha=ALPHA, delay_window=DELAY_WINDOW
        )
        agent.set_ad_embeddings(_load_ad_embeddings(env, n_arms))
        return agent, True, False

    # ── Pure LLM ──
    elif name == "LlamaReasoning":
        return (
            GeminiDeepThinkAgent(
                n_arms=n_arms, dimension=emb_dim, model_name="llama3.1"
            ),
            False,
            True,
        )
    elif name == "LlamaInstruct":
        return (
            GeminiProAgent(n_arms=n_arms, dimension=emb_dim, model_name="llama3.1"),
            False,
            True,
        )

    raise ValueError(f"Unknown agent: {name}")


def expand_classical(agent, old_n, new_n, dim):
    """Add blank per-arm matrices for classical agents."""
    if hasattr(agent, "A_inv") and isinstance(agent.A_inv, list):
        for _ in range(new_n - old_n):
            agent.A_inv.append(np.eye(dim))
            if hasattr(agent, "b"):
                for obj in agent.b:
                    agent.b[obj].append(
                        np.zeros((dim, 1))
                        if isinstance(agent.b[obj][0], np.ndarray)
                        and agent.b[obj][0].ndim == 2
                        else np.zeros(dim)
                    )
        agent.n_arms = new_n
        if hasattr(agent, "counts"):
            agent.counts.extend([0] * (new_n - old_n))
        if hasattr(agent, "confirmed_count") and isinstance(
            agent.confirmed_count, list
        ):
            agent.confirmed_count.extend([0] * (new_n - old_n))
            agent.total_count.extend([0] * (new_n - old_n))


def run_combination(
    agent,
    is_semantic,
    is_llm,
    policy_fn,
    env,
    dataset,
    user_embeddings,
    n_iter,
    shock_at,
    emb_dim,
    track_trajectory=False,
):
    """Run a single (agent, policy) combination."""
    actual_iter = 5 if is_llm else n_iter
    total_eng = 0.0
    total_rev = 0.0
    cold_start_triggered = False
    n_arms = env.get_n_arms()

    # Per-iteration tracking for trajectory plots
    eng_trajectory = [] if track_trajectory else None
    rev_trajectory = [] if track_trajectory else None

    start = time.time()
    for t in range(actual_iter):
        # Cold-Start Shock
        if not is_llm and t == shock_at and not cold_start_triggered:
            old_n = n_arms
            env.inject_cold_start_ads()
            n_arms = env.get_n_arms()

            if is_semantic:
                new_embs = {
                    idx: env.get_ad_embedding(idx) for idx in range(old_n, n_arms)
                }
                agent.expand_arms(new_embs)
            else:
                expand_classical(agent, old_n, n_arms, emb_dim)
            cold_start_triggered = True

        user = dataset.get_random_user()
        user_emb = user_embeddings[user["id"]]

        arm = agent.select_arm(user_emb, policy=policy_fn)
        arm = min(arm, n_arms - 1)

        reward = env.get_reward(user["id"], arm)
        agent.update(user_emb, arm, reward)

        total_eng += reward["click"]
        total_rev += reward["revenue"]

        if track_trajectory:
            eng_trajectory.append(reward["click"])
            rev_trajectory.append(reward["revenue"])

    duration = time.time() - start
    result = {
        "engagement": total_eng / actual_iter,
        "revenue": total_rev / actual_iter,
        "time_ms": (duration * 1000) / actual_iter,
        "actual_iter": actual_iter,
    }
    if track_trajectory:
        result["eng_trajectory"] = eng_trajectory
        result["rev_trajectory"] = rev_trajectory
    return result


def main():
    parser = argparse.ArgumentParser(description="Mega Semantic Comparison")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--shock-at", type=int, default=None)
    args = parser.parse_args()

    n_iter = args.n
    shock_at = args.shock_at or n_iter // 2

    print("=" * 70)
    print("MEGA SEMANTIC COMPARISON — 16 Agents × 10 Policies")
    print(f"N={n_iter}, Cold-Start Shock at {shock_at}")
    print("Reward mode: DETERMINISTIC (no noise)")
    print("=" * 70)

    # ── 1. Load Environment ──
    print("\n[1/4] Loading semantic environment...")
    dataset = TextDatasetLoader(cold_start_ratio=0.2, seed=42)
    encoder = SentenceTransformerClient("all-MiniLM-L6-v2")
    emb_dim = encoder.get_dimension()
    n_arms_initial = dataset.get_n_known_arms()

    user_embeddings = {}
    for user in dataset.user_profiles:
        text = dataset.get_user_text(user)
        user_embeddings[user["id"]] = encoder.get_embedding(text)

    print(f"   Embedding dim: {emb_dim}, Initial arms: {n_arms_initial}")

    # ── 2. Define Agents ──
    agent_names = [
        # Classical (Disjoint)
        "LinUCB",
        "Thompson",
        "NeuralUCB",
        "NeuralTS",
        "DeepBandit",
        "Offline2On",
        "DelayedFB",
        # Hybrid (Global Semantic)
        "H-LinUCB",
        "H-Thompson",
        "H-NeuralUCB",
        "H-NeuralTS",
        "H-DeepBandit",
        "H-Offline2On",
        "H-DelayedFB",
        # Pure LLM
        "LlamaReasoning",
        "LlamaInstruct",
    ]

    policies = get_policies()
    policy_names = list(policies.keys())
    total = len(agent_names) * len(policy_names)

    print(
        f"\n[2/4] {len(agent_names)} agents × {len(policy_names)} policies = {total} combinations"
    )

    # ── 3. Run All Combinations ──
    print(f"\n[3/4] Running {total} combinations...\n")

    eng_matrix = np.full((len(agent_names), len(policy_names)), np.nan)
    rev_matrix = np.full((len(agent_names), len(policy_names)), np.nan)
    time_matrix = np.full((len(agent_names), len(policy_names)), np.nan)

    count = 0
    for i, ag_name in enumerate(agent_names):
        for j, pol_name in enumerate(policy_names):
            # LLM agents ignore MOO policy; copy first result
            if ag_name in PURE_LLM_AGENTS and pol_name != "Scalar":
                eng_matrix[i, j] = eng_matrix[i, 0]
                rev_matrix[i, j] = rev_matrix[i, 0]
                time_matrix[i, j] = time_matrix[i, 0]
                count += 1
                continue

            count += 1
            print(
                f"  [{count:>3}/{total}] {ag_name:<14} × {pol_name:<10}",
                end=" ",
                flush=True,
            )

            try:
                env = SemanticRewardSimulator(
                    dataset, embedding_model="all-MiniLM-L6-v2", seed=42
                )
                agent, is_sem, is_llm = create_agent(
                    ag_name, n_arms_initial, emb_dim, env
                )

                result = run_combination(
                    agent,
                    is_sem,
                    is_llm,
                    policies[pol_name],
                    env,
                    dataset,
                    user_embeddings,
                    n_iter,
                    shock_at,
                    emb_dim,
                )
                eng_matrix[i, j] = result["engagement"]
                rev_matrix[i, j] = result["revenue"]
                time_matrix[i, j] = result["time_ms"]
                print(
                    f"Eng={result['engagement']:.4f}  Rev={result['revenue']:.4f}  "
                    f"T={result['time_ms']:.1f}ms  ({result['actual_iter']} iter)"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    # ── 4. Trajectory Runs (representative agents × Scalar policy) ──
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
    scalar_policy = linear_scalarization_policy({"click": 0.5, "revenue": 0.5})
    trajectories = {}  # agent_name -> {eng_trajectory, rev_trajectory}

    print(f"\n[4/6] Running trajectory tracking for {len(trajectory_agents)} agents...")
    for ag_name in trajectory_agents:
        print(f"  Trajectory: {ag_name:<14}", end=" ", flush=True)
        try:
            env = SemanticRewardSimulator(
                dataset, embedding_model="all-MiniLM-L6-v2", seed=42
            )
            agent, is_sem, is_llm = create_agent(ag_name, n_arms_initial, emb_dim, env)
            result = run_combination(
                agent,
                is_sem,
                is_llm,
                scalar_policy,
                env,
                dataset,
                user_embeddings,
                n_iter,
                shock_at,
                emb_dim,
                track_trajectory=True,
            )
            trajectories[ag_name] = result
            print(f"Eng={result['engagement']:.4f}  Rev={result['revenue']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── 5. Save Results to Disk ──
    print(f"\n[5/7] Saving results to CSV/JSON...")
    import pandas as pd
    import json

    pd.DataFrame(eng_matrix, index=agent_names, columns=policy_names).to_csv(
        "metrics/eng_matrix.csv"
    )
    pd.DataFrame(rev_matrix, index=agent_names, columns=policy_names).to_csv(
        "metrics/rev_matrix.csv"
    )
    pd.DataFrame(time_matrix, index=agent_names, columns=policy_names).to_csv(
        "metrics/time_matrix.csv"
    )

    with open("metrics/trajectories.json", "w") as f:
        json.dump(trajectories, f)
    print("   ✅ metrics/ saved to csv and json")

    # ── 6. Generate Plots ──
    print(f"\n[6/7] Generating plots...")

    def draw_heatmap(matrix, title, cmap, fname, label):
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(policy_names)))
        ax.set_yticks(range(len(agent_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(agent_names, fontsize=9)
        ax.set_title(title, fontsize=14, pad=15)

        # Draw separator lines between Classical / Hybrid / LLM
        ax.axhline(y=6.5, color="white", linewidth=3)
        ax.axhline(y=13.5, color="white", linewidth=3)

        for ii in range(len(agent_names)):
            for jj in range(len(policy_names)):
                val = matrix[ii, jj]
                if not np.isnan(val):
                    ax.text(
                        jj,
                        ii,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if val < np.nanmean(matrix) else "white",
                    )
        plt.colorbar(im, ax=ax, label=label)
        plt.tight_layout()
        plt.savefig(f"metrics/{fname}", dpi=150)
        plt.close()
        print(f"   ✅ metrics/{fname}")

    draw_heatmap(
        eng_matrix,
        "Engagement Score: Classical vs Hybrid vs LLM",
        "YlGn",
        "mega_heatmap_engagement.png",
        "Engagement",
    )
    draw_heatmap(
        rev_matrix,
        "Revenue: Classical vs Hybrid vs LLM",
        "YlOrRd",
        "mega_heatmap_revenue.png",
        "Revenue",
    )
    draw_heatmap(
        time_matrix,
        "Execution Time (ms/iter): Classical vs Hybrid vs LLM",
        "Blues",
        "mega_heatmap_time.png",
        "Time (ms)",
    )

    # Pareto Front (all points)
    fig, ax = plt.subplots(figsize=(14, 9))
    colors_c = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    colors_h = [
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
    ]
    colors_l = ["#f44336", "#e91e63"]
    all_colors = colors_c + colors_h + colors_l
    markers_c = ["o"] * 7
    markers_h = ["s"] * 7
    markers_l = ["*"] * 2
    all_markers = markers_c + markers_h + markers_l

    for i, ag_name in enumerate(agent_names):
        valid = ~np.isnan(eng_matrix[i])
        if np.any(valid):
            ax.scatter(
                eng_matrix[i][valid],
                rev_matrix[i][valid],
                c=all_colors[i],
                marker=all_markers[i],
                s=120,
                label=ag_name,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
            )
    ax.set_xlabel("Engagement Score", fontsize=12)
    ax.set_ylabel("Revenue", fontsize=12)
    ax.set_title("Pareto Front: Classical ○ vs Hybrid ■ vs LLM ★", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics/mega_pareto.png", dpi=150)
    plt.close()
    print("   ✅ metrics/mega_pareto.png")

    # ── PLOT A: Cumulative Engagement Trajectory ──
    if trajectories:
        fig, ax = plt.subplots(figsize=(14, 7))
        classical_styles = {"linestyle": "-", "alpha": 0.7}
        hybrid_styles = {"linestyle": "--", "linewidth": 2.5, "alpha": 0.95}
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
        for ag_name, traj in trajectories.items():
            cum_eng = np.cumsum(traj["eng_trajectory"])
            style = hybrid_styles if ag_name.startswith("H-") else classical_styles
            ax.plot(cum_eng, color=traj_colors[ag_name], label=ag_name, **style)

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
            "Cumulative Engagement Trajectory — Classical (solid) vs Hybrid (dashed)",
            fontsize=13,
        )
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("metrics/trajectory_cumulative_engagement.png", dpi=150)
        plt.close()
        print("   ✅ metrics/trajectory_cumulative_engagement.png")

    # ── PLOT B: Post-Shock Recovery (Moving Average) ──
    if trajectories:
        window = 100
        zoom_start = max(0, shock_at - 200)
        zoom_end = min(n_iter, shock_at + 800)
        fig, ax = plt.subplots(figsize=(14, 7))
        for ag_name, traj in trajectories.items():
            eng = np.array(traj["eng_trajectory"])
            # Moving average
            ma = np.convolve(eng, np.ones(window) / window, mode="valid")
            x_range = np.arange(len(ma))
            mask = (x_range >= zoom_start) & (x_range < zoom_end)
            if np.any(mask):
                style = hybrid_styles if ag_name.startswith("H-") else classical_styles
                ax.plot(
                    x_range[mask],
                    ma[mask],
                    color=traj_colors[ag_name],
                    label=ag_name,
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
            "Post-Shock Recovery — Classical (solid) vs Hybrid (dashed)",
            fontsize=13,
        )
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("metrics/trajectory_post_shock_recovery.png", dpi=150)
        plt.close()
        print("   ✅ metrics/trajectory_post_shock_recovery.png")

    # ── PLOT C: Best-of-Class Pareto (envelope comparison) ──
    classical_idx = list(range(7))
    hybrid_idx = list(range(7, 14))

    fig, ax = plt.subplots(figsize=(10, 8))

    # For each agent, pick its best Eng and best Rev across policies
    for idx_list, color, marker, family_label in [
        (classical_idx, "#2196F3", "o", "Classical (Disjoint)"),
        (hybrid_idx, "#4CAF50", "s", "Hybrid (Global Semantic)"),
    ]:
        points_eng = []
        points_rev = []
        labels = []
        for i in idx_list:
            valid = ~np.isnan(eng_matrix[i])
            if np.any(valid):
                # Best (eng, rev) point = the one that maximizes eng+rev
                combined = eng_matrix[i] + rev_matrix[i]
                best_j = np.nanargmax(combined)
                points_eng.append(eng_matrix[i, best_j])
                points_rev.append(rev_matrix[i, best_j])
                labels.append(agent_names[i])

        ax.scatter(
            points_eng,
            points_rev,
            c=color,
            marker=marker,
            s=180,
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
                xytext=(8, 5),
                fontsize=8,
                alpha=0.85,
            )

        # Draw Pareto envelope
        if len(points_eng) > 1:
            pts = sorted(zip(points_eng, points_rev), key=lambda p: p[0])
            envelope_x = [p[0] for p in pts]
            envelope_y = [p[1] for p in pts]
            ax.plot(
                envelope_x,
                envelope_y,
                color=color,
                alpha=0.4,
                linewidth=2,
                linestyle="--",
            )

    ax.set_xlabel("Best Engagement (across all policies)", fontsize=12)
    ax.set_ylabel("Best Revenue (across all policies)", fontsize=12)
    ax.set_title(
        "Best-of-Class Pareto — Which Family Reaches Higher?",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics/best_of_class_pareto.png", dpi=150)
    plt.close()
    print("   ✅ metrics/best_of_class_pareto.png")

    # ── PLOT D: Radar Chart of Maximum Capabilities ──
    categories = [
        "Max\nEngagement",
        "Max\nRevenue",
        "Speed\n(1/time)",
        "Cold-Start\nRecovery",
    ]
    n_cats = len(categories)

    def get_family_stats(idx_list, traj_names):
        max_eng = max(np.nanmax(eng_matrix[i]) for i in idx_list)
        max_rev = max(np.nanmax(rev_matrix[i]) for i in idx_list)
        avg_time = np.nanmean([np.nanmean(time_matrix[i]) for i in idx_list])
        speed = 1.0 / max(avg_time, 0.1)  # inverse time

        # Cold-start recovery: post-shock engagement vs pre-shock
        recovery = 0.0
        count = 0
        for tn in traj_names:
            if tn in trajectories:
                eng = np.array(trajectories[tn]["eng_trajectory"])
                pre = np.mean(eng[max(0, shock_at - 200) : shock_at])
                post = np.mean(eng[shock_at : min(len(eng), shock_at + 200)])
                if pre > 0:
                    recovery += post / pre
                    count += 1
        recovery = recovery / max(count, 1)
        return [max_eng, max_rev, speed, recovery]

    classical_traj = ["LinUCB", "Thompson", "NeuralUCB", "DeepBandit"]
    hybrid_traj = ["H-LinUCB", "H-Thompson", "H-NeuralUCB", "H-DeepBandit"]

    stats_c = get_family_stats(classical_idx, classical_traj)
    stats_h = get_family_stats(hybrid_idx, hybrid_traj)

    # Normalize to [0, 1] for radar
    all_vals = list(zip(stats_c, stats_h))
    max_vals = [max(c, h) * 1.15 for c, h in all_vals]  # 15% margin
    norm_c = [v / m if m > 0 else 0 for v, m in zip(stats_c, max_vals)]
    norm_h = [v / m if m > 0 else 0 for v, m in zip(stats_h, max_vals)]

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    norm_c += norm_c[:1]
    norm_h += norm_h[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, norm_c, alpha=0.15, color="#2196F3")
    ax.plot(
        angles,
        norm_c,
        color="#2196F3",
        linewidth=2,
        label="Classical (Disjoint)",
        marker="o",
        markersize=6,
    )
    ax.fill(angles, norm_h, alpha=0.15, color="#4CAF50")
    ax.plot(
        angles,
        norm_h,
        color="#4CAF50",
        linewidth=2,
        label="Hybrid (Global Semantic)",
        marker="s",
        markersize=6,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title(
        "Radar — Maximum Capabilities per Family",
        fontsize=13,
        pad=25,
    )
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig("metrics/radar_capabilities.png", dpi=150)
    plt.close()
    print("   ✅ metrics/radar_capabilities.png")

    # ── 7. Summary Table ──
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"\n{'Agent':<14} {'Best Pol(Eng)':<13} {'Eng':>8} {'Best Pol(Rev)':<13} {'Rev':>8} {'Avg ms':>8}"
    )
    print("-" * 70)
    for i, ag in enumerate(agent_names):
        valid = ~np.isnan(eng_matrix[i])
        if np.any(valid):
            be = np.nanargmax(eng_matrix[i])
            br = np.nanargmax(rev_matrix[i])
            avg_t = np.nanmean(time_matrix[i])
            print(
                f"{ag:<14} {policy_names[be]:<13} {eng_matrix[i, be]:>8.4f} "
                f"{policy_names[br]:<13} {rev_matrix[i, br]:>8.4f} {avg_t:>8.1f}"
            )

    # Delta table: Hybrid improvement over Classical
    print(f"\n{'=' * 80}")
    print("HYBRIDIZATION DELTA (Hybrid - Classical)")
    print(f"{'=' * 80}")
    classical_names = agent_names[:7]
    hybrid_names = agent_names[7:14]
    print(f"\n{'Model':<14} {'ΔEng (avg)':>12} {'ΔRev (avg)':>12} {'Status'}")
    print("-" * 55)
    for ci, (cn, hn) in enumerate(zip(classical_names, hybrid_names)):
        hi = ci + 7
        delta_eng = np.nanmean(eng_matrix[hi]) - np.nanmean(eng_matrix[ci])
        delta_rev = np.nanmean(rev_matrix[hi]) - np.nanmean(rev_matrix[ci])
        status = "✅ Hybrid wins" if delta_eng > 0 else "❌ Classical wins"
        print(f"{cn:<14} {delta_eng:>+12.4f} {delta_rev:>+12.4f} {status}")

    print(f"\n✅ All plots saved to metrics/")


if __name__ == "__main__":
    main()
