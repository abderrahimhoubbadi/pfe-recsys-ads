"""
Offline Validation on Real-World Data — Open Bandit Dataset (ZOZO).

Evaluates FOUR policies via IPS/SNIPS on the RANDOM logging policy logs:
  1. Random        — uniform baseline (CTR ≈ 0.35%)
  2. Bernoulli TS  — ZOZO's production policy (published CTR)
  3. H-LinUCB × ε-Constraint     — analytical hybrid
  4. H-DeepBandit × ε-Constraint  — neural hybrid (our champion)

Why Random logs?
  The Random policy's propensity scores are uniform (p = 1/K) and exactly
  known, which is the gold standard for unbiased IPS estimation (Saito et
  al. 2020). This gives our agents a ~1/K ≈ 1.25% agreement rate — small
  but sufficient for SNIPS with 1.37M events (~17K matches expected).

Usage:
    uv run python experiments/offline_obd_validation.py
    uv run python experiments/offline_obd_validation.py --quick

Output:
    metrics/obd_champion_validation.png
    metrics/obd_champion_results.json
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.obd import OBDDataLoader
from src.evaluation import IPSEvaluator
from src.agents.global_semantic_neural import GlobalSemanticDeepBandit
from src.agents.global_semantic_linucb import GlobalSemanticLinUCB
from src.policy.moo_policies import epsilon_constraint_policy

os.makedirs("metrics", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Published BTS CTR from our 10.7M-row BTS sample
BTS_CTR_OBSERVED = 0.0049  # 0.49% from our full BTS extraction


# ─────────────────────────────────────────────────────────────────────────────
# Agent Builders
# ─────────────────────────────────────────────────────────────────────────────


def build_deepbandit_agent(data_loader: OBDDataLoader) -> GlobalSemanticDeepBandit:
    """
    H-DeepBandit tuned for real-world sparse data:
      - batch_size=1024 to accumulate enough positive examples before update
      - lr=1e-4 for stable convergence on noisy real data
    """
    user_dim = data_loader.context_dim
    ad_dim = data_loader.action_context_dim if data_loader.has_action_context else data_loader.n_actions

    agent = GlobalSemanticDeepBandit(
        user_dim=user_dim,
        ad_dim=ad_dim,
        alpha=1.0,
        n_ensemble=5,
        hidden_dim=64,
        lr=1e-4,
        batch_size=1024,
        objectives=["click", "revenue"],
    )

    ad_embeddings = data_loader.get_all_action_embeddings()
    agent.set_ad_embeddings(ad_embeddings)
    logger.info(
        f"[H-DeepBandit] {user_dim + ad_dim}D context, "
        f"{data_loader.n_actions} arms, batch=1024, lr=1e-4"
    )
    return agent


def build_linucb_agent(data_loader: OBDDataLoader) -> GlobalSemanticLinUCB:
    """
    H-LinUCB — analytical, Sherman-Morrison updates.
    No sparsity issues: learns from every single matched event immediately.
    """
    user_dim = data_loader.context_dim
    ad_dim = data_loader.action_context_dim if data_loader.has_action_context else data_loader.n_actions

    agent = GlobalSemanticLinUCB(
        user_dim=user_dim,
        ad_dim=ad_dim,
        alpha=1.0,
        objectives=["click", "revenue"],
    )

    ad_embeddings = data_loader.get_all_action_embeddings()
    agent.set_ad_embeddings(ad_embeddings)
    logger.info(
        f"[H-LinUCB] {user_dim + ad_dim}D context, "
        f"{data_loader.n_actions} arms, alpha=1.0"
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def generate_plot(results: dict, output_path: str = "metrics/obd_champion_validation.png"):
    """4-bar horizontal barplot: Random | BTS | H-LinUCB | H-DeepBandit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    labels = [
        "Random\n(Uniform Baseline)",
        "Bernoulli TS\n(ZOZO Production)",
        "H-LinUCB\n× ε-Constraint",
        "H-DeepBandit\n× ε-Constraint",
    ]
    values = [
        results["random_ctr"],
        results["bts_ctr"],
        results["linucb_snips"],
        results["deepbandit_snips"],
    ]
    colors = ["#4F5D75", "#2D6A9F", "#F0A500", "#E8485C"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")

    # BTS reference line
    ax.axvline(
        x=results["bts_ctr"], color="#2D6A9F", linestyle="--",
        linewidth=1.5, alpha=0.7, label="BTS production",
    )

    # Value annotations
    max_val = max(values) if max(values) > 0 else 0.01
    for bar, val in zip(bars, values):
        ax.text(
            val + max_val * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}  ({val*100:.2f}%)",
            va="center", ha="left", color="white", fontsize=10, fontweight="bold",
        )

    # Gain annotations vs BTS
    bts = results["bts_ctr"]
    for idx, key in enumerate(["linucb_snips", "deepbandit_snips"]):
        val = results[key]
        if bts > 0 and val > 0:
            gain = (val - bts) / bts * 100
            gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
            gain_color = "#2ECC71" if gain >= 0 else "#E74C3C"
            bar_idx = idx + 2
            ax.annotate(
                f"vs BTS: {gain_str}",
                xy=(val, bar_idx),
                xytext=(val + max_val * 0.06, bar_idx + 0.25),
                color=gain_color, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=gain_color, lw=1.2),
            )

    ax.set_xlabel("V̂_SNIPS  (Estimated CTR via Inverse Propensity Scoring)", color="white", fontsize=11)
    ax.set_title(
        "Validation Offline — Open Bandit Dataset (ZOZO, 1.37M events)\n"
        "Approches Hybrides Sémantiques vs Politique de Production",
        color="white", fontsize=12, fontweight="bold", pad=14,
    )
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    legend_patches = [
        mpatches.Patch(color="#4F5D75", label="Random (plancher)"),
        mpatches.Patch(color="#2D6A9F", label="Bernoulli TS — ZOZO production"),
        mpatches.Patch(color="#F0A500", label="H-LinUCB × ε-Constraint"),
        mpatches.Patch(color="#E8485C", label="H-DeepBandit × ε-Constraint"),
    ]
    ax.legend(
        handles=legend_patches, loc="lower right", fontsize=8,
        facecolor="#161B22", edgecolor="#30363D", labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Offline OBD validation — 4-way comparison"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on first 10000 events (quick sanity check)",
    )
    parser.add_argument(
        "--data", default="data/obd/obd_bandit_feedback.npz",
        help="Path to the OBD random .npz file",
    )
    parser.add_argument(
        "--w_max", type=float, default=100.0,
        help="IPS weight clipping threshold",
    )
    args = parser.parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Offline OBD Validation — 4-Way Comparison")
    logger.info("=" * 60)

    data_loader = OBDDataLoader(data_path=args.data)
    logger.info(f"[data] {data_loader}")

    # Compute baseline BEFORE truncation
    full_random_ctr = data_loader.baseline_ctr

    if args.quick:
        N = 10000
        logger.info(f"[data] --quick mode: first {N} events")
        data_loader._context = data_loader._context[:N]
        data_loader._action  = data_loader._action[:N]
        data_loader._reward  = data_loader._reward[:N]
        data_loader._pscore  = data_loader._pscore[:N]

    # ── 2. Baselines ─────────────────────────────────────────────────────────
    logger.info("\n[1/4] Random policy ...")
    random_ctr = full_random_ctr
    logger.info(f"      Random CTR = {random_ctr:.6f} ({random_ctr*100:.3f}%)")

    logger.info("\n[2/4] Bernoulli TS (published) ...")
    bts_ctr = BTS_CTR_OBSERVED
    logger.info(f"      BTS CTR = {bts_ctr:.6f} ({bts_ctr*100:.3f}%)")

    # ── 3. Policy ────────────────────────────────────────────────────────────
    policy = epsilon_constraint_policy(
        primary_objective="click",
        constraint_objective="revenue",
        epsilon=0.0,
        use_conservative_constraint=False,
    )
    evaluator = IPSEvaluator(w_max=args.w_max)

    # ── 4. H-LinUCB ──────────────────────────────────────────────────────────
    logger.info("\n[3/4] H-LinUCB × ε-Constraint ...")
    linucb_agent = build_linucb_agent(data_loader)
    linucb_result = evaluator.evaluate(
        agent=linucb_agent, policy_fn=policy, data_loader=data_loader,
        update_agent=True, verbose=True, log_interval=50000,
    )
    logger.info(
        f"      SNIPS={linucb_result['snips_value']:.6f}  "
        f"agreement={linucb_result['agreement_rate']*100:.2f}%  "
        f"matched={linucb_result['n_matched']:,}"
    )

    # ── 5. H-DeepBandit ──────────────────────────────────────────────────────
    logger.info("\n[4/4] H-DeepBandit × ε-Constraint ...")
    deepbandit_agent = build_deepbandit_agent(data_loader)
    deepbandit_result = evaluator.evaluate(
        agent=deepbandit_agent, policy_fn=policy, data_loader=data_loader,
        update_agent=True, verbose=True, log_interval=50000,
    )
    logger.info(
        f"      SNIPS={deepbandit_result['snips_value']:.6f}  "
        f"agreement={deepbandit_result['agreement_rate']*100:.2f}%  "
        f"matched={deepbandit_result['n_matched']:,}"
    )

    # ── 6. Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Random          : {random_ctr:.6f} ({random_ctr*100:.3f}%)")
    logger.info(f"  Bernoulli TS    : {bts_ctr:.6f} ({bts_ctr*100:.3f}%)")
    logger.info(f"  H-LinUCB SNIPS  : {linucb_result['snips_value']:.6f}")
    logger.info(f"  H-DeepBandit SNIPS: {deepbandit_result['snips_value']:.6f}")

    for name, snips in [("H-LinUCB", linucb_result["snips_value"]),
                        ("H-DeepBandit", deepbandit_result["snips_value"])]:
        if bts_ctr > 0 and snips > 0:
            gain = (snips - bts_ctr) / bts_ctr * 100
            logger.info(f"  {name} vs BTS: {gain:+.1f}%")

    # ── 7. Save ──────────────────────────────────────────────────────────────
    results = {
        "random_ctr": random_ctr,
        "bts_ctr": bts_ctr,
        "linucb_ips": linucb_result["ips_value"],
        "linucb_snips": linucb_result["snips_value"],
        "linucb_agreement": linucb_result["agreement_rate"],
        "linucb_matched": linucb_result["n_matched"],
        "deepbandit_ips": deepbandit_result["ips_value"],
        "deepbandit_snips": deepbandit_result["snips_value"],
        "deepbandit_agreement": deepbandit_result["agreement_rate"],
        "deepbandit_matched": deepbandit_result["n_matched"],
        "n_total": data_loader.n_events,
        "w_max": args.w_max,
        "dataset_n_events": data_loader.n_events,
        "dataset_n_actions": data_loader.n_actions,
        "dataset_context_dim": data_loader.context_dim,
    }
    with open("metrics/obd_champion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n[save] Results → metrics/obd_champion_results.json")

    generate_plot(results)
    logger.info("[done] ✅")


if __name__ == "__main__":
    main()
