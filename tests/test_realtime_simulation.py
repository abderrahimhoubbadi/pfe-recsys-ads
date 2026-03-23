"""
Real-Time Simulation Test — Validates the closed-loop pipeline
under realistic conditions with 4 phases:

1. Warm-up     (t=0→100)   : Initial learning, 6 ads
2. Stable      (t=100→300) : Model has learned, performance plateaus
3. Cold-Start  (t=300→350) : 4 new ads injected (zero-shot test)
4. Recovery    (t=350→500) : Model recovers post-shock

Includes delayed feedback: conversions arrive 5-50 iterations after clicks.
Generates KPI evolution plots for the PFE report.
"""

import time
import random
import numpy as np
import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ── Configure matplotlib for headless ──
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════


@dataclass
class SimulationMetrics:
    """Metrics collected at each timestep."""

    t: int = 0
    phase: str = ""
    ctr_window: float = 0.0
    revenue_window: float = 0.0
    latency_ms: float = 0.0
    pending_conversions: int = 0
    selected_ad_id: int = 0
    n_arms: int = 0
    cumulative_clicks: int = 0
    cumulative_revenue: float = 0.0


@dataclass
class DelayedConversion:
    """A conversion that will arrive in the future."""

    user_text: str
    ad_id: int
    revenue: float
    arrival_t: int  # timestep when conversion arrives


# ════════════════════════════════════════════════════════════════
# User & Ad Simulation
# ════════════════════════════════════════════════════════════════

# 10 diverse user profiles
USER_PROFILES = [
    "homme 28 ans développeur passionné de technologie et de gaming",
    "femme 35 ans directrice marketing intéressée par la mode et le lifestyle",
    "étudiant 22 ans en informatique cherchant des formations en ligne",
    "couple 40 ans planifiant un voyage au Maroc pour les vacances",
    "femme 25 ans sportive passionnée de fitness et nutrition",
    "homme 50 ans chef d'entreprise intéressé par les solutions B2B",
    "adolescent 17 ans passionné de jeux vidéo et d'esports",
    "femme 30 ans mère de famille cherchant des offres pour enfants",
    "homme 45 ans investisseur intéressé par la fintech et crypto",
    "étudiant 20 ans en design graphique cherchant des outils créatifs",
]

# Initial 6 ads
INITIAL_ADS = [
    {
        "ad_id": 1,
        "title": "RTX 5090 GPU",
        "description": "La dernière carte graphique NVIDIA",
        "category": "tech",
    },
    {
        "ad_id": 2,
        "title": "Parfum Chanel N°5",
        "description": "Le parfum iconique",
        "category": "beauté",
    },
    {
        "ad_id": 3,
        "title": "Cours Python Avancé",
        "description": "Formation deep learning et IA",
        "category": "éducation",
    },
    {
        "ad_id": 4,
        "title": "Voyage Marrakech",
        "description": "Séjour tout inclus riad de luxe",
        "category": "voyage",
    },
    {
        "ad_id": 5,
        "title": "MacBook Pro M4",
        "description": "Portable Apple pour développeurs",
        "category": "tech",
    },
    {
        "ad_id": 6,
        "title": "Abonnement Netflix",
        "description": "Films et séries en streaming",
        "category": "divertissement",
    },
]

# 4 new ads for cold-start shock at t=300
COLD_START_ADS = [
    {
        "ad_id": 7,
        "title": "PS5 Pro",
        "description": "Console de jeu dernière génération",
        "category": "gaming",
    },
    {
        "ad_id": 8,
        "title": "Formation GCP Cloud",
        "description": "Certification Google Cloud Platform",
        "category": "cloud",
    },
    {
        "ad_id": 9,
        "title": "Chaussures Nike Air Max",
        "description": "Dernière collection sportswear",
        "category": "sport",
    },
    {
        "ad_id": 10,
        "title": "Investissement Crypto",
        "description": "Plateforme de trading Binance",
        "category": "finance",
    },
]


def simulate_click_probability(user_idx: int, ad_id: int) -> float:
    """
    Simulate realistic click probability based on user-ad affinity.
    Higher affinity = higher CTR.
    """
    # Affinity matrix: which user profiles like which ad categories
    affinity = {
        # user_idx: {ad_id: base_ctr_boost}
        0: {1: 0.4, 5: 0.35, 7: 0.3},  # Dev/Gamer likes tech, gaming
        1: {2: 0.5, 6: 0.2},  # Marketing dir likes beauty
        2: {3: 0.5, 8: 0.4},  # CS student likes courses, cloud
        3: {4: 0.6},  # Couple likes travel
        4: {9: 0.4},  # Sporty likes Nike
        5: {10: 0.3, 8: 0.25},  # Business likes finance, cloud
        6: {1: 0.3, 7: 0.5, 6: 0.3},  # Gamer teen likes GPU, PS5, Netflix
        7: {6: 0.3, 4: 0.2},  # Mom likes Netflix, travel
        8: {10: 0.5, 5: 0.2},  # Investor likes crypto, MacBook
        9: {5: 0.4, 3: 0.3},  # Designer likes MacBook, courses
    }
    base = 0.08  # Base CTR ~8%
    boost = affinity.get(user_idx, {}).get(ad_id, 0.0)
    return min(base + boost, 0.9)


def simulate_revenue(clicked: bool, ad_id: int) -> float:
    """Simulate revenue for a click. Higher-value ads = more revenue."""
    if not clicked:
        return 0.0
    # Revenue varies by ad category
    revenue_map = {
        1: 0.12,
        2: 0.08,
        3: 0.15,
        4: 0.20,
        5: 0.18,
        6: 0.05,
        7: 0.10,
        8: 0.22,
        9: 0.07,
        10: 0.25,
    }
    base = revenue_map.get(ad_id, 0.10)
    return base * random.uniform(0.5, 1.5)


# ════════════════════════════════════════════════════════════════
# Main Simulation
# ════════════════════════════════════════════════════════════════


def run_simulation(n_iterations: int = 500, window_size: int = 50):
    """Run the full real-time simulation."""
    from src.api.recommendation_service import RecommendationService
    from src.api.schemas import AdInfo

    print("=" * 65)
    print(" Real-Time Simulation — Closed-Loop Pipeline Validation")
    print("=" * 65)

    # ── Initialize ──
    print("\n[Init] Starting RecommendationService...")
    t0 = time.perf_counter()
    service = RecommendationService(
        embedding_model="all-MiniLM-L6-v2",
        alpha=1.0,
        n_ensemble=5,
        epsilon=0.3,
    )
    print(f"   Service ready in {time.perf_counter() - t0:.1f}s")

    # ── Register initial ads ──
    initial = [AdInfo(**ad) for ad in INITIAL_ADS]
    service.register_ads(initial)
    print(f"   Registered {len(initial)} initial ads")

    all_ads = list(initial)

    # ── Metrics buffers ──
    metrics_log: List[SimulationMetrics] = []
    ctr_window = deque(maxlen=window_size)
    rev_window = deque(maxlen=window_size)
    lat_window = deque(maxlen=window_size)
    delayed_queue: List[DelayedConversion] = []
    cumulative_clicks = 0
    cumulative_revenue = 0.0

    # ── Simulation loop ──
    print(f"\n[Sim] Running {n_iterations} iterations...\n")

    for t in range(n_iterations):
        # Determine phase
        if t < 100:
            phase = "warm-up"
        elif t < 300:
            phase = "stable"
        elif t == 300:
            phase = "cold-start"
        elif t < 350:
            phase = "shock"
        else:
            phase = "recovery"

        # ── Cold-start injection at t=300 ──
        if t == 300:
            new_ads = [AdInfo(**ad) for ad in COLD_START_ADS]
            service.register_ads(new_ads)
            all_ads.extend(new_ads)
            print(f"   ⚡ t={t}: Cold-start shock — {len(new_ads)} new ads injected!")

        # ── Pick a random user ──
        user_idx = random.randint(0, len(USER_PROFILES) - 1)
        user_text = USER_PROFILES[user_idx]

        # ── Get recommendation ──
        t_req = time.perf_counter()
        ad_id, eng, rev, latency = service.recommend(
            user_text=user_text,
            available_ads=all_ads,
        )
        latency_ms = (time.perf_counter() - t_req) * 1000

        # ── Simulate click ──
        click_prob = simulate_click_probability(user_idx, ad_id)
        clicked = random.random() < click_prob
        revenue = simulate_revenue(clicked, ad_id)

        if clicked:
            cumulative_clicks += 1

        # ── Send click feedback immediately ──
        service.process_feedback(
            user_text=user_text,
            ad_id=ad_id,
            click=clicked,
            conversion=False,
            revenue=0.0,
            feedback_type="click",
        )

        # ── Schedule delayed conversion (if clicked) ──
        if clicked and random.random() < 0.6:  # 60% of clicks convert
            delay = random.randint(5, 50)
            delayed_queue.append(
                DelayedConversion(
                    user_text=user_text,
                    ad_id=ad_id,
                    revenue=revenue,
                    arrival_t=t + delay,
                )
            )

        # ── Process arriving delayed conversions ──
        remaining = []
        for dc in delayed_queue:
            if dc.arrival_t <= t:
                service.process_feedback(
                    user_text=dc.user_text,
                    ad_id=dc.ad_id,
                    click=False,
                    conversion=True,
                    revenue=dc.revenue,
                    feedback_type="conversion",
                )
                cumulative_revenue += dc.revenue
            else:
                remaining.append(dc)
        delayed_queue = remaining

        # ── Track metrics ──
        ctr_window.append(1.0 if clicked else 0.0)
        rev_window.append(revenue)
        lat_window.append(latency_ms)

        m = SimulationMetrics(
            t=t,
            phase=phase,
            ctr_window=float(np.mean(ctr_window)),
            revenue_window=float(np.mean(rev_window)),
            latency_ms=float(np.mean(lat_window)),
            pending_conversions=len(delayed_queue),
            selected_ad_id=ad_id,
            n_arms=service.agent.n_arms,
            cumulative_clicks=cumulative_clicks,
            cumulative_revenue=cumulative_revenue,
        )
        metrics_log.append(m)

        # ── Progress ──
        if (t + 1) % 100 == 0:
            print(
                f"   t={t + 1:>3d} | phase={phase:10s} | "
                f"CTR={m.ctr_window:.3f} | Rev={m.revenue_window:.4f} | "
                f"Lat={m.latency_ms:.1f}ms | Pending={m.pending_conversions} | "
                f"Arms={m.n_arms}"
            )

    # ── Process remaining delayed conversions ──
    for dc in delayed_queue:
        service.process_feedback(
            user_text=dc.user_text,
            ad_id=dc.ad_id,
            click=False,
            conversion=True,
            revenue=dc.revenue,
            feedback_type="conversion",
        )
        cumulative_revenue += dc.revenue

    print(f"\n   ✅ Simulation complete!")
    print(
        f"   Total clicks: {cumulative_clicks}/{n_iterations} (CTR={cumulative_clicks / n_iterations:.3f})"
    )
    print(f"   Total revenue: {cumulative_revenue:.4f}")

    return metrics_log, all_ads, service


# ════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════


def generate_plots(metrics_log: List[SimulationMetrics], output_dir: str = "metrics"):
    """Generate publication-quality plots from simulation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    ts = [m.t for m in metrics_log]
    ctrs = [m.ctr_window for m in metrics_log]
    revs = [m.revenue_window for m in metrics_log]
    lats = [m.latency_ms for m in metrics_log]
    pendings = [m.pending_conversions for m in metrics_log]
    arms = [m.n_arms for m in metrics_log]
    ad_ids = [m.selected_ad_id for m in metrics_log]

    # ── Phase colors ──
    phase_colors = {
        "warm-up": "#3498db",
        "stable": "#2ecc71",
        "cold-start": "#e74c3c",
        "shock": "#e67e22",
        "recovery": "#9b59b6",
    }

    def add_phase_shading(ax):
        ax.axvspan(0, 100, alpha=0.08, color=phase_colors["warm-up"])
        ax.axvspan(100, 300, alpha=0.08, color=phase_colors["stable"])
        ax.axvspan(300, 350, alpha=0.12, color=phase_colors["shock"])
        ax.axvspan(350, 500, alpha=0.08, color=phase_colors["recovery"])
        ax.axvline(x=300, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)

    # ═══ Plot 1: KPI Evolution (4-panel) ═══
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Real-Time KPI Evolution — H-DeepBandit × ε-Constraint",
        fontweight="bold",
        fontsize=14,
    )

    # CTR
    ax = axes[0, 0]
    add_phase_shading(ax)
    ax.plot(ts, ctrs, color="#2c3e50", linewidth=1.2)
    ax.set_ylabel("CTR (fenêtre glissante)")
    ax.set_title("Engagement (CTR)")
    ax.text(
        300,
        ax.get_ylim()[1] * 0.95,
        "Cold-Start\nShock",
        ha="center",
        fontsize=8,
        color="#e74c3c",
        fontweight="bold",
    )

    # Revenue
    ax = axes[0, 1]
    add_phase_shading(ax)
    ax.plot(ts, revs, color="#27ae60", linewidth=1.2)
    ax.set_ylabel("Revenue (fenêtre glissante)")
    ax.set_title("Revenue")

    # Latency
    ax = axes[1, 0]
    add_phase_shading(ax)
    ax.plot(ts, lats, color="#e67e22", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Latence (ms)")
    ax.set_title("Latence Moyenne")

    # Pending Conversions
    ax = axes[1, 1]
    add_phase_shading(ax)
    ax.fill_between(ts, pendings, alpha=0.4, color="#8e44ad")
    ax.plot(ts, pendings, color="#8e44ad", linewidth=1.0)
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Conversions en attente")
    ax.set_title("Delayed Feedback Buffer")

    # Legend
    patches = [
        mpatches.Patch(color=c, alpha=0.3, label=l.capitalize())
        for l, c in phase_colors.items()
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(output_dir, "realtime_kpi_evolution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Saved: {path}")

    # ═══ Plot 2: Cold-Start Recovery Zoom ═══
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Cold-Start Recovery Analysis (t=250→450)", fontweight="bold", fontsize=13
    )

    shock_range = range(250, min(450, len(ts)))
    shock_ts = [ts[i] for i in shock_range]

    # CTR Zoom
    shock_ctrs = [ctrs[i] for i in shock_range]
    ax1.plot(shock_ts, shock_ctrs, color="#2c3e50", linewidth=1.5)
    ax1.axvline(
        x=300, color="#e74c3c", linestyle="--", linewidth=2, label="Cold-Start Shock"
    )
    ax1.axvspan(300, 350, alpha=0.15, color="#e74c3c")
    pre_shock_ctr = np.mean([ctrs[i] for i in range(250, 300)])
    ax1.axhline(
        y=pre_shock_ctr,
        color="#2ecc71",
        linestyle=":",
        linewidth=1,
        label=f"Pre-shock CTR={pre_shock_ctr:.3f}",
    )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("CTR")
    ax1.set_title("CTR Recovery")
    ax1.legend(fontsize=8)

    # Revenue Zoom
    shock_revs = [revs[i] for i in shock_range]
    ax2.plot(shock_ts, shock_revs, color="#27ae60", linewidth=1.5)
    ax2.axvline(
        x=300, color="#e74c3c", linestyle="--", linewidth=2, label="Cold-Start Shock"
    )
    ax2.axvspan(300, 350, alpha=0.15, color="#e74c3c")
    pre_shock_rev = np.mean([revs[i] for i in range(250, 300)])
    ax2.axhline(
        y=pre_shock_rev,
        color="#2ecc71",
        linestyle=":",
        linewidth=1,
        label=f"Pre-shock Rev={pre_shock_rev:.4f}",
    )
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Revenue")
    ax2.set_title("Revenue Recovery")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "realtime_cold_start_recovery.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Saved: {path}")

    # ═══ Plot 3: Arm Selection Distribution ═══
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        "Distribution des Annonces Sélectionnées au Fil du Temps",
        fontweight="bold",
        fontsize=13,
    )

    # Bin selections into windows
    bin_size = 25
    n_bins = len(ts) // bin_size
    unique_ads = sorted(set(ad_ids))
    ad_counts = np.zeros((len(unique_ads), n_bins))

    for b in range(n_bins):
        window_ads = ad_ids[b * bin_size : (b + 1) * bin_size]
        for ad in window_ads:
            idx = unique_ads.index(ad)
            ad_counts[idx, b] += 1
        ad_counts[:, b] /= max(sum(ad_counts[:, b]), 1)

    # Stacked bar chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ads)))
    bottom = np.zeros(n_bins)
    x_positions = [b * bin_size + bin_size // 2 for b in range(n_bins)]

    for i, ad in enumerate(unique_ads):
        ax.bar(
            x_positions,
            ad_counts[i],
            bottom=bottom,
            width=bin_size * 0.9,
            label=f"Ad #{ad}",
            color=colors[i],
            alpha=0.8,
        )
        bottom += ad_counts[i]

    ax.axvline(x=300, color="#e74c3c", linestyle="--", linewidth=2)
    ax.text(305, 0.95, "Cold-Start", fontsize=8, color="#e74c3c", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Proportion de sélection")
    ax.set_xlim(0, len(ts))
    ax.legend(fontsize=7, ncol=5, loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "realtime_arm_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Saved: {path}")


def save_csv(metrics_log: List[SimulationMetrics], output_dir: str = "metrics"):
    """Save raw metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "realtime_simulation_metrics.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "phase",
                "ctr_window",
                "revenue_window",
                "latency_ms",
                "pending_conversions",
                "selected_ad_id",
                "n_arms",
                "cumulative_clicks",
                "cumulative_revenue",
            ]
        )
        for m in metrics_log:
            writer.writerow(
                [
                    m.t,
                    m.phase,
                    f"{m.ctr_window:.6f}",
                    f"{m.revenue_window:.6f}",
                    f"{m.latency_ms:.2f}",
                    m.pending_conversions,
                    m.selected_ad_id,
                    m.n_arms,
                    m.cumulative_clicks,
                    f"{m.cumulative_revenue:.6f}",
                ]
            )

    print(f"   📄 Saved: {path}")


# ════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    metrics_log, all_ads, service = run_simulation(n_iterations=500, window_size=50)

    print("\n[Viz] Generating plots...")
    generate_plots(metrics_log)
    save_csv(metrics_log)

    # ── Summary statistics ──
    final = service.get_metrics()
    print("\n" + "=" * 65)
    print(" ✅ REAL-TIME SIMULATION COMPLETE")
    print("=" * 65)
    print(f"   Total requests:     {final['total_requests']}")
    print(f"   Total feedbacks:    {final['total_feedbacks']}")
    print(f"   Final CTR:          {final['avg_ctr']:.3f}")
    print(f"   Final Revenue:      {final['avg_revenue']:.4f}")
    print(f"   Avg Latency:        {final['avg_latency_ms']:.1f}ms")
    print(f"   Pending:            {final['pending_conversions']}")
    print(f"   Active arms:        {service.agent.n_arms}")
    print()
