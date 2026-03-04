"""
Best-Across-Policies Comparison — Fair Agent Evaluation.

For each agent, selects the BEST result across all 10 MOO policies,
ensuring we compare true maxima rather than just the Scalar policy.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load saved data ──
eng = pd.read_csv("metrics/eng_matrix.csv", index_col=0)
rev = pd.read_csv("metrics/rev_matrix.csv", index_col=0)
time_ = pd.read_csv("metrics/time_matrix.csv", index_col=0)

agents = eng.index.tolist()
policies = eng.columns.tolist()

# ── Classify agents ──
classical = agents[:7]
hybrid = agents[7:14]
llm = agents[14:]

# ═══════════════════════════════════════════════════════════════
# TABLE 1: Best Engagement per Agent (across all policies)
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print("BEST ENGAGEMENT PER AGENT (across all 10 policies)")
print("=" * 90)
print(f"{'Agent':<16} {'Best Eng':>10} {'Policy':>12}  {'Avg Time':>10}")
print("-" * 55)
for ag in agents:
    best_pol = eng.loc[ag].idxmax()
    best_eng = eng.loc[ag].max()
    avg_t = time_.loc[ag].mean()
    print(f"{ag:<16} {best_eng:>10.4f} {best_pol:>12}  {avg_t:>10.1f}ms")

# ═══════════════════════════════════════════════════════════════
# TABLE 2: Best Revenue per Agent (across all policies)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print("BEST REVENUE PER AGENT (across all 10 policies)")
print("=" * 90)
print(f"{'Agent':<16} {'Best Rev':>10} {'Policy':>12}  {'Avg Time':>10}")
print("-" * 55)
for ag in agents:
    best_pol = rev.loc[ag].idxmax()
    best_rev = rev.loc[ag].max()
    avg_t = time_.loc[ag].mean()
    print(f"{ag:<16} {best_rev:>10.4f} {best_pol:>12}  {avg_t:>10.1f}ms")

# ═══════════════════════════════════════════════════════════════
# TABLE 3: HYBRIDIZATION DELTA (Best vs Best)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print("HYBRIDIZATION DELTA — BEST vs BEST (across all policies)")
print("=" * 90)
print(
    f"{'Classical':<14} {'C.BestEng':>10} {'Policy':>12} | {'Hybrid':<14} {'H.BestEng':>10} {'Policy':>12} | {'Δ':>8}"
)
print("-" * 100)
for c, h in zip(classical, hybrid):
    c_best = eng.loc[c].max()
    c_pol = eng.loc[c].idxmax()
    h_best = eng.loc[h].max()
    h_pol = eng.loc[h].idxmax()
    delta = h_best - c_best
    mark = "✅" if delta > 0 else "❌"
    print(
        f"{c:<14} {c_best:>10.4f} {c_pol:>12} | {h:<14} {h_best:>10.4f} {h_pol:>12} | {delta:>+8.4f} {mark}"
    )

print(
    f"\n{'Classical':<14} {'C.BestRev':>10} {'Policy':>12} | {'Hybrid':<14} {'H.BestRev':>10} {'Policy':>12} | {'Δ':>8}"
)
print("-" * 100)
for c, h in zip(classical, hybrid):
    c_best = rev.loc[c].max()
    c_pol = rev.loc[c].idxmax()
    h_best = rev.loc[h].max()
    h_pol = rev.loc[h].idxmax()
    delta = h_best - c_best
    mark = "✅" if delta > 0 else "❌"
    print(
        f"{c:<14} {c_best:>10.4f} {c_pol:>12} | {h:<14} {h_best:>10.4f} {h_pol:>12} | {delta:>+8.4f} {mark}"
    )

# ═══════════════════════════════════════════════════════════════
# PLOT 1: Grouped Bar — Best Engagement (Classical vs Hybrid)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Engagement bars
base_names = [
    "LinUCB",
    "Thompson",
    "NeuralUCB",
    "NeuralTS",
    "DeepBandit",
    "Offline2On",
    "DelayedFB",
]
c_eng = [eng.loc[c].max() for c in classical]
h_eng = [eng.loc[h].max() for h in hybrid]
c_eng_pol = [eng.loc[c].idxmax() for c in classical]
h_eng_pol = [eng.loc[h].idxmax() for h in hybrid]

x = np.arange(len(base_names))
width = 0.35
bars1 = axes[0].bar(
    x - width / 2,
    c_eng,
    width,
    label="Classical (Disjoint)",
    color="#2196F3",
    edgecolor="black",
    linewidth=0.5,
)
bars2 = axes[0].bar(
    x + width / 2,
    h_eng,
    width,
    label="Hybrid (Global Semantic)",
    color="#4CAF50",
    edgecolor="black",
    linewidth=0.5,
)

# Add policy labels on top of bars
for bar, pol in zip(bars1, c_eng_pol):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        pol,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=45,
        color="#1565C0",
    )
for bar, pol in zip(bars2, h_eng_pol):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        pol,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=45,
        color="#2E7D32",
    )

axes[0].set_ylabel("Best Engagement", fontsize=12)
axes[0].set_title(
    "Best Engagement per Agent\n(across all 10 MOO policies)", fontsize=13
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(base_names, fontsize=10)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0.55, max(max(c_eng), max(h_eng)) + 0.06)
axes[0].grid(True, alpha=0.3, axis="y")

# Revenue bars
c_rev = [rev.loc[c].max() for c in classical]
h_rev = [rev.loc[h].max() for h in hybrid]
c_rev_pol = [rev.loc[c].idxmax() for c in classical]
h_rev_pol = [rev.loc[h].idxmax() for h in hybrid]

bars3 = axes[1].bar(
    x - width / 2,
    c_rev,
    width,
    label="Classical (Disjoint)",
    color="#2196F3",
    edgecolor="black",
    linewidth=0.5,
)
bars4 = axes[1].bar(
    x + width / 2,
    h_rev,
    width,
    label="Hybrid (Global Semantic)",
    color="#4CAF50",
    edgecolor="black",
    linewidth=0.5,
)

for bar, pol in zip(bars3, c_rev_pol):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        pol,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=45,
        color="#1565C0",
    )
for bar, pol in zip(bars4, h_rev_pol):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        pol,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=45,
        color="#2E7D32",
    )

axes[1].set_ylabel("Best Revenue", fontsize=12)
axes[1].set_title("Best Revenue per Agent\n(across all 10 MOO policies)", fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(base_names, fontsize=10)
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, max(max(c_rev), max(h_rev)) + 0.015)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("metrics/best_across_policies_bars.png", dpi=150)
plt.close()
print("\n✅ metrics/best_across_policies_bars.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 2: Delta chart — Improvement of Hybrid over Classical
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

delta_eng = [h - c for h, c in zip(h_eng, c_eng)]
delta_rev = [h - c for h, c in zip(h_rev, c_rev)]
colors_eng = ["#4CAF50" if d > 0 else "#F44336" for d in delta_eng]
colors_rev = ["#4CAF50" if d > 0 else "#F44336" for d in delta_rev]

axes[0].barh(base_names, delta_eng, color=colors_eng, edgecolor="black", linewidth=0.5)
axes[0].axvline(x=0, color="black", linewidth=1)
axes[0].set_xlabel("Δ Engagement (Hybrid - Classical)", fontsize=11)
axes[0].set_title("Engagement Gain from Hybridization\n(Best vs Best)", fontsize=13)
for i, (d, cn, hn) in enumerate(zip(delta_eng, c_eng_pol, h_eng_pol)):
    label = f"{d:+.4f}"
    axes[0].text(
        d + (0.002 if d > 0 else -0.002),
        i,
        label,
        va="center",
        ha="left" if d > 0 else "right",
        fontsize=9,
        fontweight="bold",
    )
axes[0].grid(True, alpha=0.3, axis="x")

axes[1].barh(base_names, delta_rev, color=colors_rev, edgecolor="black", linewidth=0.5)
axes[1].axvline(x=0, color="black", linewidth=1)
axes[1].set_xlabel("Δ Revenue (Hybrid - Classical)", fontsize=11)
axes[1].set_title("Revenue Gain from Hybridization\n(Best vs Best)", fontsize=13)
for i, (d, cn, hn) in enumerate(zip(delta_rev, c_rev_pol, h_rev_pol)):
    label = f"{d:+.4f}"
    axes[1].text(
        d + (0.001 if d > 0 else -0.001),
        i,
        label,
        va="center",
        ha="left" if d > 0 else "right",
        fontsize=9,
        fontweight="bold",
    )
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("metrics/hybridization_delta_best_vs_best.png", dpi=150)
plt.close()
print("✅ metrics/hybridization_delta_best_vs_best.png")

print("\n✅ Done!")
