import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("metrics", exist_ok=True)

eng_df = pd.read_csv("metrics/eng_matrix.csv", index_col=0)
rev_df = pd.read_csv("metrics/rev_matrix.csv", index_col=0)

model_pairs = {
    "LinUCB": "H-LinUCB",
    "Thompson": "H-Thompson",
    "NeuralUCB": "H-NeuralUCB",
    "NeuralTS": "H-NeuralTS",
    "DeepBandit": "H-DeepBandit",
    "Offline2On": "H-Offline2On",
    "DelayedFB": "H-DelayedFB",
}

max_eng = eng_df.max(axis=1)
max_rev = rev_df.max(axis=1)

delta_eng = []
delta_rev = []
labels = []
colors = []

# Using Devoteam/EMI colors from report
win_color = "#2E7D32"  # Green
lose_color = "#C62828"  # Red
mixed_color = "#F9A825"  # Yellow

for classic, hybrid in model_pairs.items():
    if classic in max_eng and hybrid in max_eng:
        d_eng = max_eng[hybrid] - max_eng[classic]
        d_rev = max_rev[hybrid] - max_rev[classic]

        delta_eng.append(d_eng)
        delta_rev.append(d_rev)
        labels.append(f"{classic} → {hybrid}")

        if d_eng > 0 and d_rev > 0:
            colors.append(win_color)
        elif d_eng < 0 and d_rev < 0:
            colors.append(lose_color)
        else:
            colors.append(mixed_color)

plt.figure(figsize=(10, 8))

# Draw quadrants
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)

# Fill quadrants for visual clarity
plt.axhspan(
    0,
    max(max(delta_rev) * 1.1, 0.05),
    xmin=0.5,
    xmax=1,
    facecolor="#E8F5E9",
    alpha=0.5,
    zorder=0,
)  # Top Right (Win-Win)
plt.axhspan(
    min(min(delta_rev) * 1.1, -0.05),
    0,
    xmin=0,
    xmax=0.5,
    facecolor="#FFEBEE",
    alpha=0.5,
    zorder=0,
)  # Bottom Left (Lose-Lose)

plt.scatter(
    delta_eng, delta_rev, c=colors, s=150, zorder=3, edgecolors="black", linewidths=1
)

for i, label in enumerate(labels):
    # Adjust position to avoid overlap
    y_offset = 0.002
    x_offset = 0.002
    if "Thompson" in label:
        x_offset = -0.02
    elif "NeuralTS" in label:
        x_offset = -0.015
    plt.annotate(
        label,
        (delta_eng[i] + x_offset, delta_rev[i] + y_offset),
        fontsize=11,
        fontweight="bold",
        zorder=4,
    )

plt.title(
    "Impact de l’Hybridation (Zero-Shot) : Δ Engagement vs Δ Revenue",
    fontsize=14,
    pad=20,
)
plt.xlabel("Gain d’Engagement (Δ CTR)", fontsize=12)
plt.ylabel("Gain de Revenue (Δ eCPM)", fontsize=12)

# Quadrant labels
plt.text(
    max(delta_eng) * 0.8,
    max(delta_rev) * 0.9,
    "Hybride Gagnant\n(Win-Win)",
    fontsize=12,
    color=win_color,
    fontweight="bold",
    ha="center",
    va="center",
)
plt.text(
    min(delta_eng) * 0.8,
    min(delta_rev) * 0.9,
    "Classique Gagnant\n(Lose-Lose)",
    fontsize=12,
    color=lose_color,
    fontweight="bold",
    ha="center",
    va="center",
)

plt.grid(True, linestyle=":", alpha=0.7, zorder=1)

# Set limits symmetrically if possible
max_abs_eng = max(abs(min(delta_eng)), abs(max(delta_eng))) * 1.2
max_abs_rev = max(abs(min(delta_rev)), max(delta_rev)) * 1.2
plt.xlim(-max_abs_eng, max_abs_eng)
plt.ylim(-max_abs_rev, max_abs_rev)

plt.tight_layout()
output_path = "metrics/hybridization_delta_joint.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to {output_path}")
