#!/usr/bin/env python3
# plot_f1_distributions.py
# ------------------------------------------------------------
# Plot F1 score distributions across iterations for each model
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV = "iteration_summary.csv"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Load the summary file
df = pd.read_csv(CSV)

# Auto-detect iteration column
iteration_col = "run" if "run" in df.columns else "iteration"
if iteration_col not in df.columns:
    raise ValueError("❌ Could not find 'run' or 'iteration' column.")

# Clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Check for f1-score columns
f1_candidates = [c for c in df.columns if "f1" in c and "score" in c]
if not f1_candidates:
    raise ValueError("❌ No F1-score columns found in the summary file.")
f1_col = f1_candidates[0]

# Rename for consistency
df = df.rename(columns={iteration_col: "iteration", f1_col: "f1_score"})

# Plot: F1 boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="model", y="f1_score")
plt.title("F1 Score Distribution per Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/f1_distribution_boxplot.png")
print(f"📦 Saved: {OUT_DIR}/f1_distribution_boxplot.png")

# Plot: F1 score vs iteration per model
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="iteration", y="f1_score", hue="model", marker="o")
plt.title("F1 Score Across Iterations")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/f1_over_time_lineplot.png")
print(f"📦 Saved: {OUT_DIR}/f1_over_time_lineplot.png")
