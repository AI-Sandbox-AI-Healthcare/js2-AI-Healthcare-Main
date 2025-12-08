# best_stacking_meta_learner_across_iterations.py
"""
Finds the best stacking meta-learner across all iterations
"""

import pandas as pd
import glob
import os
import re
import shutil
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
BASE = "./"
output_model_card = os.path.join(BASE, "best-stacker-model-across-iterations.md")

# -----------------------------
# Step 1: Find all CSVs
# -----------------------------
fold_score_files = sorted(glob.glob(f"{BASE}/stacker_best_model_folds_*.csv"))

if not fold_score_files:
    raise FileNotFoundError("No fold-score CSV files found!")

# -----------------------------
# Step 2: Collect averages per iteration
# -----------------------------
iteration_scores = []

for file in fold_score_files:
    df = pd.read_csv(file)
    
    avg_f1 = df["Macro-F1"].mean()
    iteration_scores.append({
        "file": file,
        "avg_macro_f1": avg_f1
    })

# Convert to DataFrame
iteration_scores_df = pd.DataFrame(iteration_scores)

# -----------------------------
# Step 3: Find the best iteration
# -----------------------------
best_row = iteration_scores_df.loc[iteration_scores_df["avg_macro_f1"].idxmax()]
best_file = best_row["file"]
best_avg_f1 = best_row["avg_macro_f1"]

# Extract iteration string, e.g., iter1, iter2, etc.
match = re.search(r"(iter[0-9]+)", best_file)
if not match:
    raise ValueError(f"Cannot extract iteration from {best_file}")
best_iter = match.group(1)

# -----------------------------
# Step 4: Copy & rename artifacts
# -----------------------------
# Text file
txt_file = os.path.join(BASE, f"stacker_best_model_{best_iter}.txt")
if os.path.exists(txt_file):
    shutil.copy(txt_file, os.path.join(BASE, "stacker_best_model_across_iterations.txt"))

# Pickle file
pkl_file = os.path.join(BASE, f"stacker_best_model_{best_iter}.pkl")
if os.path.exists(pkl_file):
    shutil.copy(pkl_file, os.path.join(BASE, "stacker_best_model_across_iterations.pkl"))

# Binary metrics CSVs
metrics_files = glob.glob(os.path.join(BASE, f"stacker_binary_metrics_{best_iter}_*.csv"))
for f in metrics_files:
    fname = os.path.basename(f)
    new_fname = fname.replace(best_iter, "across_iterations")
    shutil.copy(f, os.path.join(BASE, new_fname))

# -----------------------------
# Step 5: Update model card
# -----------------------------
with open(output_model_card, "w") as f:
    f.write(f"# Model Card: Best Stacking Meta-Learner Across Iterations\n\n")
    f.write(f"**Best iteration:** `{best_iter}`\n")
    f.write(f"**Average Macro-F1:** {best_avg_f1:.4f}\n\n")
    f.write("## Saved Artifacts\n")
    f.write("- `stacker_best_model_across_iterations.txt`\n")
    f.write("- `stacker_best_model_across_iterations.pkl`\n")

    for csv_file in metrics_files:  
        new_fname = os.path.basename(csv_file).replace(best_iter, "across_iterations")
        f.write(f"- `{new_fname}`\n") 

print("Number of iterations found: ", len(iteration_scores_df))

print(f"Model card saved to {output_model_card}")
