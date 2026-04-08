#!/usr/bin/env bash
# generate_readme.sh -----------------------------------------------------
# Auto-generate a README.md file describing benchmark output structure
# -------------------------------------------------------------------------

set -euo pipefail

README_FILE="../README.md"

echo "📝 Generating $README_FILE..."

cat > "$README_FILE" <<EOF
# AI Sandbox Benchmark Results

This directory contains the results from the **AI Sandbox Benchmark Pipeline**. Files are automatically organized by type for easier navigation and reproducibility.

## 📂 Directory Structure

- \`analysis/logs/\` — Raw logs from all iterations and pipeline steps.
- \`analysis/results/figures/\` — Visualizations such as:
  - F1 score distributions per model
  - F1 over iterations
  - Confusion matrices
  - SHAP force/waterfall plots
- \`analysis/results/figures/stacking_meta_learner/\` — Confusion matrices, SHAP plots, and coefficient plots for the meta-learner.
- \`analysis/results/model_cards/\` — Human-readable model documentation:
  - Final model card (\`model_card_iterX.md\`)
  - Logistic regression coefficient plots
- \`analysis/results/metrics/\` — Model metrics and predictions:
  - Per-iteration candidate scores
  - Final stacked model metrics
  - Raw prediction outputs
- \`analysis/models/\` — Saved trained model artifacts.
- \`analysis/experiments/mlruns/\` — MLflow experiment tracking outputs.
- \`analysis/experiments/catboost_info/\` — CatBoost training logs and metadata.
- \`analysis/data/derivedData/\` — Reusable intermediate outputs such as probability \`.npz\` files.

## 🧪 Pipeline Steps

This project follows a standardized benchmark workflow:

1. Run training iterations with logging
2. Summarize benchmark statistics
3. Generate visualizations
4. Log to MLflow
5. Organize artifacts
6. (This file) Document results

## 📦 Output Files

| File | Description |
|------|-------------|
| \`analysis/results/metrics/iteration_summary.csv\` | Per-model metrics across iterations |
| \`analysis/results/metrics/benchmark_timing_summary.csv\` | Timing info and seed logs |
| \`analysis/results/metrics/results_summary.txt\` | High-level summary report |
| \`analysis/results/metrics/stacker_preds_iterX.csv\` | Final predictions from the meta-learner |
| \`analysis/results/metrics/stacker_candidate_scores_iterX.csv\` | Comparison of meta-learner candidates |
| \`analysis/results/figures/stacking_meta_learner/logreg_coef_iterX.png\` | Logistic regression coefficient plot |
| \`analysis/results/model_cards/model_card_iterX.md\` | Model card for the selected meta-learner |

## 🧠 About

This benchmark was generated using the **AI Sandbox Benchmark Pipeline** on MIMIC-III data and fine-tuned models such as ClinicalBERT, LSTM, and Transformer.

EOF

echo "✅ ../README.md created!"
