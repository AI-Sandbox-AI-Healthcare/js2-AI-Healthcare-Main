#!/usr/bin/env bash
# generate_readme.sh -----------------------------------------------------
# Auto-generate a README.md file describing benchmark output structure
# -------------------------------------------------------------------------

set -euo pipefail

README_FILE="README.md"

echo "📝 Generating $README_FILE..."

cat > "$README_FILE" <<EOF
# AI Sandbox Benchmark Results

This directory contains the results from the **AI Sandbox Benchmark Pipeline**. Files are automatically organized by type for easier navigation and reproducibility.

## 📂 Directory Structure

- \`logs/\` — Raw logs from all iterations and pipeline steps.
- \`plots/\` — Visualizations such as:
  - F1 score distributions per model
  - F1 over iterations
  - Confusion matrices
  - SHAP force/waterfall plots
- \`shap_outputs/\` — SHAP feature explanations:
  - \`.html\`: Interactive force plots
  - \`.csv\`: Top feature importance scores
- \`model_cards/\` — Human-readable model documentation:
  - Final model card (\`model_card_iterX.md\`)
  - Logistic regression coefficient plots
- \`metrics/\` — Model metrics and predictions:
  - Per-iteration candidate scores
  - Final stacked model metrics
  - Raw prediction outputs

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
| \`iteration_summary.csv\` | Per-model metrics across iterations |
| \`benchmark_timing_summary.csv\` | Timing info and seed logs |
| \`results_summary.txt\` | High-level summary report |
| \`stacker_preds_iterX.csv\` | Final predictions from the meta-learner |
| \`stacker_multiclass_metrics_iterX_logisticregression.csv\` | Classification report for meta-learner |
| \`stacker_candidate_scores_iterX.csv\` | Comparison of meta-learner candidates |
| \`logreg_coef_iterX.png\` | Logistic regression coefficient plot |

## 🧠 About

This benchmark was generated using the **AI Sandbox Benchmark Pipeline** on MIMIC-III data and fine-tuned models such as ClinicalBERT, LSTM, and Transformer.

EOF

echo "✅ README.md created!"
