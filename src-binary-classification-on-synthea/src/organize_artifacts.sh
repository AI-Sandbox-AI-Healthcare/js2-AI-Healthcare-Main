#!/usr/bin/env bash
# organize_artifacts.sh ---------------------------------------------------
# Organize all benchmark outputs into standard folders
# -------------------------------------------------------------------------

set -euo pipefail

echo "📁 Organizing benchmark artifacts..."

# -------------------------------------------------------------------------
# Create folders if they don't exist
# -------------------------------------------------------------------------
mkdir -p logs plots shap_outputs model_cards metrics

# -------------------------------------------------------------------------
# Move logs
# -------------------------------------------------------------------------
mv -v logs/*.out logs/*.txt logs/*.log logs/ 2>/dev/null || echo "⚠️  No new logs to move."

# -------------------------------------------------------------------------
# Move plots (F1, confusion, SHAP images)
# -------------------------------------------------------------------------
mv -v *_distribution*.png *_lineplot*.png plots/ 2>/dev/null || true
mv -v stacker_shap_*.png stacker_confusion*.png plots/ 2>/dev/null || true

# -------------------------------------------------------------------------
# Move SHAP HTML and feature CSV
# -------------------------------------------------------------------------
mv -v stacker_shap_force*.html shap_outputs/ 2>/dev/null || true
mv -v stacker_shap_top_features*.csv shap_outputs/ 2>/dev/null || true

# -------------------------------------------------------------------------
# Move model cards and coefficient plots
# -------------------------------------------------------------------------
mv -v model_card_iter*.md model_cards/ 2>/dev/null || true
mv -v logreg_coef_iter*.png model_cards/ 2>/dev/null || true

# -------------------------------------------------------------------------
# Move metrics and prediction outputs
# -------------------------------------------------------------------------
mv -v stacker_preds_iter*.csv stacker_multiclass_metrics_iter*.csv \
      stacker_candidate_scores_iter*.csv metrics/ 2>/dev/null || true

echo "✅ Artifact organization complete!"
