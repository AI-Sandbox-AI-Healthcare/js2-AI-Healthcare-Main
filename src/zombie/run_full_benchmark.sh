#!/usr/bin/env bash
# run_full_benchmark.sh ---------------------------------------------------
# Master script: Run full AI Sandbox benchmark with iterations, summary, and plots
# -------------------------------------------------------------------------

set -euo pipefail

# -------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------
LOGFILE="full_benchmark.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "🧠 Starting Full AI Sandbox Benchmark Pipeline..."
echo "🕒 Started at: $(date)"
echo "--------------------------------------------------"

# -------------------------------------------------------------------------
# STEP 1: Run iterations with benchmarking and logging
# -------------------------------------------------------------------------
echo "🚀 Launching Benchmark Iterations (total)..."
bash run_benchmark_iterations.sh

# -------------------------------------------------------------------------
# STEP 2: Summarize results + log to MLflow
# -------------------------------------------------------------------------
echo ""
echo "📊 Summarizing Benchmark..."
bash run_summarize_benchmarks.sh

# -------------------------------------------------------------------------
# STEP 3: Artifact Summary
# -------------------------------------------------------------------------
echo ""
echo "📂 Benchmark Artifacts:"
ls -lh results_summary*.csv iteration_summary.csv logs/*.out 2>/dev/null | grep -v '.err' || echo "⚠️  No artifacts found."

# -------------------------------------------------------------------------
# STEP 4: Organize Outputs
# -------------------------------------------------------------------------
echo ""
echo "🧹 Organizing output files..."
bash organize_artifacts.sh

# -------------------------------------------------------------------------
# STEP 5: Generate README
# -------------------------------------------------------------------------
echo ""
echo "📝 Creating README.md..."
bash generate_readme.sh

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo ""
echo "✅ Full Benchmark Complete!"
echo "🕔 Finished at: $(date)"
