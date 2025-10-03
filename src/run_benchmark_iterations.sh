#!/usr/bin/env bash
# run_benchmark_iterations.sh
# ---------------------------------------------------------------------
# Run N iterations of full model training and save benchmark results
# ---------------------------------------------------------------------

set -euo pipefail

TOTAL_ITERATIONS=30        # adjust as needed
RUN_SCRIPT="./run_all_models.sh"
LOG_DIR="./logs"
SUMMARY_CSV="./benchmark_timing_summary.csv"
GLOBAL_LOG="full_benchmark.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "❌ Error: $RUN_SCRIPT not found!"
  exit 1
fi

# Detect where to resume
existing_iters=($(ls "$LOG_DIR"/iter*.out 2>/dev/null | sed -E 's/.*iter([0-9]+)\.out/\1/' | sort -n))
if [ ${#existing_iters[@]} -gt 0 ]; then
  last_completed=${existing_iters[-1]}
  START_ITER=$((last_completed + 1))
else
  echo "⚠️ No existing iteration logs found. Starting from iteration 1."
  START_ITER=1   # ✅ force start at 1, not 0
fi

# Initialize timing summary if missing
if [ ! -f "$SUMMARY_CSV" ]; then
  echo "iteration,start_time,end_time,duration_sec,gpu_id" > "$SUMMARY_CSV"
fi

echo "🔎 Resuming from iteration $START_ITER" | tee -a "$GLOBAL_LOG"

# GPU ID detection
if command -v nvidia-smi &> /dev/null; then
  GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n1)
else
  GPU_ID=0
fi

run_iteration() {
  local i=$1
  local tag="iter${i}"
  local log_file="${LOG_DIR}/${tag}.out"
  local start_time=$(date +%s)

  echo "🚀 [$tag] Starting on GPU $GPU_ID at $(date)" | tee -a "$log_file"
  
  # ensure run_all_models.sh is executable
  chmod +x "$RUN_SCRIPT"

  CUDA_VISIBLE_DEVICES="$GPU_ID" METRIC_PREFIX="$tag" SEED_OFFSET="$i" \
    bash "$RUN_SCRIPT" 2>&1 | tee -a "$log_file"

  local end_time=$(date +%s)
  local duration=$((end_time - start_time))

  echo "$tag,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration,$GPU_ID" >> "$SUMMARY_CSV"
  echo "✅ [$tag] Done in ${duration}s" | tee -a "$log_file"
}

if (( START_ITER > TOTAL_ITERATIONS )); then
  echo "✅ All $TOTAL_ITERATIONS iterations already completed." | tee -a "$GLOBAL_LOG"
  exit 0
fi

# -------------------------------------------------------------------------
# STEP 1: Iterations
# -------------------------------------------------------------------------
echo "=== STEP 1: Benchmark Iterations ===" | tee -a "$GLOBAL_LOG"

for ((i = START_ITER; i <= TOTAL_ITERATIONS; i++)); do
  run_iteration "$i"
  sleep 2  # Small pause to reduce load on filesystem
  sync     # Ensure filesystem buffers are flushed
  echo "🚀 Finished iteration $i." | tee -a "$GLOBAL_LOG"
done

# -------------------------------------------------------------------------
# STEP 2: Merge results
# -------------------------------------------------------------------------
echo "=== STEP 2: Merge Results ===" | tee -a "$GLOBAL_LOG"

if ls "$LOG_DIR"/iter*.out 1> /dev/null 2>&1; then
  python3 merge_benchmark_results.py | tee -a "$GLOBAL_LOG"
  python3 wilcoxon_test.py | tee -a "$GLOBAL_LOG"
  python3 plot_f1_distributions.py | tee -a "$GLOBAL_LOG"
  python3 summarize_benchmark.py | tee -a "$GLOBAL_LOG"
  echo "🎉 Benchmarking complete! All outputs updated." | tee -a "$GLOBAL_LOG"
else
  echo "⚠️ No iteration outputs found to merge. Skipping post-processing." | tee -a "$GLOBAL_LOG"
fi

# -------------------------------------------------------------------------
# STEP 3: Summarize results + log to MLflow
# -------------------------------------------------------------------------
echo "=== STEP 3: Summarize Benchmark to MLflow ===" | tee -a "$GLOBAL_LOG"
bash run_summarize_benchmarks.sh | tee -a "$GLOBAL_LOG"

# -------------------------------------------------------------------------
# STEP 4: Artifact Summary
# -------------------------------------------------------------------------
echo "=== STEP 4: Artifact Summary ===" | tee -a "$GLOBAL_LOG"
ls -lh results_summary*.csv iteration_summary.csv logs/*.out 2>/dev/null | grep -v '.err' | tee -a "$GLOBAL_LOG" || echo "⚠️  No artifacts found." | tee -a "$GLOBAL_LOG"

# -------------------------------------------------------------------------
# STEP 5: Organize Outputs
# -------------------------------------------------------------------------
echo "=== STEP 5: Organize Outputs ===" | tee -a "$GLOBAL_LOG"
bash organize_artifacts.sh | tee -a "$GLOBAL_LOG"

# -------------------------------------------------------------------------
# STEP 6: Generate README
# -------------------------------------------------------------------------
echo "=== STEP 6: Generate README.md ===" | tee -a "$GLOBAL_LOG"
bash generate_readme.sh | tee -a "$GLOBAL_LOG"

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo "✅ Full Benchmark Complete!" | tee -a "$GLOBAL_LOG"
echo "🕔 Finished at: $(date)" | tee -a "$GLOBAL_LOG"
