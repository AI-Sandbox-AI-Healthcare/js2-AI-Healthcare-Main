#!/usr/bin/env bash
# run_benchmark_iterations.sh
# ---------------------------------------------------------------------
# Run N iterations of full model training (or meta-only) and save benchmark results
# Robust to missing log files.
# ---------------------------------------------------------------------

set -euo pipefail

TOTAL_ITERATIONS=4
RUN_SCRIPT="./run_all_models.sh"
LOG_DIR="../analysis/logs"
SUMMARY_CSV="../analysis/results/metrics/benchmark_timing_summary.csv"
GLOBAL_LOG="$LOG_DIR/full_benchmark.log"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------
# Validate required script
# ---------------------------------------------------------------------
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "❌ Error: $RUN_SCRIPT not found!"
  exit 1
fi

# ---------------------------------------------------------------------
# Detect where to resume
# ---------------------------------------------------------------------
existing_iters=()
shopt -s nullglob
for f in "$LOG_DIR"/iter*.out; do
  [[ $f =~ iter([0-9]+)\.out$ ]] && existing_iters+=("${BASH_REMATCH[1]}")
done
shopt -u nullglob

if [ ${#existing_iters[@]} -gt 0 ]; then
  IFS=$'\n' sorted=($(sort -n <<<"${existing_iters[*]}")); unset IFS
  last_completed=${sorted[-1]}
  START_ITER=$((last_completed + 1))
  echo "🔄 Resuming from iteration $START_ITER (last completed: $last_completed)" | tee -a "$GLOBAL_LOG"
else
  echo "⚠️  No existing iteration logs found. Starting from iteration 1." | tee -a "$GLOBAL_LOG"
  START_ITER=1
fi

# ---------------------------------------------------------------------
# Initialize timing summary
# ---------------------------------------------------------------------
if [ ! -f "$SUMMARY_CSV" ]; then
  echo "iteration,start_time,end_time,duration_sec,gpu_id" > "$SUMMARY_CSV"
fi

# ---------------------------------------------------------------------
# GPU detection (defaults to 0 if not available)
# ---------------------------------------------------------------------
# GPU ID detection (safe for CPU-only instances)
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n1)
  echo "🧠 Detected GPU ID: $GPU_ID"
else
  GPU_ID=""
  echo "🧩 No GPU detected — running in CPU-only mode."
fi

# ---------------------------------------------------------------------
# Function: run one iteration
# ---------------------------------------------------------------------
run_iteration() {
  local i=$1
  local tag="iter${i}"
  local log_file="${LOG_DIR}/${tag}.out"
  local start_time=$(date +%s)

  echo "🚀 [$tag] Starting on GPU $GPU_ID at $(date)" | tee -a "$log_file"

  chmod +x "$RUN_SCRIPT"

  CUDA_VISIBLE_DEVICES="$GPU_ID" METRIC_PREFIX="$tag" SEED_OFFSET="$i" \
    bash "$RUN_SCRIPT" 2>&1 | tee -a "$log_file"


  local end_time=$(date +%s)
  local duration=$((end_time - start_time))

  echo "$tag,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration,$GPU_ID" \
    >> "$SUMMARY_CSV"
  echo "✅ [$tag] Done in ${duration}s" | tee -a "$log_file"
  return 0
}

# ---------------------------------------------------------------------
# Skip if already complete
# ---------------------------------------------------------------------
if (( START_ITER > TOTAL_ITERATIONS )); then
  echo "✅ All $TOTAL_ITERATIONS iterations already completed." | tee -a "$GLOBAL_LOG"
  exit 0
fi

# ---------------------------------------------------------------------
# STEP 1: Iterations (bulletproof version)
# ---------------------------------------------------------------------
echo "=== STEP 1: Benchmark Iterations ===" | tee -a "$GLOBAL_LOG"

for ((i = START_ITER; i <= TOTAL_ITERATIONS; i++)); do
  echo "⚙️  Running iteration $i..." | tee -a "$GLOBAL_LOG"

  (
    # Run each iteration in a subshell so failure doesn’t stop the loop
    set +e
    run_iteration "$i"
    status=$?
    echo "Iteration $i exited with status $status" >> "$GLOBAL_LOG"
    exit $status
  )

  status=$?

  if [[ $status -ne 0 ]]; then
    echo "⚠️ Iteration $i failed (exit $status) — continuing to next." | tee -a "$GLOBAL_LOG"
  else
    echo "✅ Iteration $i completed successfully." | tee -a "$GLOBAL_LOG"
  fi

  sleep 2
  sync
  echo "🚀 Finished iteration $i." | tee -a "$GLOBAL_LOG"
done

# ---------------------------------------------------------------------
# STEP 2: Find Best Meta-Learner Across Iterations
# ---------------------------------------------------------------------
echo "=== STEP 2: Best Stacking Meta-Learner Across Iterations ===" | tee -a "$GLOBAL_LOG"
python3 best_stacking_meta_learner_across_iterations.py | tee -a "$GLOBAL_LOG"

# ---------------------------------------------------------------------
# STEP 3–7: Postprocessing
# ---------------------------------------------------------------------
echo "=== STEP 3: Merge Results ===" | tee -a "$GLOBAL_LOG"

if compgen -G "$LOG_DIR/iter*.out" > /dev/null; then
  python3 merge_benchmark_results.py | tee -a "$GLOBAL_LOG"
  python3 wilcoxon_test.py | tee -a "$GLOBAL_LOG"
  #python3 plot_f1_distributions.py | tee -a "$GLOBAL_LOG"
  python3 summarize_benchmark.py | tee -a "$GLOBAL_LOG"
  echo "🎉 Benchmarking complete! All outputs updated." | tee -a "$GLOBAL_LOG"
else
  echo "⚠️  No iteration outputs found to merge. Skipping post-processing." | tee -a "$GLOBAL_LOG"
fi

echo "=== STEP 4: Summarize Benchmark to MLflow ===" | tee -a "$GLOBAL_LOG"
bash run_summarize_benchmarks.sh | tee -a "$GLOBAL_LOG"

echo "=== STEP 5: Artifact Summary ===" | tee -a "$GLOBAL_LOG"
ls -lh ../analysis/results/metrics/results_summary*.csv \
       ../analysis/results/metrics/iteration_summary.csv \
       ../analysis/logs/*.out 2>/dev/null \
  | grep -v '.err' | tee -a "$GLOBAL_LOG" || echo "⚠️  No artifacts found." | tee -a "$GLOBAL_LOG"

echo "=== STEP 6: Organize Outputs ===" | tee -a "$GLOBAL_LOG"
bash organize_artifacts.sh | tee -a "$GLOBAL_LOG"

echo "=== STEP 7: Generate README.md ===" | tee -a "$GLOBAL_LOG"
bash generate_readme.sh | tee -a "$GLOBAL_LOG"

echo "✅ Full Benchmark Complete!" | tee -a "$GLOBAL_LOG"
echo "🕔 Finished at: $(date)" | tee -a "$GLOBAL_LOG"
