#!/usr/bin/env bash
# summarize_benchmark.sh ---------------------------------------------------
# Quickly summarize total benchmarking results across iterations
# Pulls timing, seed info, and calculates averages

set -euo pipefail

LOG_DIR="./logs"
SUMMARY_CSV="./benchmark_timing_summary.csv"

# Check if logs and summary exist
if [ ! -d "$LOG_DIR" ]; then
  echo "❌ Log directory $LOG_DIR not found!"
  exit 1
fi

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "❌ Benchmark timing summary $SUMMARY_CSV not found!"
  exit 1
fi

echo "📊 Benchmark Summary Report"
echo "--------------------------------------------------"

# Print total iterations
ITERATIONS=$(tail -n +2 "$SUMMARY_CSV" | wc -l)
echo "🔢 Total Iterations Completed: $ITERATIONS"

# Calculate total duration in seconds
TOTAL_SECONDS=$(awk -F',' 'NR>1 {s+=$4} END {print s}' "$SUMMARY_CSV")
TOTAL_MINUTES=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SECONDS/60}")
echo "⏱️  Total Wall Clock Time (Sum of All Iterations): $TOTAL_MINUTES minutes"

# Average duration
if [ "$ITERATIONS" -gt 0 ]; then
  AVERAGE_SECONDS=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SECONDS/$ITERATIONS}")
  AVERAGE_MINUTES=$(awk "BEGIN {printf \"%.2f\", $AVERAGE_SECONDS/60}")
  echo "🧮 Average Time per Iteration: $AVERAGE_MINUTES minutes"
else
  echo "⚠️  No iterations recorded. Cannot compute average."
fi

# Summarize individual seeds
echo ""
echo "🎯 Seeds Used:"
SEED_LOGS=$(grep -h "🎯 SEED_OFFSET" "$LOG_DIR"/iter*.out 2>/dev/null || true)
if [ -n "$SEED_LOGS" ]; then
  echo "$SEED_LOGS" | sed 's/.*iter\([0-9]\+\)\.out:.*SEED_OFFSET = \(.*\)/Iteration \1 → SEED_OFFSET \2/' | sort -n
else
  echo "⚠️  No seed logs found."
fi

# Optional: Show per-iteration durations
echo ""
echo "🕑 Iteration Durations:"
awk -F',' 'NR>1 {printf "Iteration %s took %d seconds\n", $1, $4}' "$SUMMARY_CSV"

echo ""
echo "✅ Summary Complete!"

# --------------------------------------------------
#  Log benchmark summary to MLflow
# --------------------------------------------------
echo ""
echo "📝 Logging summary to MLflow..."

python3 - <<PYTHON
import mlflow
import pandas as pd

summary_path = "$SUMMARY_CSV"
df = pd.read_csv(summary_path)

total_iterations = len(df)
total_duration = df["duration_sec"].sum()
average_duration = df["duration_sec"].mean()

with mlflow.start_run(run_name="benchmark_summary_${ITERATIONS}", nested=True):
    mlflow.set_tag("benchmark_type", "model_training")
    mlflow.log_param("total_iterations", total_iterations)
    mlflow.log_metric("total_duration_seconds", total_duration)
    mlflow.log_metric("average_duration_seconds", average_duration)

    # Optional: log individual iteration durations
    for _, row in df.iterrows():
        mlflow.log_metric(f"duration_iter_{row['iteration']}", row["duration_sec"])

print("✅ MLflow logging complete.")
PYTHON