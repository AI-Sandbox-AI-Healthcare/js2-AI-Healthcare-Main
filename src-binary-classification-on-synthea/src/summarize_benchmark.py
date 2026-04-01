#!/usr/bin/env python3
"""
Summarize Benchmark Results and Resource Usage
"""

import pandas as pd

# === 1. Load Files ===
benchmark_file = "results_summary_all_iterations.csv"
resource_file = "resource_usage.csv"

benchmark_df = pd.read_csv(benchmark_file)
resource_df = pd.read_csv(resource_file)

# === 2. Clean Model Names ===
model_name_map = {
    "transformer_multiclass_metrics": "Transformer",
    "lstm_multiclass_metrics": "LSTM",
    "gru_multiclass_metrics": "GRU",
    "clinicalbert_multiclass_metrics": "ClinicalBERT-LSTM",
    "tfidf_logreg_notes_metrics": "TFIDF Logistic Regression",
    "tabular_logreg_multiclass_metrics": "Tabular Logistic Regression",
    "stacker_multiclass_metrics": "Stacking Meta-Learner",
}

benchmark_df["model_clean"] = benchmark_df["model"].map(model_name_map)

resource_name_map = {
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "clinicalbert_lstm": "ClinicalBERT-LSTM",
    "lstm_multiclass": "LSTM",
    "gru_multiclass": "GRU",
    "stacker": "Stacking Meta-Learner",
    "tfidf_logreg_notes": "TFIDF Logistic Regression",
    "tabular_logreg_multiclass": "Tabular Logistic Regression",
}

resource_df["Model"] = resource_df["tag"].map(resource_name_map)

# === 3. Summarize Performance ===
performance_summary = benchmark_df.groupby("model_clean").agg({
    "Precision": ["mean", "std"],
    "Recall": ["mean", "std"],
    "F1-score": ["mean", "std"]
}).reset_index()

performance_summary.columns = [
    "Model",
    "Precision (avg)", "Precision (std)",
    "Recall (avg)", "Recall (std)",
    "F1-score (avg)", "F1-score (std)"
]

# === 4. Summarize Resources ===
resource_summary = resource_df.groupby("Model").agg({
    "elapsed_hr": ["mean", "std"],
    "gpu_hrs": ["mean", "std"],
    "cpu_pct": ["mean", "std"],
    "disk_used_gb": ["mean", "std"]
}).reset_index()

resource_summary.columns = [
    "Model",
    "Elapsed Hours (avg)", "Elapsed Hours (std)",
    "GPU Hours (avg)", "GPU Hours (std)",
    "CPU Utilization (avg)", "CPU Utilization (std)",
    "Disk Used (avg GB)", "Disk Used (std GB)"
]

# === 5. Merge Everything Together ===
# Training Time Estimates
training_time_estimates = {
    "Transformer": 55,
    "LSTM": 45,
    "GRU": 45,
    "ClinicalBERT-LSTM": 150,
    "TFIDF Logistic Regression": 5,
    "Tabular Logistic Regression": 5,
    "Stacking Meta-Learner": 15,
}

hardware_notes = {
    "Transformer": "Jetstream2 GPU.xlarge",
    "LSTM": "Jetstream2 GPU.xlarge",
    "GRU": "Jetstream2 GPU.xlarge",
    "ClinicalBERT-Transformer": "Jetstream2 GPU.xlarge",
    "TFIDF Logistic Regression": "CPU only",
    "Tabular Logistic Regression": "CPU only",
    "Stacking Meta-Learner": "CPU only",
}

# Merge resource and performance
merged = pd.merge(resource_summary, performance_summary, on="Model", how="outer")

# Add Training Time and GPU Instance
merged["Training Time (min)"] = merged["Model"].map(training_time_estimates)
merged["GPU Instance"] = merged["Model"].map(hardware_notes)

# Rearrange columns
merged = merged[[
    "Model", "Training Time (min)", "GPU Instance",
    "GPU Hours (avg)", "GPU Hours (std)",
    "CPU Utilization (avg)", "CPU Utilization (std)",
    "Disk Used (avg GB)", "Disk Used (std GB)",
    "Precision (avg)", "Precision (std)",
    "Recall (avg)", "Recall (std)",
    "F1-score (avg)", "F1-score (std)"
]]

# === 6. Save Final Summary ===
output_file = "final_benchmark_summary.csv"
merged.to_csv(output_file, index=False)

print(f"✅ Final Benchmark Summary saved to {output_file}")
