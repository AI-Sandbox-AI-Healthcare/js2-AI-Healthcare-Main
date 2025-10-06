#!/usr/bin/env bash
# run_all_models.sh ---------------------------------------------------
# End-to-end benchmark + resource-logging pipeline for MIMIC-III
# With validation alignment and file sanity checks
# ---------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------
# 0. Setup logging
# ---------------------------------------------------------------------
BASE="./"
METRIC_PREFIX="${METRIC_PREFIX:-iter1}"
SEED_OFFSET="${SEED_OFFSET:-0}"
LOG_DIR="./logs"
LOGFILE="${LOG_DIR}/timing_${METRIC_PREFIX}.txt"

mkdir -p "$LOG_DIR"
echo "🧠 Benchmark script started at $(date)" > "$LOGFILE"

# helper to stamp durations
log_time() {
  local label="$1"
  local start_ts="$2"
  local end_ts
  end_ts=$(date +%s)
  local dur=$((end_ts - start_ts))
  echo "⏱️ $label took ${dur}s" | tee -a "$LOGFILE"
}

echo "💡 METRIC_PREFIX = $METRIC_PREFIX" | tee -a "$LOGFILE"
echo "🎯 SEED_OFFSET   = $SEED_OFFSET" | tee -a "$LOGFILE"
echo "🔁 Starting run at $(date)" | tee -a "$LOGFILE"

# ---------------------------------------------------------------------
# 1. Text Preprocessing (ClinicalBERT Tokenization)
# ---------------------------------------------------------------------
echo -e "\n=== [1] Text Preprocessing ===" | tee -a "$LOGFILE"

PROCESSED_NOTES_FILE="boosted_features_complete.txt"
TOKENIZED_MARKER="tokenization_complete_iter1.txt"

if [ ! -f "$PROCESSED_NOTES_FILE" ]; then
  echo "🔹 Running process_noteevents_text.py..." | tee -a "$LOGFILE"
  python process_noteevents_text.py 2>&1 | tee -a "$LOGFILE"
else
  echo "✅ process_noteevents_text.py already run. Skipping." | tee -a "$LOGFILE"
fi

# Tokenization only once (iter1)
if [ "$METRIC_PREFIX" = "iter1" ]; then
  if [ ! -f "$TOKENIZED_MARKER" ]; then
    echo "🔹 Running clinicalbert_tokenize_notes.py (iter1 only)..." | tee -a "$LOGFILE"
    python clinicalbert_tokenize_notes.py --metric_prefix iter1 2>&1 | tee -a "$LOGFILE"
    cp tokenization_complete.txt "$TOKENIZED_MARKER"
  else
    echo "✅ Tokenization already completed for iter1. Skipping." | tee -a "$LOGFILE"
  fi
else
  echo "🔄 [$METRIC_PREFIX] Reusing tokenized files from iter1..."
  for f in tokenized_input_ids_iter1.npy tokenized_attention_masks_iter1.npy tokenized_subject_ids_iter1.npy; do
    link_name="${f/iter1/$METRIC_PREFIX}"
    if [ ! -f "$link_name" ]; then
      ln -s "$f" "$link_name"
    fi
  done
fi


# ---------------------------------------------------------------------
# 2. Shared Validation IDs
# ---------------------------------------------------------------------
echo -e "\n=== [2] Shared Validation IDs ===" | tee -a "$LOGFILE"
python generate_shared_val_ids.py --seed_offset "$SEED_OFFSET" --metric_prefix "$METRIC_PREFIX" 2>&1 | tee -a "$LOGFILE"

# ---------------------------------------------------------------------
# 3. Visit Sequences
# ---------------------------------------------------------------------
echo -e "\n=== [3] Structured Visit Sequences ===" | tee -a "$LOGFILE"
start=$(date +%s)
SEED_OFFSET="$SEED_OFFSET" python lstm_sequences.py 2>&1 | tee -a "$LOGFILE"
SEED_OFFSET="$SEED_OFFSET" python transformer_sequences.py --metric_prefix "$METRIC_PREFIX" 2>&1 | tee -a "$LOGFILE"
log_time "Generating structured visit sequences" "$start"

# ---------------------------------------------------------------------
# 4. CPU Baselines
# ---------------------------------------------------------------------
echo -e "\n=== [4] CPU Baselines ===" | tee -a "$LOGFILE"
start=$(date +%s)
CUDA_VISIBLE_DEVICES="" SEED_OFFSET="$SEED_OFFSET" python tfidf_logreg_notes.py --metric_prefix "$METRIC_PREFIX" &
CUDA_VISIBLE_DEVICES="" SEED_OFFSET="$SEED_OFFSET" python mimic_classification.py --metric_prefix "$METRIC_PREFIX" &
wait
log_time "CPU model training (TF-IDF + RF/XGB)" "$start"

# ---------------------------------------------------------------------
# 5. GPU Models
# ---------------------------------------------------------------------
echo -e "\n=== [5] GPU Models ===" | tee -a "$LOGFILE"
start=$(date +%s)
SEED_OFFSET="$SEED_OFFSET" python train_lstm_mimic.py --metric_prefix "$METRIC_PREFIX" &
SEED_OFFSET="$SEED_OFFSET" python train_gru_mimic.py --metric_prefix "$METRIC_PREFIX" &
SEED_OFFSET="$SEED_OFFSET" python train_transformer_mimic.py --metric_prefix "$METRIC_PREFIX" &
wait

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CLS_FILE="precomputed_bert_cls_${METRIC_PREFIX}.npz"

if [ "$METRIC_PREFIX" = "iter1" ]; then
  if [ -f "$CLS_FILE" ]; then
    echo "📝 [$METRIC_PREFIX] Skipping BERT embedding precompute (found $CLS_FILE)"
  else
    echo "📝 [$METRIC_PREFIX] Precomputing BERT embeddings..."
    SEED_OFFSET="$SEED_OFFSET" METRIC_PREFIX="$METRIC_PREFIX" \
        python precompute_bert_embeddings.py 2>&1 | tee -a "$LOGFILE"
  fi
else
  echo "🔄 [$METRIC_PREFIX] Reusing precomputed BERT embeddings from iter1..."
  if [ ! -f "$CLS_FILE" ]; then
    ln -s "precomputed_bert_cls_iter1.npz" "$CLS_FILE"
  fi
fi

echo "🚀 [$METRIC_PREFIX] Training ClinicalBERT..."
SEED_OFFSET="$SEED_OFFSET" python clinicalbert_training.py --metric_prefix "$METRIC_PREFIX" 2>&1 | tee -a "$LOGFILE"

log_time "GPU-based model training" "$start"

# optional GPU memory snapshot
if command -v nvidia-smi &> /dev/null; then
  echo -e "\n🔍 GPU Memory Summary:" | tee -a "$LOGFILE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$LOGFILE"
fi


# ---------------------------------------------------------------------
# 6. Stacking Meta-Learner
# ---------------------------------------------------------------------
echo -e "\n=== [6] Stacking Meta-Learner ===" | tee -a "$LOGFILE"
start=$(date +%s)

# sanity-check that all .npz are present
missing=false
for m in lstm gru transformer clinicalbert_transformer rf xgb tfidf; do
  f="./${m}_probs_${METRIC_PREFIX}.npz"
  if [ ! -f "$f" ]; then
    echo "❌ Missing $f" | tee -a "$LOGFILE"
    missing=true
  fi
done
$missing && { echo "⚠️ Skipping stacking"; touch stacking_skipped_"$METRIC_PREFIX".txt; exit 0; }

# alignment check
python3 - <<PY
import numpy as np
mods = ['lstm','gru','transformer','clinicalbert_transformer','rf','xgb','tfidf']
sets = [set(np.load(f"{m}_probs_${METRIC_PREFIX}.npz")['subject_ids']) for m in mods]
shared = set.intersection(*sets)
print(f"✅ Aligned subjects: {len(shared)}")
with open("aligned_ids_count_${METRIC_PREFIX}.txt","w") as f: f.write(str(len(shared)))
PY
log_time "Validating alignment" "$start"

# run the meta-learner
SEED_OFFSET="$SEED_OFFSET" CUDA_VISIBLE_DEVICES="" python stacking_meta_learner.py --metric_prefix "$METRIC_PREFIX" 2>&1 | tee -a "$LOGFILE"
log_time "Stacking meta-learner" "$start"

if [ -f "stacker_best_model_${METRIC_PREFIX}.txt" ]; then
  echo "🏆 Best meta-learner: $(< stacker_best_model_${METRIC_PREFIX}.txt)" | tee -a "$LOGFILE"
fi

# ---------------------------------------------------------------------
# 7. Merge Metrics
# ---------------------------------------------------------------------
echo -e "\n=== [7] Merging Metrics ===" | tee -a "$LOGFILE"
start=$(date +%s)
python3 - <<PY
import pandas as pd, glob, os
pref = os.getenv("METRIC_PREFIX","iter1")
files = glob.glob(f"./**/*{pref}*.csv",recursive=True)
dfs=[]
for fp in files:
    tag = os.path.basename(fp).replace(f"_metrics_{pref}.csv","")
    try:
        df = pd.read_csv(fp)
        df.insert(0,"model",tag)
        df.insert(1,"iter",pref)
        dfs.append(df)
    except:
        pass
if dfs:
    pd.concat(dfs).to_csv(f"./results_summary_{pref}.csv",index=False)
    print("✅ Saved → results_summary_"+pref+".csv")
else:
    print("⚠️ No metrics found.")
PY
log_time "Metrics merging" "$start"

# append to global summary
summ="results_summary_${METRIC_PREFIX}.csv"
glob="iteration_summary.csv"
if [ -f "$summ" ]; then
  if [ ! -f "$glob" ]; then
    cp "$summ" "$glob"
  else
    tail -n +2 "$summ" >> "$glob"
  fi
fi

# ---------------------------------------------------------------------
# 8. Done
# ---------------------------------------------------------------------
echo -e "\n=== [8] Done ===" | tee -a "$LOGFILE"
echo "🎉 Finished at $(date)" | tee -a "$LOGFILE"
