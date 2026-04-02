# clinicalbert_tokenize_notes.py
# ---------------------------------------------------------------------
# Efficient Parallel Tokenization for ClinicalBERT
# ---------------------------------------------------------------------

import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()

# ---------------------------------------------------------------------
# 1. Load notes
# ---------------------------------------------------------------------
notes_path = "../analysis/data/derivedData/note_sequences_per_patient.npy"
note_sequences = np.load(notes_path, allow_pickle=True).item()

# ---------------------------------------------------------------------
# No filtering by val/train IDs here → tokenize all subjects
# ---------------------------------------------------------------------
print(f"ℹ️ Tokenizing all subjects: {len(note_sequences)} available")

# ---------------------------------------------------------------------
# 2. Detect SEQUENCE_LENGTH dynamically from structured sequences
# ---------------------------------------------------------------------
seq_file = "../analysis/data/derivedData/X_train_transformer.npy"
if os.path.exists(seq_file):
    X_train = np.load(seq_file, mmap_mode="r")
    SEQUENCE_LENGTH = X_train.shape[1]   # dimension T
    print(f"🔄 Detected SEQUENCE_LENGTH={SEQUENCE_LENGTH} from {seq_file}")
else:
    SEQUENCE_LENGTH = 10  # fallback default
    print(f"⚠️ No structured sequences found, defaulting SEQUENCE_LENGTH={SEQUENCE_LENGTH}")

MAX_NOTES_PER_ADMISSION = SEQUENCE_LENGTH
MAX_TOKENS_PER_NOTE = 256
CACHE_DIR = "../analysis/cache/tokenized_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 3. Global Tokenizer for Worker Processes
# ---------------------------------------------------------------------
tokenizer = None
def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# ---------------------------------------------------------------------
# 4. Tokenization function for a single patient
# ---------------------------------------------------------------------
def tokenize_patient(subject_entry):
    global tokenizer
    subject_id, admission_notes = subject_entry
    admission_texts = []

    for note_list in admission_notes[:MAX_NOTES_PER_ADMISSION]:
        if not note_list:
            continue
        joined_note = " ".join(note_list)[:10_000]
        admission_texts.append(joined_note)

    if len(admission_texts) == 0:
        return subject_id, None  # skip empty

    while len(admission_texts) < MAX_NOTES_PER_ADMISSION:
        admission_texts.append("")  # pad with blanks

    encoded = tokenizer.batch_encode_plus(
        admission_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS_PER_NOTE,
        return_tensors="pt"
    )

    return subject_id, {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy()
    }

# ---------------------------------------------------------------------
# 5. Parallel tokenization → collect aligned arrays
# ---------------------------------------------------------------------
all_ids, all_inputs, all_masks = [], [], []
skipped_subjects = []

print("\n🚀 Starting parallel batch tokenization...")

max_workers = min(8, os.cpu_count() or 1)
with ProcessPoolExecutor(max_workers=max_workers, initializer=init_tokenizer) as executor:
    futures = {executor.submit(tokenize_patient, entry): entry[0] for entry in note_sequences.items()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing patients (parallel)"):
        try:
            subject_id, tokenized = future.result()
            if tokenized is None:
                skipped_subjects.append(subject_id)
            else:
                all_ids.append(subject_id)
                all_inputs.append(tokenized["input_ids"])     # (T, L)
                all_masks.append(tokenized["attention_mask"]) # (T, L)
        except Exception as e:
            print(f"Error tokenizing subject {futures[future]}: {e}")
            skipped_subjects.append(futures[future])
        
# ---------------------------------------------------------------------
# 6. Convert lists → NumPy arrays
# ---------------------------------------------------------------------
if not all_inputs:
    raise RuntimeError("❌ No patients successfully tokenized!")

all_ids = np.array(all_ids, dtype=str)
all_inputs = np.stack(all_inputs)   # (N, T, L)
all_masks = np.stack(all_masks)     # (N, T, L)

# ---------------------------------------------------------------------
# 7. Save aligned arrays (fast load at training)
# ---------------------------------------------------------------------
np.save(f"../analysis/data/derivedData/tokenized_input_ids_{args.metric_prefix}.npy", all_inputs)
np.save(f"../analysis/data/derivedData/tokenized_attention_masks_{args.metric_prefix}.npy", all_masks)
np.save(f"../analysis/data/derivedData/tokenized_subject_ids_{args.metric_prefix}.npy", all_ids)

# optional marker
with open("../analysis/logs/tokenization_complete.txt", "w") as f:
    f.write(f"Tokenization completed successfully.\nPatients: {len(all_ids)}\n")

print("\n📋 Parallel Tokenization Summary")
print("--------------------------------------------------")
print(f"🧑‍⚕️ Patients tokenized: {len(all_ids)}")
print(f"🕒 Sequence length (visits): {all_inputs.shape[1]}")
print(f"🔠 Tokens per note: {all_inputs.shape[2]}")
print("✅ Saved aligned .npy arrays (fast training ready)")
