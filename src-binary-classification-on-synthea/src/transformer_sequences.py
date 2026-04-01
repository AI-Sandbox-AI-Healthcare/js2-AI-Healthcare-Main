import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)

# ---------------------------------------------------------------------
# 1) Load
# ---------------------------------------------------------------------
print("🔹 Loading data...")
df = pd.read_csv("synthea_enriched_features_w_notes.csv")

required_cols = {
    'id', 'race', 'ethnicity', 'healthcare_expenses', 'healthcare_coverage',
       'binary_label', 'age', 'is_female', 'num_conditions',
       'avg_condition_duration', 'num_unique_meds', 'num_pain_meds',
       'num_encounters', 'num_procedures', 'unique_procedures',
       'pain_severity_0_10_verbal_numeric_rating_score_reported',
       'body_height', 'body_weight', 'body_mass_index',
       'systolic_blood_pressure', 'diastolic_blood_pressure', 'heart_rate',
       'respiratory_rate', 'qaly', 'daly', 'qols', 'sentiment', 'note_cluster'
}
tfidf_terms = [
    "opioid", "acetaminophen", "ibuprofen", "gabapentin", "morphine",
    "tramadol", "oxycodone", "hydrocodone", "meperidine", "fentanyl",
    "pregabalin", "naproxen"
]
topic_cols = [f"topic_{i+1}" for i in range(5)]
feature_cols = list(required_cols - {"id", "binary_label"}) \
               + [f"tfidf_{t}" for t in tfidf_terms] + topic_cols

missing = ({"id", "binary_label"} | set(feature_cols)) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

label_col = "binary_label"

# ---------------------------------------------------------------------
# 2) Preprocess
# ---------------------------------------------------------------------
print("🔹 Preprocessing...")
df[feature_cols] = df[feature_cols].fillna(0)
df = df.dropna(subset=[label_col, "id"])
df[label_col] = df[label_col].astype(int)

# enforce known classes (0,1)
bad_labels = set(df[label_col].unique()) - {0, 1}
if bad_labels:
    raise ValueError(f"Unexpected labels found: {bad_labels}. Expected only {{0,1}}.")

# ---------------------------------------------------------------------
# 3) Build sequences + visit masks (left-pad) per subject
# ---------------------------------------------------------------------
print("🔹 Building sequences...")
SEQUENCE_LENGTH = 10

sequences, labels, masks, subj_ids = [], [], [], []

for subject_id, group in df.groupby("id", sort=False):
    visit_feats = group[feature_cols].to_numpy(dtype=np.float32)  # (n_visits, F)
    label_seq = group[label_col].to_numpy()

    if visit_feats.size == 0:
        continue

    # keep last SEQUENCE_LENGTH visits
    if len(visit_feats) >= SEQUENCE_LENGTH:
        visit_feats = visit_feats[-SEQUENCE_LENGTH:]
        mask = np.ones(SEQUENCE_LENGTH, dtype=np.float32)
        label = int(label_seq[-1])
    else:
        pad_len = SEQUENCE_LENGTH - len(visit_feats)
        visit_feats = np.pad(visit_feats, ((pad_len, 0), (0, 0)), mode="constant")
        mask = np.concatenate([np.zeros(pad_len, dtype=np.float32),
                               np.ones(SEQUENCE_LENGTH - pad_len, dtype=np.float32)])
        label = int(label_seq[-1])

    sequences.append(visit_feats)
    labels.append(label)
    masks.append(mask)
    subj_ids.append(subject_id)

X = np.stack(sequences)              # (N, T, F)
y = np.array(labels, dtype=np.int64) # (N,)
M = np.stack(masks)                  # (N, T)
S = np.array(subj_ids, dtype=str)  # (N,)

print(f"   → Built {len(y)} subject sequences; shape X={X.shape}, mask={M.shape}")

# ---------------------------------------------------------------------
# 4) Train/Val split (carry masks + subject_ids)
# ---------------------------------------------------------------------
print("🔹 Splitting train/val...")

X_train, X_val, y_train, y_val, m_train, m_val, sid_train, sid_val = train_test_split(
    X, y, M, S, stratify=y, test_size=0.2, random_state=SEED
)

def check_class_coverage(y_arr, name):
    classes, counts = np.unique(y_arr, return_counts=True)
    print(f"📦 {name} class counts:", dict(zip(classes.tolist(), counts.tolist())))
    missing = set([0, 1]) - set(classes.tolist())
    if missing:
        print(f"⚠️ WARNING: {name} missing classes: {missing}")
    return missing

missing = check_class_coverage(y_train, "Train")
check_class_coverage(y_val, "Validation")
if missing:
    raise ValueError(f"🚨 Training data missing class(es): {missing}")

# ---------------------------------------------------------------------
# 5) Optional oversampling for class 2 on TRAIN only (keeps masks + IDs aligned)
# ---------------------------------------------------------------------
OVERSAMPLE_TARGET = int(os.getenv("OVERSAMPLE_TARGET", 100))
train_counts = Counter(y_train)
minority_class = 1

if train_counts.get(minority_class, 0) < OVERSAMPLE_TARGET:
    print("⚠️ Oversampling class 2 (co-morbid)...")

    idx_min = np.where(y_train == minority_class)[0]
    if len(idx_min) > 0:
        n_needed = OVERSAMPLE_TARGET - len(idx_min)
        X_min = X_train[idx_min]
        y_min = y_train[idx_min]
        m_min = m_train[idx_min]
        sid_min = sid_train[idx_min]

        X_up, y_up, m_up, sid_up = resample(
            X_min, y_min, m_min, sid_min,
            replace=True, n_samples=n_needed, random_state=SEED
        )

        X_train = np.concatenate([X_train, X_up], axis=0)
        y_train = np.concatenate([y_train, y_up], axis=0)
        m_train = np.concatenate([m_train, m_up], axis=0)
        sid_train = np.concatenate([sid_train, sid_up], axis=0)

        print("✅ Oversampling complete.")
        check_class_coverage(y_train, "Train (post-oversample)")
    else:
        print("⚠️ No samples of class 2 in training to oversample from.")

# Standardization
scaler = StandardScaler()

# reshape to (N*T, F) to fit the scaler
N, T, F = X_train.shape
X_train_flat = X_train.reshape(-1, F)
X_train_scaled_flat = scaler.fit_transform(X_train_flat)
X_train = X_train_scaled_flat.reshape(N, T, F)

# same for validation using the **same scaler**
N_val, T_val, F_val = X_val.shape
X_val_flat = X_val.reshape(-1, F_val)
X_val_scaled_flat = scaler.transform(X_val_flat)
X_val = X_val_scaled_flat.reshape(N_val, T_val, F_val)

# ---------------------------------------------------------------------
# 6) Save outputs (features, labels, masks, subject_ids, feature list)
# ---------------------------------------------------------------------
print("🔹 Saving sequences...")
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X_train_transformer.npy", X_train)
np.save(f"{out_dir}/y_train_transformer.npy", y_train)
np.save(f"{out_dir}/mask_train_transformer.npy", m_train)
np.save(f"{out_dir}/subject_ids_train_transformer.npy", sid_train)

np.save(f"{out_dir}/X_val_transformer.npy", X_val)
np.save(f"{out_dir}/y_val_transformer.npy", y_val)
np.save(f"{out_dir}/mask_val_transformer.npy", m_val)
np.save(f"{out_dir}/subject_ids_val_transformer.npy", sid_val)

with open(f"{out_dir}/feature_cols_transformer.txt", "w") as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# ---------------------------------------------------------------------
# 7) Summary
# ---------------------------------------------------------------------
print("\n📋 Transformer Sequence Summary")
print("--------------------------------------------------")
print(f"🔢 Features per visit: {X.shape[2]}")
print(f"🧮 Sequence length (T): {X.shape[1]}")
print(f"👩‍💻 Training samples: {X_train.shape[0]}")
print(f"👨‍💻 Validation samples: {X_val.shape[0]}")
print(f"🪪 Train subject IDs:   {sid_train.shape[0]}")
print(f"🪪 Val subject IDs:     {sid_val.shape[0]}")
print(f"🔎 Seed used: {SEED}")
print("✅ Saved Transformer-ready sequences, visit masks, and subject IDs.")
