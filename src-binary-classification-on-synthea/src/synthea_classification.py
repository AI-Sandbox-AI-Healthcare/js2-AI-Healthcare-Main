"""synthea_classification.py
Tabular baseline (RandomForest + XGBoost) with resource logging and metrics output.
Supports METRIC_PREFIX for iteration-based benchmarking.
"""

import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
from resource_logger import ResourceLogger
import argparse

# ------------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)

BASE = "./"
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")

# ------------------------------------------------------------------
# 1. Load & preprocess
# ------------------------------------------------------------------
df = pd.read_csv(f"{BASE}/synthea_enriched_features.csv")

FEATURES = [
    'race', 'ethnicity', 'healthcare_expenses', 'healthcare_coverage', 'age', 'is_female', 'num_conditions',
       'avg_condition_duration', 'num_unique_meds', 'num_pain_meds',
       'num_encounters', 'num_procedures', 'unique_procedures',
       'pain_severity_0_10_verbal_numeric_rating_score_reported',
       'body_height', 'body_weight', 'body_mass_index',
       'systolic_blood_pressure', 'diastolic_blood_pressure', 'heart_rate',
       'respiratory_rate', 'qaly', 'daly', 'qols']

X = df[FEATURES]
y = df["binary_label"]
if y.isnull().any():
    print("⚠️ Dropping rows with missing binary_label...")
    df = df[~y.isnull()]
    y = df["binary_label"]

y = y.astype(int)

# ------------------------------------------------------------------
# 2. Train/val split with shared validation IDs
# ------------------------------------------------------------------
val_ids = np.load(f"shared_val_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
is_val = df["id"].isin(val_ids)
df_train = df[~is_val]
df_test  = df[is_val]

X_train = df_train[FEATURES]
y_train = df_train["binary_label"].astype(int)
X_test  = df_test[FEATURES]
y_test  = df_test["binary_label"].astype(int)

subj_train = df_train["id"].values
subj_test  = df_test["id"].values

cat_indices = [FEATURES.index("race"), FEATURES.index("ethnicity"), FEATURES.index("is_female")]
scaler = StandardScaler()

# ------------------------------------------------------------------
# 3. Resampling search
# ------------------------------------------------------------------
def resample_data(X, y, method, categorical_features=None):
    if method == "smote":
        sampler = SMOTE(random_state=SEED)
    elif method == "adasyn":
        sampler = ADASYN(random_state=SEED, sampling_strategy='not minority')
    elif method == "borderline":
        sampler = BorderlineSMOTE(random_state=SEED)
    elif method == "smotenc":
        if categorical_features is None:
            raise ValueError("categorical_features must be provided for SMOTE-NC")
        sampler = SMOTENC(categorical_features=categorical_features, random_state=SEED)
    else:
        raise ValueError(f"Unknown method: {method}")
    return sampler.fit_resample(X, y)

best_method, best_score, best_data = None, -np.inf, None
methods = ["smote", "adasyn", "borderline", "smotenc"]

for method in methods:
    print(f"🔁 Testing {method.upper()}")
    if method == "smotenc":
        X_res, y_res = resample_data(X_train, y_train, method, cat_indices)
    else:
        X_res, y_res = resample_data(X_train, y_train, method)
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
    rf.fit(X_res_scaled, y_res)
    f1_rf = classification_report(y_test, rf.predict(X_test_scaled),
                                  output_dict=True)["macro avg"]["f1-score"]

    if f1_rf > best_score:
        best_score = f1_rf
        best_method = method
        best_data = (X_res_scaled, y_res)
        print(f"✅ New best method: {method} (Macro F1: {f1_rf:.3f})")

# ------------------------------------------------------------------
# 4. Finalize best data
# ------------------------------------------------------------------
if best_data is None:
    raise RuntimeError("No resampling method produced a valid dataset.")
X_train_final, y_train_final = best_data
X_test_final = scaler.transform(X_test)

# ------------------------------------------------------------------
# 5. Train & evaluate inside ResourceLogger
# ------------------------------------------------------------------
with ResourceLogger(tag=f"tabular_rf_xgb_{METRIC_PREFIX}"):
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
    rf.fit(X_train_final, y_train_final)
    prob_rf = rf.predict_proba(X_test_final)[:,1]
    pred_rf = rf.predict(X_test_final)
    auc_rf = roc_auc_score(y_test, prob_rf, multi_class="ovr")

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=SEED)
    xgb.fit(X_train_final, y_train_final)
    prob_xgb = xgb.predict_proba(X_test_final)[:,1]
    pred_xgb = xgb.predict(X_test_final)
    auc_xgb = roc_auc_score(y_test, prob_xgb, multi_class="ovr")

# ------------------------------------------------------------------
# 6. Save models & outputs
# ------------------------------------------------------------------
joblib.dump(rf, f"{BASE}/synthea_randomforest_model_{METRIC_PREFIX}.pkl")
joblib.dump(xgb, f"{BASE}/synthea_xgb_model_{METRIC_PREFIX}.pkl")
joblib.dump(scaler, f"{BASE}/scaler_{METRIC_PREFIX}.pkl")

np.savez_compressed(
    f"{BASE}/rf_probs_{METRIC_PREFIX}.npz",
    probs=prob_rf,
    y_true=y_test,
    subject_ids=subj_test
)

np.savez_compressed(
    f"{BASE}/xgb_probs_{METRIC_PREFIX}.npz",
    probs=prob_xgb,
    y_true=y_test,
    subject_ids=subj_test
)

# ------------------------------------------------------------------
# 7. Metrics + ROC curve
# ------------------------------------------------------------------
with open(f"{BASE}/tabular_metrics_{METRIC_PREFIX}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "AUC"])
    writer.writerow(["RandomForest", f"{auc_rf:.4f}"])
    writer.writerow(["XGBoost", f"{auc_xgb:.4f}"])

# ROC curve (XGB example)
fpr, tpr, _ = roc_curve(y_test, prob_xgb)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_xgb:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("XGBoost ROC (Binary)")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{BASE}/xgb_roc_curve_{METRIC_PREFIX}.png")
plt.close()

print(f"📊 Metrics saved using best resampling method: {best_method.upper()} (Macro F1: {best_score:.3f})")
print("\nRandomForest Report:")
print(classification_report(y_test, pred_rf, zero_division=0))
print("\nXGBoost Report:")
print(classification_report(y_test, pred_xgb, zero_division=0))
print("Class distribution (train after resampling):", np.bincount(y_train_final))
