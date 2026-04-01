"""train_lstm_mimic.py
Binary LSTM baseline with shared validation splits.
Saves stacking probs + metrics + confusion/ROC plots.
"""

import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from resource_logger import ResourceLogger
import argparse

# ---------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# 1. Dataset & Model
# ---------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, seq, labels, subj_ids):
        self.X, self.y, self.sids = seq, labels, subj_ids
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.sids[idx],
        )

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)  # single output for sigmoid
    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits  # raw logits for BCEWithLogitsLoss

# ---------------------------------------------------------------------
# 2. CLI & Paths
# ---------------------------------------------------------------------
BASE = "."
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix

# ---------------------------------------------------------------------
# 3. Load train/val split arrays
# ---------------------------------------------------------------------
X_train = np.load(f"{BASE}/X_train_transformer.npy")
y_train = np.load(f"{BASE}/y_train_transformer.npy")
sid_train = np.load(f"{BASE}/subject_ids_train_transformer.npy")

X_val = np.load(f"{BASE}/X_val_transformer.npy")
y_val = np.load(f"{BASE}/y_val_transformer.npy")
sid_val = np.load(f"{BASE}/subject_ids_val_transformer.npy")

print(f"Train size: {len(y_train)} | Val size: {len(y_val)}")

train_loader = DataLoader(
    SequenceDataset(X_train, y_train, sid_train),
    batch_size=32, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    SequenceDataset(X_val, y_val, sid_val),
    batch_size=32, shuffle=False, num_workers=2
)

# ---------------------------------------------------------------------
# 4. Model / Optim / Loss
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Compute positive class weight (for imbalance)
pos_weight = torch.tensor(
    [len(y_train) / (y_train.sum() + 1e-6) - 1.0],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

BEST_PATH = f"{BASE}/lstm_model_{METRIC_PREFIX}.pt"
METRIC_CSV = f"{BASE}/lstm_metrics_{METRIC_PREFIX}.csv"

# ---------------------------------------------------------------------
# 5. Train + validate with ResourceLogger
# ---------------------------------------------------------------------
with ResourceLogger(tag=f"lstm_binary_{METRIC_PREFIX}"):
    best_val, patience, counter = float("inf"), 3, 0
    for epoch in range(20):
        # --- Train ---
        model.train(); tr_loss = 0
        for Xb, yb, _ in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb.unsqueeze(1))
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | TrainLoss {tr_loss/len(train_loader):.4f}")

        # --- Validate ---
        model.eval(); val_loss=0; preds=[]; trues=[]; all_probs=[]; subj_out=[]
        with torch.no_grad():
            for Xb, yb, sids in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb.unsqueeze(1))
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend((probs > 0.5).astype(int))
                all_probs.extend(probs)
                trues.extend(yb.cpu().numpy())
                subj_out.extend(sids)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d} | ValLoss  {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss; counter = 0
            torch.save(model.state_dict(), BEST_PATH)
            print("  ✓ checkpoint saved")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

# ---------------------------------------------------------------------
# 6. Final Evaluation
# ---------------------------------------------------------------------
model.load_state_dict(torch.load(BEST_PATH)); model.eval()
preds, trues, all_probs, subj_out = [], [], [], []
with torch.no_grad():
    for Xb, yb, sids in val_loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds.extend((probs > 0.5).astype(int))
        all_probs.extend(probs)
        trues.extend(yb.numpy())
        subj_out.extend(sids)

preds, trues = np.array(preds), np.array(trues)
probs = np.array(all_probs)
subj_out = np.array(subj_out)
acc = accuracy_score(trues, preds)
auc = roc_auc_score(trues, probs)

# Save metrics
with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc])
print(f"Metrics → {METRIC_CSV}")

# Confusion matrix
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("LSTM Binary Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{BASE}/lstm_confusion_{METRIC_PREFIX}.png")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(trues, probs)
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("LSTM Binary ROC")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{BASE}/lstm_roc_curve_{METRIC_PREFIX}.png")
plt.close()

# Save stacking probabilities
np.savez_compressed(
    f"{BASE}/lstm_probs_{METRIC_PREFIX}.npz",
    probs=probs,
    y_true=trues,
    subject_ids=subj_out
)
print(f"Saved LSTM probs → lstm_probs_{METRIC_PREFIX}.npz")

print(f"✅ Finished LSTM training with shared val IDs [{METRIC_PREFIX}]")
