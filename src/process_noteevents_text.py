import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. Load NOTEEVENTS Data
# ---------------------------------------------------------------------
note_path = "./NOTEEVENTS.csv"
notes_df = pd.read_csv(note_path, low_memory=False)

# Drop empty or malformed notes
notes_df = notes_df[notes_df['TEXT'].notna() & notes_df['HADM_ID'].notna()]

# Optional: filter for relevant note types
keep_categories = ['Discharge summary', 'Nursing', 'Physician']
notes_df = notes_df[notes_df['CATEGORY'].isin(keep_categories)]

# ---------------------------------------------------------------------
# 2. Clean and Preprocess Text
# ---------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Collapse all whitespace
    text = re.sub(r"_+", " ", text)   # Remove underlines
    text = text.replace("[**", "").replace("**]", "")  # Remove anonymization markers
    text = text.strip()
    return text

# Drop common metadata-only templates
drop_patterns = ["dictated by", "signed electronically"]
notes_df = notes_df[~notes_df['TEXT'].str.lower().str.contains('|'.join(drop_patterns), na=False)]

# Clean content
notes_df['TEXT'] = notes_df['TEXT'].map(clean_text)

# ---------------------------------------------------------------------
# 3. Aggregate Notes by SUBJECT_ID and HADM_ID
# ---------------------------------------------------------------------
notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'], errors='coerce')
notes_df = notes_df.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'])

# Create hierarchical structure: subject → admission → notes
notes_grouped = defaultdict(lambda: defaultdict(list))

for _, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
    subj = int(row['SUBJECT_ID'])
    hadm = int(row['HADM_ID'])
    notes_grouped[subj][hadm].append(row['TEXT'])

# ---------------------------------------------------------------------
# 4. Postprocess: Group notes per admission
# ---------------------------------------------------------------------
note_sequences = {}
min_note_len = 40

for subj_id, hadm_dict in notes_grouped.items():
    grouped = []
    for note_list in hadm_dict.values():
        # Combine short notes with buffer
        combined, buffer = [], ""
        for note in note_list:
            if len(note) < min_note_len:
                buffer += " " + note
            else:
                if buffer:
                    combined.append(buffer.strip())
                    buffer = ""
                combined.append(note)
        if buffer:
            combined.append(buffer.strip())
        grouped.append(combined)
    note_sequences[subj_id] = grouped

# ---------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------
out_path = "./note_sequences_per_patient.npy"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.save(out_path, note_sequences)

# ---------------------------------------------------------------------
# 6. Stats
# ---------------------------------------------------------------------
note_lengths = [len(adm) for subj in note_sequences.values() for adm in subj]
print(f"\nSaved cleaned note sequences → {out_path}")
print(f"Total patients:      {len(note_sequences):,}")
print(f"Admissions per patient: mean={np.mean([len(v) for v in note_sequences.values()]):.1f}")
print(f"Notes per admission:  mean={np.mean(note_lengths):.1f} | 95%ile={np.percentile(note_lengths,95):.0f} | max={np.max(note_lengths)}")
