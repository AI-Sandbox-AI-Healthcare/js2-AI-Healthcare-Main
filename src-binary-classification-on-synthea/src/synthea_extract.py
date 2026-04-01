"""
synthea_extract.py
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
load_dotenv()

# ---------------------------------------------------------------------
# 1. Database Connection
# ---------------------------------------------------------------------
HOST = os.getenv("SYNTHEA_HOST", "localhost")
DBNAME = os.getenv("SYNTHEA_DBNAME", "synthea")
USER = os.getenv("SYNTHEA_USER", "hmfattah")
PASSWORD = os.getenv("SYNTHEA_PASSWORD", "hmfattah")  # <-- set in .env in practice
SCHEMA = os.getenv("SYNTHEA_SCHEMA", "synthea")
PORT = int(os.getenv("SYNTHEA_PORT", 5432))

'''
DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text(f"SET search_path TO {SCHEMA};"))
'''
print("Connected to the Synthea database.")

# ---------------------------------------------------------------------
# 2. Load Tables (from CSVs instead of database)
# ---------------------------------------------------------------------
print("Loading tables from CSVs...")

def load_csv(file, usecols=None):
    """Helper: load CSV, lowercase all column names, and subset if needed."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    if usecols:
        df = df[usecols]
    return df

try:
    patients_df = load_csv("./synthea_data/patients.csv")
    conditions_df = load_csv("./synthea_data/conditions.csv")
    medications_df = load_csv("./synthea_data/medications.csv")
    encounters_df = load_csv("./synthea_data/encounters.csv")
    observations_df = load_csv("./synthea_data/observations.csv")
    procedures_df = load_csv("./synthea_data/procedures.csv")
except Exception as e:
    print("Error loading tables:", e)
    raise

print(
    "Loaded:",
    f"patients={len(patients_df):,}\n",
    f"conditions={len(conditions_df):,}\n",
    f"medications={len(medications_df):,}\n",
    f"encounters={len(encounters_df):,}\n",
    f"observations={len(observations_df):,}\n",
    f"procedures={len(procedures_df)}"
)


# ---------------------------
# 3. Normalize column names (lowercase)
# ---------------------------
def lc_cols(df):
    df.columns = [c.lower() for c in df.columns]
    return df

patients_df = lc_cols(patients_df)
conditions_df = lc_cols(conditions_df)
medications_df = lc_cols(medications_df)
encounters_df = lc_cols(encounters_df)
observations_df = lc_cols(observations_df)
procedures_df = lc_cols(procedures_df)

# ---------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------
print("Processing features...")

pain_keywords = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash"
]

pain_pattern = r'\b(?:' + '|'.join(map(re.escape, pain_keywords)) + r')\b'

conditions_df["description"] = conditions_df["description"].str.lower()

pain_patients = conditions_df[
    conditions_df["description"].str.contains(pain_pattern, na=False, regex=True)
]["patient"].unique()

patients_df["binary_label"] = patients_df["id"].isin(pain_patients).astype(int)

patients_df["age"] = (pd.Timestamp.today() - pd.to_datetime(patients_df["birthdate"])).dt.days // 365
patients_df["is_female"] = (patients_df["gender"] == "F").astype(int)

# Convert date columns to datetime
conditions_df["start"] = pd.to_datetime(conditions_df["start"], errors="coerce")
conditions_df["stop"] = pd.to_datetime(conditions_df["stop"], errors="coerce")

# Compute duration in days
conditions_df["duration_days"] = (conditions_df["stop"] - conditions_df["start"]).dt.days

# Fill missing durations with median
median_duration = conditions_df["duration_days"].median(skipna=True)
conditions_df["duration_days"] = conditions_df["duration_days"].fillna(median_duration)

# Aggregate features per patient
cond_features = (
    conditions_df.groupby("patient")
    .agg(
        num_conditions=("description", pd.Series.nunique),
        avg_condition_duration=("duration_days", "mean")
    )
    .reset_index()
    .rename(columns={"patient": "id"})
)

# Merge features
patients_df = pd.merge(patients_df, cond_features, on="id", how="left")
patients_df[["num_conditions", "avg_condition_duration"]] = (patients_df[["num_conditions", "avg_condition_duration"]].fillna(0))

# Convert description to lowercase
medications_df["description"] = medications_df["description"].str.lower()

# Count unique meds per patient
med_feat = (
    medications_df.groupby("patient")["description"]
    .nunique()
    .reset_index()
    .rename(columns={"description": "num_unique_meds", "patient": "id"})
)

# Define common pain-related medications
pain_medications = [
    "opioid", "acetaminophen", "ibuprofen", "gabapentin", "morphine",
    "tramadol", "oxycodone", "hydrocodone", "meperidine", "fentanyl",
    "pregabalin", "naproxen"
]

# Regex pattern for keyword matching
pain_pattern = r'\b(?:' + '|'.join(map(re.escape, pain_medications)) + r')\b'

# Filter for pain-related meds
pain_meds = medications_df[
    medications_df["description"].str.contains(pain_pattern, na=False, regex=True)
]

# Count pain-related meds per patient
pain_feat = (
    pain_meds.groupby("patient")["description"]
    .count()
    .reset_index()
    .rename(columns={"description": "num_pain_meds", "patient": "id"})
)

# Merge both medication features ===
med_features = pd.merge(med_feat, pain_feat, on="id", how="left")
med_features["num_pain_meds"] = med_features["num_pain_meds"].fillna(0).astype(int)

# Merge into patients dataframe ===
patients_df = pd.merge(patients_df, med_features, on="id", how="left")

# Fill missing medication info
patients_df[["num_unique_meds", "num_pain_meds"]] = (patients_df[["num_unique_meds", "num_pain_meds"]].fillna(0).astype(int))

# Count number of encounters per patient ===
encounter_counts = (
    encounters_df.groupby("patient")
    .size()
    .reset_index(name="num_encounters")
    .rename(columns={"patient": "id"})
)

# Merge with patients dataframe ===
patients_df = pd.merge(patients_df, encounter_counts, on="id", how="left")

# Fill patients with 0 encounters
patients_df["num_encounters"] = patients_df["num_encounters"].fillna(0).astype(int)

# Aggregate procedure features per patient ===
proc_feat = (
    procedures_df.groupby("patient")
    .agg(
        num_procedures=("description", "count"),
        unique_procedures=("description", pd.Series.nunique)
    )
    .reset_index()
    .rename(columns={"patient": "id"})
)

# Merge with patients dataframe ===
patients_df = pd.merge(patients_df, proc_feat, on="id", how="left")

# Fill patients with 0 procedures
patients_df["num_procedures"] = patients_df["num_procedures"].fillna(0).astype(int)
patients_df["unique_procedures"] = patients_df["unique_procedures"].fillna(0).astype(int)

columns_to_keep = [
    'id',
    'race',
    'ethnicity',
    'healthcare_expenses',
    'healthcare_coverage',
    'binary_label',
    'age',
    'is_female',
    'num_conditions',
    'avg_condition_duration',
    'num_unique_meds',
    'num_pain_meds',
    'num_encounters',
    'num_procedures',
    'unique_procedures'
]

patients_df = patients_df[columns_to_keep].copy()

# Make sure 'value' is numeric where possible
observations_df["value"] = pd.to_numeric(observations_df["value"], errors="coerce")

# Pivot table ? rows = patient, columns = description, values = value
obs_pivot = observations_df.pivot_table(
    index="patient",
    columns="description",
    values="value",
    aggfunc="mean"   # if multiple measurements per patient, take the mean
).reset_index()

# Clean up column names ? replace non-alphanumeric chars with underscore
obs_pivot.columns = (
    obs_pivot.columns.str.replace('[^0-9a-zA-Z]+', '_', regex=True)
    .str.lower()  # convert all column names to lowercase
)

relevant_features = [
    "pain_severity_0_10_verbal_numeric_rating_score_reported",
    "body_height",
    "body_weight",
    "body_mass_index",
    "body_mass_index_bmi_percentile_per_age_and_gender",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "heart_rate",
    "respiratory_rate",
    "qaly",
    "daly",
    "qols"
]

# Select relevant features (assuming 'patient' column is lowercase)
obs_features = obs_pivot[["patient"] + relevant_features].copy()

# Merge dataframes
merged_df = patients_df.merge(
    obs_features,
    how="left",
    left_on="id",
    right_on="patient"
)

# Drop the redundant PATIENT column
merged_df = merged_df.drop(columns=["patient"])

non_numeric_cols = [col for col in merged_df.select_dtypes(exclude=["number"]).columns if col != "id"]

# Encode categorical columns
le_dict = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    merged_df[col] = merged_df[col].astype(str) 
    merged_df[col] = le.fit_transform(merged_df[col])
    le_dict[col] = le 

merged_df = merged_df.drop(columns=["body_mass_index_bmi_percentile_per_age_and_gender"])

# Impute NaNs with median on numeric columns
numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())


# ---------------------------------------------------------------------
# Save Output
# ---------------------------------------------------------------------
final_cols = ['id', 'race', 'ethnicity', 'healthcare_expenses', 'healthcare_coverage',
       'binary_label', 'age', 'is_female', 'num_conditions',
       'avg_condition_duration', 'num_unique_meds', 'num_pain_meds',
       'num_encounters', 'num_procedures', 'unique_procedures',
       'pain_severity_0_10_verbal_numeric_rating_score_reported',
       'body_height', 'body_weight', 'body_mass_index',
       'systolic_blood_pressure', 'diastolic_blood_pressure', 'heart_rate',
       'respiratory_rate', 'qaly', 'daly', 'qols']

# Class counts
counts = merged_df["binary_label"].value_counts(dropna=False).sort_index()
print("\nClass Distribution Before Saving:")
for label in [0, 1]:
    print(f"  Class {label}: {counts.get(label, 0)}")

summary = {
    "n_rows": len(merged_df),
    "class_counts": counts.to_dict(),
}
with open("synthea_enriched_features_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

output_csv = "synthea_enriched_features.csv"
merged_df[final_cols].to_csv(output_csv, index=False)
print(f"Saved enriched dataset to {output_csv}.")

#engine.dispose()
print("Database connection closed.")
