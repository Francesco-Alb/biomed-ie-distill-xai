import argparse
from pathlib import Path

import pandas as pd

data = [
    ("01", "Aspirin caused severe stomach irritation in the patient.", "Aspirin", "stomach irritation", 1, "causation"),
    ("02", "The patient developed a rash after taking amoxicillin.", "amoxicillin", "rash", 0, "temporal association"),
    ("03", "Amoxicillin did not cause a rash in the patient.", "Amoxicillin", "rash", 0, "negation"),
    ("04", "The patient developed a rash, but denied taking amoxicillin.", "amoxicillin", "rash", 0, "negation"),
    ("05", "The rash might have been caused by amoxicillin, although the cause remains uncertain.", "amoxicillin", "rash", 0, "uncertainty"),
    ("06", "Ibuprofen relieved the patient's headache.", "Ibuprofen", "headache", 2, "treatment"),
    ("07", "Metformin is used to treat type 2 diabetes.", "Metformin", "type 2 diabetes", 2, "treatment"),
    ("08", "Lisinopril is prescribed for hypertension.", "Lisinopril", "hypertension", 2, "treatment"),
    ("09", "Metformin was prescribed for diabetes, while the patient also had hypertension.", "Metformin", "hypertension", 0, "unrelated pair"),
    ("10", "The patient received aspirin and subsequently developed gastrointestinal bleeding.", "aspirin", "gastrointestinal bleeding", 0, "temporal association"),
    ("11", "The patient received aspirin to prevent blood clots.", "aspirin", "blood clots", 2, "prevention"),
    ("12", "Naloxone reversed the respiratory depression caused by morphine.", "Naloxone", "respiratory depression", 3, "antagonism"),
    ("13", "Naloxone reversed the respiratory depression caused by morphine.", "Morphine", "respiratory depression", 1, "causation"),
    ("14", "Morphine was administered to relieve severe pain.", "Morphine", "pain", 2, "treatment"),
    ("15", "The patient received naloxone, but no respiratory depression was observed.", "Naloxone", "respiratory depression", 0, "no relation"),
    ("16", "Metformin treats diabetes, whereas lisinopril treats hypertension.", "Metformin", "hypertension", 0, "multiple entities"),
    ("17", "Metformin treats diabetes, whereas lisinopril treats hypertension.", "Lisinopril", "hypertension", 2, "multiple entities"),
    ("18", "Metformin may cause lactic acidosis in some patients.", "Metformin", "lactic acidosis", 1, "causation"),
    ("19", "Metformin is commonly used for diabetes; lactic acidosis was mentioned as a possible complication, but was not observed.", "Metformin", "lactic acidosis", 0, "uncertainty"),
    ("20", "The patient started amoxicillin. Two days later, a skin rash appeared, which the physician attributed to the medication.", "Amoxicillin", "skin rash", 1, "document context"),
    ("21", "The patient started amoxicillin. Two days later, a skin rash appeared, but the physician attributed it to a viral infection.", "Amoxicillin", "skin rash", 0, "document context"),
    ("22", "The patient received both morphine and naloxone. Naloxone counteracted the respiratory depression induced by morphine.", "Naloxone", "respiratory depression", 3, "antagonism"),
]

df = pd.DataFrame(
    data,
    columns=[
        "id", "text", "target_chemical", "target_disease",
        "expected_multiclass", "test_category"
    ]
)

# Binary task: only causal relations count as positive
df["expected_binary"] = (df["expected_multiclass"] == 1).astype(int)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "data",
    help="Directory where diagnostic_set.csv will be saved.",
)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output_dir / "diagnostic_set.csv", index=False)