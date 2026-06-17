# =========================
# Task A BioMM-Eval
# TCGA-SKCM: shortcut-controlled evaluation
# Uses existing fixed splits:
# - taskA_test_full.csv
# - taskA_test_balanced.csv
# =========================

import os, json, random, time, traceback
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# -------------------------
# Config
# -------------------------

# API_KEY = "you api key"  
MODEL = "gpt-5.4-mini"

BASE_DIR = ""
OUT_DIR = os.path.join(BASE_DIR, "biomm_taskA_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Existing fixed split files. Do not regenerate.
TEST_FULL_CSV = os.path.join(OUT_DIR, "taskA_test_full.csv")
TEST_BAL_CSV = os.path.join(OUT_DIR, "taskA_test_balanced.csv")

# "shortcut_present" or "shortcut_controlled"
TASK_SETTING = "shortcut_controlled"

# "balanced" or "full"
EVAL_MODE = "full"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_JSONL = os.path.join(
    OUT_DIR, f"taskA_{TASK_SETTING}_{EVAL_MODE}_results_{RUN_ID}.jsonl"
)
FAILED_JSONL = os.path.join(
    OUT_DIR, f"taskA_{TASK_SETTING}_{EVAL_MODE}_failed_{RUN_ID}.jsonl"
)
METRICS_CSV = os.path.join(
    OUT_DIR, f"taskA_{TASK_SETTING}_{EVAL_MODE}_metrics_{RUN_ID}.csv"
)
RELIANCE_CSV = os.path.join(
    OUT_DIR, f"taskA_{TASK_SETTING}_{EVAL_MODE}_reliance_{RUN_ID}.csv"
)
REPORT_TXT = os.path.join(
    OUT_DIR, f"taskA_{TASK_SETTING}_{EVAL_MODE}_report_{RUN_ID}.txt"
)

TARGET_COL = "sample_type"
CLASSES = ["Primary Tumor", "Metastatic"]

SEED = 42
random.seed(SEED)

VARIANTS = [
    "full",
    "clinical_only",
    "rna_only",
    "image_summary_only",
    "clinical_rna",
    "counterfactual_clinical",
    "counterfactual_rna"
]

client = OpenAI(api_key=API_KEY)

# -------------------------
# Helpers
# -------------------------

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clean_val(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip()
    if x.lower() in ["", "nan", "none", "null", "not reported", "not available"]:
        return "unknown"
    return x


def load_existing_test_set(mode=EVAL_MODE):
    if mode == "full":
        path = TEST_FULL_CSV
        split_name = "FULL"
    elif mode == "balanced":
        path = TEST_BAL_CSV
        split_name = "BALANCED"
    else:
        raise ValueError("EVAL_MODE must be 'full' or 'balanced'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Existing split not found: {path}")

    df = pd.read_csv(path)

    required = [
        "case_barcode",
        TARGET_COL,
        "has_image",
        "has_rna",
        "age_at_diagnosis",
        "gender",
        "expr_mean"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in existing split: {missing}")

    print(f"\nUsing EXISTING {split_name} test set")
    print("split file:", path)
    print(df[TARGET_COL].value_counts())
    print("n =", len(df))

    return df


# -------------------------
# Prompt blocks
# -------------------------

def clinical_block(row, counterfactual=False):
    age = row.get("age_at_diagnosis", np.nan)
    gender = clean_val(row.get("gender"))
    race = clean_val(row.get("race"))
    vital = clean_val(row.get("vital_status"))
    stage = clean_val(row.get("ajcc_pathologic_stage"))
    tumor_stage = clean_val(row.get("tumor_stage"))
    primary_diagnosis = clean_val(row.get("primary_diagnosis"))

    tissue_type = clean_val(row.get("tissue_type"))
    tumor_descriptor = clean_val(row.get("tumor_descriptor"))

    if counterfactual:
        if gender == "male":
            gender = "female"
        elif gender == "female":
            gender = "male"

        try:
            age = float(age)
            age = max(20, min(95, 100 - age))
        except Exception:
            age = "unknown"

        stage_choices = ["Stage I", "Stage II", "Stage III", "Stage IV", "unknown"]
        stage = random.choice([s for s in stage_choices if s != stage])

        tumor_stage_choices = ["stage i", "stage ii", "stage iii", "stage iv", "unknown"]
        tumor_stage = random.choice([s for s in tumor_stage_choices if s != tumor_stage])

        if TASK_SETTING == "shortcut_present":
            tumor_descriptor_choices = ["Primary", "Metastatic", "Recurrence", "unknown"]
            tumor_descriptor = random.choice(
                [s for s in tumor_descriptor_choices if s != tumor_descriptor]
            )

    if TASK_SETTING == "shortcut_present":
        return f"""
Clinical metadata:
- age_at_diagnosis: {age}
- gender: {gender}
- race: {race}
- vital_status: {vital}
- primary_diagnosis: {primary_diagnosis}
- tumor_stage: {tumor_stage}
- ajcc_pathologic_stage: {stage}
- tissue_type: {tissue_type}
- tumor_descriptor: {tumor_descriptor}
""".strip()

    if TASK_SETTING == "shortcut_controlled":
        return f"""
Clinical metadata:
- age_at_diagnosis: {age}
- gender: {gender}
- race: {race}
- vital_status: {vital}
- primary_diagnosis: {primary_diagnosis}
- tumor_stage: {tumor_stage}
- ajcc_pathologic_stage: {stage}
""".strip()

    raise ValueError("TASK_SETTING must be 'shortcut_present' or 'shortcut_controlled'")


def rna_block(row, counterfactual=False):
    cols = [
        "expr_n_genes",
        "expr_mean",
        "expr_std",
        "expr_median",
        "expr_q25",
        "expr_q75",
        "expr_min",
        "expr_max",
        "expr_nonzero_fraction"
    ]

    values = {}
    for c in cols:
        v = row.get(c, np.nan)
        values[c] = None if pd.isna(v) else float(v)

    if counterfactual:
        for c in ["expr_mean", "expr_std", "expr_median", "expr_q25", "expr_q75", "expr_max"]:
            if values.get(c) is not None:
                factor = random.choice([0.5, 0.7, 1.3, 1.5])
                values[c] = values[c] * factor

        if values.get("expr_nonzero_fraction") is not None:
            values["expr_nonzero_fraction"] = max(
                0.0,
                min(1.0, 1.0 - values["expr_nonzero_fraction"])
            )

    return f"""
RNA expression summary:
- expr_n_genes: {values["expr_n_genes"]}
- expr_mean: {values["expr_mean"]}
- expr_std: {values["expr_std"]}
- expr_median: {values["expr_median"]}
- expr_q25: {values["expr_q25"]}
- expr_q75: {values["expr_q75"]}
- expr_min: {values["expr_min"]}
- expr_max: {values["expr_max"]}
- expr_nonzero_fraction: {values["expr_nonzero_fraction"]}
""".strip()


def image_summary_block(row):
    return f"""
Image availability metadata:
- has_image: {clean_val(row.get("has_image"))}
- n_series: {clean_val(row.get("n_series"))}
- n_studies: {clean_val(row.get("n_studies"))}
- image_modalities: {clean_val(row.get("image_modalities"))}
""".strip()


def build_prompt(row, variant):
    if variant == "full":
        instruction = "You are given image availability metadata, clinical metadata, and RNA expression summary."
        blocks = [
            image_summary_block(row),
            clinical_block(row),
            rna_block(row)
        ]

    elif variant == "clinical_only":
        instruction = "You are given only clinical metadata."
        blocks = [clinical_block(row)]

    elif variant == "rna_only":
        instruction = "You are given only RNA expression summary."
        blocks = [rna_block(row)]

    elif variant == "image_summary_only":
        instruction = "You are given only image availability metadata. No pixel-level pathology image is provided."
        blocks = [image_summary_block(row)]

    elif variant == "clinical_rna":
        instruction = "You are given clinical metadata and RNA expression summary."
        blocks = [
            clinical_block(row),
            rna_block(row)
        ]

    elif variant == "counterfactual_clinical":
        instruction = "You are given image availability metadata, modified clinical metadata, and original RNA expression summary."
        blocks = [
            image_summary_block(row),
            clinical_block(row, counterfactual=True),
            rna_block(row)
        ]

    elif variant == "counterfactual_rna":
        instruction = "You are given image availability metadata, original clinical metadata, and perturbed RNA expression summary."
        blocks = [
            image_summary_block(row),
            clinical_block(row),
            rna_block(row, counterfactual=True)
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    joined = "\n\n".join(blocks)

    prompt = f"""
Task:
Predict the TCGA-SKCM sample type.

Allowed labels:
- Primary Tumor
- Metastatic

{instruction}

{joined}

Output requirements:
- Return only valid JSON.
- prediction must be exactly "Primary Tumor" or "Metastatic".
- confidence must be between 0 and 1.
- image_reliance, clinical_reliance, and rna_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance should be 0.

JSON fields:
prediction
confidence
used_modalities
image_reliance
clinical_reliance
rna_reliance
rationale_short
""".strip()

    return prompt


def call_model(row, variant):
    prompt = build_prompt(row, variant)

    response = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "taskA_prediction",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prediction": {
                            "type": "string",
                            "enum": CLASSES
                        },
                        "confidence": {
                            "type": "number"
                        },
                        "used_modalities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["image_summary", "clinical", "rna"]
                            }
                        },
                        "image_reliance": {
                            "type": "number"
                        },
                        "clinical_reliance": {
                            "type": "number"
                        },
                        "rna_reliance": {
                            "type": "number"
                        },
                        "rationale_short": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "prediction",
                        "confidence",
                        "used_modalities",
                        "image_reliance",
                        "clinical_reliance",
                        "rna_reliance",
                        "rationale_short"
                    ]
                },
                "strict": True
            }
        }
    )

    return json.loads(response.output_text)


# -------------------------
# Metrics
# -------------------------

def compute_metrics(results_path):
    res = pd.read_json(results_path, lines=True)

    rows = []
    report_lines = []

    for variant, g in res.groupby("variant"):
        g = g.dropna(subset=["prediction", "true_label"]).copy()
        g = g[g["prediction"].isin(CLASSES)]
        g = g[g["true_label"].isin(CLASSES)]

        if len(g) == 0:
            continue

        y_true = g["true_label"]
        y_pred = g["prediction"]

        rows.append({
            "task_setting": TASK_SETTING,
            "eval_mode": EVAL_MODE,
            "variant": variant,
            "n": len(g),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mean_confidence": float(g["confidence"].mean()),
            "mean_image_reliance_self_report": float(g["image_reliance"].mean()),
            "mean_clinical_reliance_self_report": float(g["clinical_reliance"].mean()),
            "mean_rna_reliance_self_report": float(g["rna_reliance"].mean())
        })

        report_lines.append(f"\n\n===== {variant} =====\n")
        report_lines.append(
            classification_report(
                y_true,
                y_pred,
                labels=CLASSES,
                zero_division=0
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_CSV, index=False)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = metrics.set_index("variant")["balanced_accuracy"].to_dict()

    full = score.get("full")
    clinical = score.get("clinical_only")
    rna = score.get("rna_only")
    image = score.get("image_summary_only")
    clinical_rna = score.get("clinical_rna")
    cf_clinical = score.get("counterfactual_clinical")
    cf_rna = score.get("counterfactual_rna")

    unimodal_scores = [x for x in [clinical, rna, image] if x is not None]

    reliance = {
        "task_setting": TASK_SETTING,
        "eval_mode": EVAL_MODE,

        "full_balanced_accuracy": full,
        "clinical_only_balanced_accuracy": clinical,
        "rna_only_balanced_accuracy": rna,
        "image_summary_only_balanced_accuracy": image,
        "clinical_rna_balanced_accuracy": clinical_rna,
        "counterfactual_clinical_balanced_accuracy": cf_clinical,
        "counterfactual_rna_balanced_accuracy": cf_rna,

        "integration_gain_over_best_unimodal": None if full is None or not unimodal_scores else full - max(unimodal_scores),
        "clinical_shortcut_advantage_over_full": None if full is None or clinical is None else clinical - full,

        "clinical_necessity_full_minus_rna_only": None if full is None or rna is None else full - rna,
        "rna_necessity_full_minus_clinical_only": None if full is None or clinical is None else full - clinical,
        "image_summary_necessity_full_minus_clinical_rna": None if full is None or clinical_rna is None else full - clinical_rna,

        "clinical_counterfactual_sensitivity_full_minus_cf": None if full is None or cf_clinical is None else full - cf_clinical,
        "rna_counterfactual_sensitivity_full_minus_cf": None if full is None or cf_rna is None else full - cf_rna,
    }

    wide = res.pivot_table(
        index="case_barcode",
        columns="variant",
        values="prediction",
        aggfunc="first"
    ).reset_index()

    for other in [
        "clinical_only",
        "rna_only",
        "image_summary_only",
        "clinical_rna",
        "counterfactual_clinical",
        "counterfactual_rna"
    ]:
        key = f"full_vs_{other}_prediction_change_rate"
        if "full" in wide.columns and other in wide.columns:
            valid = wide.dropna(subset=["full", other]).copy()
            reliance[key] = float((valid["full"] != valid[other]).mean()) if len(valid) else None
        else:
            reliance[key] = None

    reliance_df = pd.DataFrame([reliance])
    reliance_df.to_csv(RELIANCE_CSV, index=False)

    print("\n=== Task A metrics by variant ===")
    print(metrics.to_string(index=False))

    print("\n=== Task A reliance summary ===")
    print(reliance_df.to_string(index=False))

    print("\nSaved files:")
    print("results:", RESULTS_JSONL)
    print("metrics:", METRICS_CSV)
    print("reliance:", RELIANCE_CSV)
    print("report:", REPORT_TXT)


# -------------------------
# Main
# -------------------------

def main():
    df = load_existing_test_set(EVAL_MODE)

    preview_cols = [
        "case_barcode",
        TARGET_COL,
        "gender",
        "age_at_diagnosis",
        "tumor_descriptor",
        "tissue_type",
        "has_image",
        "has_rna"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nSamples:", len(df))
    print(df[preview_cols].head().to_string(index=False))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        for variant in VARIANTS:
            try:
                pred = call_model(row, variant)

                out = {
                    "task_setting": TASK_SETTING,
                    "eval_mode": EVAL_MODE,
                    "case_barcode": row["case_barcode"],
                    "variant": variant,
                    "true_label": row[TARGET_COL],

                    "gender": None if pd.isna(row.get("gender")) else row.get("gender"),
                    "age_at_diagnosis": None if pd.isna(row.get("age_at_diagnosis")) else row.get("age_at_diagnosis"),
                    "tumor_descriptor_raw": None if pd.isna(row.get("tumor_descriptor")) else row.get("tumor_descriptor"),
                    "tissue_type_raw": None if pd.isna(row.get("tissue_type")) else row.get("tissue_type"),
                    "sample_type": row.get("sample_type"),

                    **pred
                }

                append_jsonl(RESULTS_JSONL, out)
                time.sleep(0.25)

            except Exception as e:
                append_jsonl(FAILED_JSONL, {
                    "task_setting": TASK_SETTING,
                    "eval_mode": EVAL_MODE,
                    "case_barcode": row.get("case_barcode", None),
                    "variant": variant,
                    "true_label": row.get(TARGET_COL, None),
                    "error": repr(e),
                    "traceback": traceback.format_exc()
                })
                print("FAILED:", row.get("case_barcode", None), variant, repr(e))
                time.sleep(2)

    compute_metrics(RESULTS_JSONL)
    print("SUCCESS")


if __name__ == "__main__":
    main()
