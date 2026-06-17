# ============================================================
# Task A BioMM-Eval
# DeepSeek V4 Flash version
# TCGA-SKCM shortcut-controlled evaluation
# Uses existing fixed splits only:
# - taskA_test_full.csv
# - taskA_test_balanced.csv
# Resume + retry + interim metrics + final metrics
# ============================================================


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
# DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"

MODEL = "deepseek-v4-flash"
MODEL_TAG = "deepseekv4flash"

BASE_DIR = ""
OUT_DIR = os.path.join(BASE_DIR, "biomm_taskA_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_FULL_CSV = os.path.join(OUT_DIR, "taskA_test_full.csv")
TEST_BAL_CSV = os.path.join(OUT_DIR, "taskA_test_balanced.csv")

TASK_SETTING = "shortcut_controlled"   # "shortcut_present" or "shortcut_controlled"
EVAL_MODE = "full"                     # "balanced" or "full"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_JSONL = os.path.join(
    OUT_DIR, f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_results_{RUN_ID}.jsonl"
)
FAILED_JSONL = os.path.join(
    OUT_DIR, f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_failed_{RUN_ID}.jsonl"
)
METRICS_CSV = os.path.join(
    OUT_DIR, f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_metrics_{RUN_ID}.csv"
)
RELIANCE_CSV = os.path.join(
    OUT_DIR, f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_reliance_{RUN_ID}.csv"
)
REPORT_TXT = os.path.join(
    OUT_DIR, f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_report_{RUN_ID}.txt"
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

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

print("MODEL:", MODEL)
print("RUN_ID:", RUN_ID)
print("RESULTS:", RESULTS_JSONL)

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


def safe_float01(x, default=0.0):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


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
        blocks = [image_summary_block(row), clinical_block(row), rna_block(row)]

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
        blocks = [clinical_block(row), rna_block(row)]

    elif variant == "counterfactual_clinical":
        instruction = "You are given image availability metadata, modified clinical metadata, and original RNA expression summary."
        blocks = [image_summary_block(row), clinical_block(row, counterfactual=True), rna_block(row)]

    elif variant == "counterfactual_rna":
        instruction = "You are given image availability metadata, original clinical metadata, and perturbed RNA expression summary."
        blocks = [image_summary_block(row), clinical_block(row), rna_block(row, counterfactual=True)]

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


def parse_prediction_response(response_text):
    text = response_text.strip()

    try:
        obj = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start:end + 1])
        else:
            raise RuntimeError(f"Could not parse JSON from response: {text[:500]}")

    pred = obj.get("prediction")
    if pred not in CLASSES:
        raise RuntimeError(f"Invalid prediction: {pred}")

    used = obj.get("used_modalities", [])
    if not isinstance(used, list):
        used = []

    used = [
        m for m in used
        if m in ["image_summary", "clinical", "rna"]
    ]

    return {
        "prediction": pred,
        "confidence": safe_float01(obj.get("confidence", 0)),
        "used_modalities": used,
        "image_reliance": safe_float01(obj.get("image_reliance", 0)),
        "clinical_reliance": safe_float01(obj.get("clinical_reliance", 0)),
        "rna_reliance": safe_float01(obj.get("rna_reliance", 0)),
        "rationale_short": str(obj.get("rationale_short", ""))[:500],
    }


def call_model_once(row, variant):
    prompt = build_prompt(row, variant)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical multimodal evaluation assistant. "
                    "Return only valid JSON matching the requested fields."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=700,

        extra_body={
            "thinking": {"type": "disabled"}
        }
    )

    obj = parse_prediction_response(response.choices[0].message.content)

    # modality-specific safety correction
    if variant == "clinical_only":
        obj["image_reliance"] = 0.0
        obj["rna_reliance"] = 0.0
        obj["used_modalities"] = [m for m in obj["used_modalities"] if m == "clinical"]

    elif variant == "rna_only":
        obj["image_reliance"] = 0.0
        obj["clinical_reliance"] = 0.0
        obj["used_modalities"] = [m for m in obj["used_modalities"] if m == "rna"]

    elif variant == "image_summary_only":
        obj["clinical_reliance"] = 0.0
        obj["rna_reliance"] = 0.0
        obj["used_modalities"] = [m for m in obj["used_modalities"] if m == "image_summary"]

    elif variant == "clinical_rna":
        obj["image_reliance"] = 0.0
        obj["used_modalities"] = [m for m in obj["used_modalities"] if m in ["clinical", "rna"]]

    return obj


def call_model(row, variant, max_retries=4):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_model_once(row, variant)

        except Exception as e:
            last_error = e
            wait = min(90, 2 ** attempt * 5)
            print(
                f"\n[Retry] case={row.get('case_barcode')} "
                f"variant={variant} attempt={attempt+1}/{max_retries} "
                f"error={repr(e)} wait={wait}s"
            )
            time.sleep(wait)

    raise last_error


def load_existing_done(results_path):
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        return pd.DataFrame(), set()

    res = pd.read_json(results_path, lines=True)

    if len(res) == 0:
        return res, set()

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()
    done = set(zip(res["case_barcode"], res["variant"]))

    return res, done


# -------------------------
# Metrics
# -------------------------
def compute_metrics(results_path, suffix=""):
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        print("[Metrics] No results yet.")
        return None

    res = pd.read_json(results_path, lines=True)
    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()

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

    metrics_path = METRICS_CSV if suffix == "" else METRICS_CSV.replace(".csv", f"{suffix}.csv")
    reliance_path = RELIANCE_CSV if suffix == "" else RELIANCE_CSV.replace(".csv", f"{suffix}.csv")
    report_path = REPORT_TXT if suffix == "" else REPORT_TXT.replace(".txt", f"{suffix}.txt")

    metrics.to_csv(metrics_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = metrics.set_index("variant")["balanced_accuracy"].to_dict() if len(metrics) else {}

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
    reliance_df.to_csv(reliance_path, index=False)

    print(f"\n=== Task A metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Task A reliance summary {suffix or 'final/current'} ===")
    print(reliance_df.to_string(index=False))

    print("\nSaved files:")
    print("results:", RESULTS_JSONL)
    print("metrics:", metrics_path)
    print("reliance:", reliance_path)
    print("report:", report_path)

    return metrics


def summarize_progress(df, done):
    expected = len(df) * len(VARIANTS)
    completed = len(done)
    remaining = expected - completed

    print("\n=== Progress ===")
    print("samples:", len(df))
    print("variants:", len(VARIANTS))
    print("expected total rows:", expected)
    print("completed rows:", completed)
    print("remaining rows:", remaining)
    print("completion rate:", round(completed / expected * 100, 2), "%")

    if done:
        done_df = pd.DataFrame(list(done), columns=["case_barcode", "variant"])
        print("\nCompleted by variant:")
        print(
            done_df["variant"]
            .value_counts()
            .reindex(VARIANTS)
            .fillna(0)
            .astype(int)
            .to_string()
        )


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

    existing_res, done = load_existing_done(RESULTS_JSONL)
    summarize_progress(df, done)

    print("\nComputing current/interim metrics before run...")
    compute_metrics(RESULTS_JSONL, suffix="_interim")

    expected_total = len(df) * len(VARIANTS)
    total_remaining = expected_total - len(done)

    if total_remaining <= 0:
        print("\nNothing left to run. Computing final metrics only.")
        compute_metrics(RESULTS_JSONL)
        print("SUCCESS")
        return

    print("\nRunning remaining predictions...")

    completed_since_metric = 0
    pbar = tqdm(total=total_remaining)

    for _, row in df.iterrows():
        for variant in VARIANTS:
            key = (row["case_barcode"], variant)

            if key in done:
                continue

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
                done.add(key)

                pbar.update(1)
                completed_since_metric += 1

                time.sleep(0.25)

                if completed_since_metric >= 50:
                    completed_since_metric = 0
                    print("\n[Interim] computing metrics...")
                    compute_metrics(RESULTS_JSONL, suffix="_interim")
                    summarize_progress(df, done)

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
                print("\nFAILED:", row.get("case_barcode", None), variant, repr(e))
                pbar.update(1)
                time.sleep(3)

    pbar.close()

    print("\nComputing final metrics...")
    compute_metrics(RESULTS_JSONL)

    print("\nSUCCESS")
    print("results:", RESULTS_JSONL)
    print("failed:", FAILED_JSONL)
    print("metrics:", METRICS_CSV)
    print("reliance:", RELIANCE_CSV)
    print("report:", REPORT_TXT)


if __name__ == "__main__":
    main()
