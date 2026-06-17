# ============================================================
# Task A BioMM-Eval
# DeepSeek-VL2 Sequential GPU Version
# TCGA-SKCM shortcut-controlled evaluation
# Existing fixed splits only
# Full / Clinical-only / RNA-only / Image-summary-only /
# Clinical+RNA / Counterfactual clinical / Counterfactual RNA
# Resume + retry + interim metrics + final metrics
# ============================================================

import os
import re
import json
import time
import random
import traceback
from datetime import datetime
from typing import List, Literal, Any, Dict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# -------------------------
# Config
# -------------------------
EVAL_MODE = "full"                     # "balanced" or "full"
TASK_SETTING = "shortcut_controlled"   # "shortcut_present" or "shortcut_controlled"

MODEL_ID = "deepseek-ai/deepseek-vl2"
# MODEL_ID = "deepseek-ai/deepseek-vl2-small"
# MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"

MODEL_TAG = MODEL_ID.split("/")[-1].replace("-", "_")

BASE_DIR = ""
OUT_DIR = os.path.join(BASE_DIR, "biomm_taskA_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_FULL_CSV = os.path.join(OUT_DIR, "taskA_test_full.csv")
TEST_BAL_CSV = os.path.join(OUT_DIR, "taskA_test_balanced.csv")

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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_JSONL = os.path.join(
    OUT_DIR,
    f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_results_{RUN_ID}.jsonl"
)
FAILED_JSONL = os.path.join(
    OUT_DIR,
    f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_failed_{RUN_ID}.jsonl"
)
METRICS_CSV = os.path.join(
    OUT_DIR,
    f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_metrics_{RUN_ID}.csv"
)
RELIANCE_CSV = os.path.join(
    OUT_DIR,
    f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_reliance_{RUN_ID}.csv"
)
REPORT_TXT = os.path.join(
    OUT_DIR,
    f"taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_report_{RUN_ID}.txt"
)

RESUME = True
INTERIM_EVERY = 50
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
TEMPERATURE = 0.0
MAX_RETRIES = 4

print("=" * 90)
print("TASK          : Task A BioMM-Eval")
print("MODEL_ID      :", MODEL_ID)
print("TASK_SETTING  :", TASK_SETTING)
print("EVAL_MODE     :", EVAL_MODE)
print("RUN_ID        :", RUN_ID)
print("RESULTS_JSONL :", RESULTS_JSONL)
print("=" * 90)

# -------------------------
# Schema
# -------------------------
class TaskAPrediction(BaseModel):
    prediction: Literal["Primary Tumor", "Metastatic"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image_summary", "clinical", "rna"]]
    image_reliance: float = Field(ge=0, le=1)
    clinical_reliance: float = Field(ge=0, le=1)
    rna_reliance: float = Field(ge=0, le=1)
    rationale_short: str

# -------------------------
# Helpers
# -------------------------
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl_safe(path: str) -> pd.DataFrame:
    rows = []

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                print(f"[WARN] skipped broken JSONL line {i}: {path}")

    return pd.DataFrame(rows)


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


def extract_json(text: str):
    if text is None:
        raise RuntimeError("Empty response text")

    text = text.strip()

    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1].strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise RuntimeError(f"Could not parse JSON from response: {text[:500]}")


def parse_prediction_response(response_text: str) -> Dict[str, Any]:
    obj = extract_json(response_text)

    obj["confidence"] = safe_float01(obj.get("confidence", 0.0))
    obj["image_reliance"] = safe_float01(obj.get("image_reliance", 0.0))
    obj["clinical_reliance"] = safe_float01(obj.get("clinical_reliance", 0.0))
    obj["rna_reliance"] = safe_float01(obj.get("rna_reliance", 0.0))

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["image_summary", "clinical", "rna"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskAPrediction.model_validate(obj).model_dump()

# -------------------------
# Dataset
# -------------------------
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

    print(f"\nUsing EXISTING {split_name} Task A test set")
    print("split file:", path)
    print(df[TARGET_COL].value_counts())
    print("n =", len(df))

    return df


def load_existing_done(results_path):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()
    done = set(zip(res["case_barcode"], res["variant"]))

    return res, done

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


def build_taskA_prompt(row, variant):
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
        instruction = "You are given image availability metadata, counterfactually modified clinical metadata, and original RNA expression summary."
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

This is for benchmark evaluation only.

Return only valid JSON. Do not use markdown.

Required JSON:
{{
  "prediction": "Primary Tumor or Metastatic",
  "confidence": number between 0 and 1,
  "used_modalities": ["image_summary", "clinical", "rna"],
  "image_reliance": number between 0 and 1,
  "clinical_reliance": number between 0 and 1,
  "rna_reliance": number between 0 and 1,
  "rationale_short": "brief reason"
}}

Rules:
- prediction must be exactly one of: Primary Tumor, Metastatic.
- confidence must be between 0 and 1.
- image_reliance, clinical_reliance, and rna_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance must be 0.
- used_modalities must include only actually provided modalities.
""".strip()

    return prompt


def build_taskA_conversation(row, variant):
    prompt = build_taskA_prompt(row, variant)

    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [],
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]

    return conversation

# -------------------------
# Load model
# -------------------------
print("\n[Load] Loading DeepSeek-VL2 processor...")
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
tokenizer = vl_chat_processor.tokenizer

print("[Load] Loading DeepSeek-VL2 model...")
vl_gpt = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
vl_gpt.eval()

target_device = next(vl_gpt.parameters()).device
print("[Load] Model target device:", target_device)

# -------------------------
# Inference
# -------------------------
def call_deepseek_taskA_once(row, variant):
    conversation = build_taskA_conversation(row, variant)

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[],
        force_batchify=True,
        system_prompt=(
            "You are a biomedical multimodal evaluation assistant. "
            "Return only valid JSON matching the requested fields."
        )
    )

    target_device = next(vl_gpt.parameters()).device
    prepare_inputs = prepare_inputs.to(target_device)

    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "use_cache": False,
        }

        if DO_SAMPLE:
            gen_kwargs["temperature"] = TEMPERATURE

        outputs = vl_gpt.language.generate(**gen_kwargs)

    output_text = tokenizer.decode(
        outputs[0].cpu().tolist(),
        skip_special_tokens=True
    ).strip()

    if "<|Assistant|>" in output_text:
        output_text = output_text.split("<|Assistant|>")[-1].strip()

    pred = parse_prediction_response(output_text)
    pred["raw_output"] = output_text

    # modality-specific correction
    if variant == "clinical_only":
        pred["image_reliance"] = 0.0
        pred["rna_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m == "clinical"
        ]

    elif variant == "rna_only":
        pred["image_reliance"] = 0.0
        pred["clinical_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m == "rna"
        ]

    elif variant == "image_summary_only":
        pred["clinical_reliance"] = 0.0
        pred["rna_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m == "image_summary"
        ]

    elif variant == "clinical_rna":
        pred["image_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m in ["clinical", "rna"]
        ]

    return pred


def call_deepseek_taskA(row, variant, max_retries=MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_deepseek_taskA_once(row, variant)

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** attempt * 3)

            print(
                f"\n[Retry] case={row.get('case_barcode')} "
                f"variant={variant} "
                f"attempt={attempt + 1}/{max_retries} "
                f"error={repr(e)} "
                f"wait={wait}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(wait)

    raise last_error

# -------------------------
# Metrics
# -------------------------
def compute_metrics(results_path, suffix=""):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        print("[Metrics] No valid rows.")
        return None

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()

    rows = []
    report_lines = []

    for variant, g in res.groupby("variant"):
        g = g.dropna(subset=["prediction", "true_label"]).copy()
        g = g[g["true_label"].isin(CLASSES)]
        g = g[g["prediction"].isin(CLASSES)]

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

    print(f"\n=== Task A Metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Task A Reliance summary {suffix or 'final/current'} ===")
    print(reliance_df.to_string(index=False))

    print("\nSaved:")
    print("metrics :", metrics_path)
    print("reliance:", reliance_path)
    print("report  :", report_path)

    return metrics


def summarize_progress(df, done):
    expected = len(df) * len(VARIANTS)
    completed = len(done)
    remaining = expected - completed

    print("\n=== Task A Progress ===")
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


def print_saved_status():
    print("\n==============================")
    print(" Task A Saved / Resume Status")
    print("==============================")
    print("RUN_ID:", RUN_ID)
    print("RESULTS_JSONL:", RESULTS_JSONL)
    print("FAILED_JSONL:", FAILED_JSONL)

    if os.path.exists(RESULTS_JSONL):
        print("\n[results file exists]")
        print("size bytes:", os.path.getsize(RESULTS_JSONL))

        if os.path.getsize(RESULTS_JSONL) > 0:
            res = load_jsonl_safe(RESULTS_JSONL)
            res_dedup = res.drop_duplicates(
                subset=["case_barcode", "variant"],
                keep="last"
            )

            print("raw rows:", len(res))
            print("dedup rows:", len(res_dedup))

            if "variant" in res_dedup.columns:
                print("\nCompleted by variant:")
                print(
                    res_dedup["variant"]
                    .value_counts()
                    .reindex(VARIANTS)
                    .fillna(0)
                    .astype(int)
                    .to_string()
                )

            print("\nPreview:")
            display(res_dedup.head())
        else:
            print("results file is empty.")
    else:
        print("\n[results file does not exist yet]")

    if os.path.exists(FAILED_JSONL) and os.path.getsize(FAILED_JSONL) > 0:
        fail = load_jsonl_safe(FAILED_JSONL)
        print("\nFailed rows:", len(fail))
        display(fail.tail())
    else:
        print("\nNo failed file or failed file is empty.")

# -------------------------
# Main
# -------------------------
def main_taskA_deepseekvl2_sequential():
    print_saved_status()

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

    print("\nTask A sample preview:")
    print(df[preview_cols].head().to_string(index=False))

    existing_res, done = load_existing_done(RESULTS_JSONL) if RESUME else (pd.DataFrame(), set())
    summarize_progress(df, done)

    print("\nComputing current/interim metrics before run...")
    compute_metrics(RESULTS_JSONL, suffix="_interim")

    todo = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["case_barcode"], variant)

            if not RESUME or key not in done:
                todo.append((row_dict, variant))

    if len(todo) == 0:
        print("\nNothing left to run. Computing final Task A metrics only.")
        compute_metrics(RESULTS_JSONL)
        print("\nTASK A SUCCESS")
        return

    print("\nRunning Task A remaining predictions sequentially on GPU...")
    print("Remaining tasks:", len(todo))

    completed_since_metric = 0
    start_time = time.time()

    for idx, (row_dict, variant) in enumerate(tqdm(todo, total=len(todo)), 1):
        key = (row_dict["case_barcode"], variant)

        try:
            pred = call_deepseek_taskA(row_dict, variant)

            out = {
                "task_setting": TASK_SETTING,
                "eval_mode": EVAL_MODE,
                "case_barcode": row_dict["case_barcode"],
                "variant": variant,
                "true_label": row_dict[TARGET_COL],

                "gender": None if pd.isna(row_dict.get("gender")) else row_dict.get("gender"),
                "age_at_diagnosis": None if pd.isna(row_dict.get("age_at_diagnosis")) else row_dict.get("age_at_diagnosis"),
                "tumor_descriptor_raw": None if pd.isna(row_dict.get("tumor_descriptor")) else row_dict.get("tumor_descriptor"),
                "tissue_type_raw": None if pd.isna(row_dict.get("tissue_type")) else row_dict.get("tissue_type"),
                "sample_type": row_dict.get("sample_type"),

                **pred
            }

            append_jsonl(RESULTS_JSONL, out)
            done.add(key)

        except Exception as e:
            fail = {
                "task_setting": TASK_SETTING,
                "eval_mode": EVAL_MODE,
                "case_barcode": row_dict.get("case_barcode", None),
                "variant": variant,
                "true_label": row_dict.get(TARGET_COL, None),
                "error": repr(e),
                "traceback": traceback.format_exc()
            }

            append_jsonl(FAILED_JSONL, fail)

            print(
                "\nFAILED:",
                fail.get("case_barcode"),
                fail.get("variant"),
                fail.get("error")
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\n[Task A] processed this run: {idx}/{len(todo)} | elapsed={elapsed/60:.1f} min")

        if completed_since_metric >= INTERIM_EVERY:
            completed_since_metric = 0
            print("\n[Task A Interim] computing metrics...")
            compute_metrics(RESULTS_JSONL, suffix="_interim")
            summarize_progress(df, done)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nComputing final Task A metrics...")
    compute_metrics(RESULTS_JSONL)

    print("\nTASK A SUCCESS")
    print("results :", RESULTS_JSONL)
    print("failed  :", FAILED_JSONL)
    print("metrics :", METRICS_CSV)
    print("reliance:", RELIANCE_CSV)
    print("report  :", REPORT_TXT)

# -------------------------
# Run
# -------------------------
main_taskA_deepseekvl2_sequential()
