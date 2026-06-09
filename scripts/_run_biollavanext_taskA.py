# ============================================================
# Task A BioMM-Eval
# AdaptLLM biomed-LLaVA-NeXT-Llama3-8B
# Sequential GPU Version
# Existing fixed splits only
# Full / Clinical-only / RNA-only / Image-summary-only /
# Clinical+RNA / Counterfactual clinical / Counterfactual RNA
# Resume + retry + interim metrics + final metrics
# ============================================================

import os, re, json, time, random, traceback
from datetime import datetime
from typing import List, Literal, Any, Dict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# Config
# ============================================================

EVAL_MODE = "full"                     # "balanced" or "full"
TASK_SETTING = "shortcut_controlled"   # "shortcut_present" or "shortcut_controlled"

MODEL_ID = "AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B"
MODEL_TAG = "biomed_llava_next_llama3_8b"

BASE_DIR = ""

OUT_DIR = os.path.join(BASE_DIR, "biomm_taskA_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_FULL_CSV = os.path.join(OUT_DIR, "taskA_test_full.csv")
TEST_BAL_CSV = os.path.join(OUT_DIR, "taskA_test_balanced.csv")

TARGET_COL = "sample_type"
CLASSES = ["Primary Tumor", "Metastatic"]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

VARIANTS = [
    "full",
    "clinical_only",
    "rna_only",
    "image_summary_only",
    "clinical_rna",
    "counterfactual_clinical",
    "counterfactual_rna",
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

CONFUSION_DIR = os.path.join(
    OUT_DIR,
    f"confusion_matrices_taskA_{MODEL_TAG}_{TASK_SETTING}_{EVAL_MODE}_{RUN_ID}"
)
os.makedirs(CONFUSION_DIR, exist_ok=True)

RESUME = True
INTERIM_EVERY = 50
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.0
MAX_RETRIES = 4

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("=" * 90)
print("TASK          : Task A BioMM-Eval")
print("MODEL_ID      :", MODEL_ID)
print("TASK_SETTING  :", TASK_SETTING)
print("EVAL_MODE     :", EVAL_MODE)
print("RUN_ID        :", RUN_ID)
print("OUT_DIR       :", OUT_DIR)
print("RESULTS_JSONL :", RESULTS_JSONL)
print("=" * 90)

# ============================================================
# Schema
# ============================================================

class TaskAPrediction(BaseModel):
    prediction: Literal["Primary Tumor", "Metastatic"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image_summary", "clinical", "rna"]]
    image_reliance: float = Field(ge=0, le=1)
    clinical_reliance: float = Field(ge=0, le=1)
    rna_reliance: float = Field(ge=0, le=1)
    rationale_short: str

# ============================================================
# Helpers
# ============================================================

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
    if x.lower() in ["", "nan", "none", "null", "missing", "unknown", "not reported", "not available"]:
        return "unknown"
    return x


def safe_float01(x, default=0.0):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def clean_response(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    for tok in [
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "</s>",
    ]:
        text = text.replace(tok, "")

    return text.strip()


def extract_json(text: str):
    if text is None:
        raise RuntimeError("Empty response text")

    text = clean_response(text)

    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"Could not find JSON object in response: {text[:500]}")

    candidate = text[start:].strip()
    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(candidate)
        return obj
    except Exception:
        pass

    end = text.rfind("}")
    if end > start:
        return json.loads(text[start:end + 1])

    raise RuntimeError(f"Could not parse JSON from response: {text[:500]}")


def fallback_extract_prediction(response_text: str) -> Dict[str, Any]:
    text = clean_response(response_text)
    upper = text.upper()

    if "PRIMARY TUMOR" in upper or "PRIMARY" in upper:
        pred = "Primary Tumor"
    elif "METASTATIC" in upper or "METASTASIS" in upper or "METASTASES" in upper:
        pred = "Metastatic"
    else:
        raise RuntimeError(f"Could not find Task A prediction in response: {text[:500]}")

    return {
        "prediction": pred,
        "confidence": 0.5,
        "used_modalities": ["image_summary", "clinical", "rna"],
        "image_reliance": 0.33,
        "clinical_reliance": 0.33,
        "rna_reliance": 0.34,
        "rationale_short": text[:500],
    }


def parse_prediction_response(response_text: str) -> Dict[str, Any]:
    text = clean_response(response_text)

    try:
        obj = extract_json(text)

        bad_pred_values = {
            "Primary Tumor or Metastatic",
            "Primary Tumor/Metastatic",
            "Primary or Metastatic",
            "primary tumor or metastatic",
        }

        if str(obj.get("prediction", "")).strip() in bad_pred_values:
            return TaskAPrediction.model_validate(
                fallback_extract_prediction(text)
            ).model_dump()

    except Exception:
        return TaskAPrediction.model_validate(
            fallback_extract_prediction(text)
        ).model_dump()

    obj["confidence"] = safe_float01(obj.get("confidence", 0.0), default=0.5)
    obj["image_reliance"] = safe_float01(obj.get("image_reliance", 0.0), default=0.33)
    obj["clinical_reliance"] = safe_float01(obj.get("clinical_reliance", 0.0), default=0.33)
    obj["rna_reliance"] = safe_float01(obj.get("rna_reliance", 0.0), default=0.34)

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["image_summary", "clinical", "rna"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskAPrediction.model_validate(obj).model_dump()

# ============================================================
# Dataset
# ============================================================

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
        "expr_mean",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in existing split: {missing}")

    df = df[df[TARGET_COL].isin(CLASSES)].copy().reset_index(drop=True)

    print(f"\nUsing EXISTING {split_name} Task A test set")
    print("split file:", path)
    print(df[TARGET_COL].value_counts().reindex(CLASSES))
    print("n =", len(df))

    return df


def load_existing_done(results_path):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()
    done = set(zip(res["case_barcode"], res["variant"]))

    return res, done

# ============================================================
# Prompt blocks
# ============================================================

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
        "expr_nonzero_fraction",
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


def build_taskA_instruction(row, variant):
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

    task = f"""
You are a strict JSON generator. Output JSON only.

Task:
Predict the TCGA-SKCM sample type.

Allowed labels:
- Primary Tumor
- Metastatic

{instruction}

{joined}

This is for benchmark evaluation only, not clinical use.

Return only valid JSON. Do not use markdown. Do not explain.

Required JSON:
{{
  "prediction": "Primary Tumor or Metastatic",
  "confidence": 0.0,
  "used_modalities": ["image_summary", "clinical", "rna"],
  "image_reliance": 0.0,
  "clinical_reliance": 0.0,
  "rna_reliance": 0.0,
  "rationale_short": "brief reason"
}}

Rules:
- prediction must be exactly one of: Primary Tumor, Metastatic.
- confidence must be between 0 and 1.
- image_reliance, clinical_reliance, and rna_reliance must each be between 0 and 1.
- If image_summary is not provided, image_reliance must be 0.
- If clinical metadata is not provided, clinical_reliance must be 0.
- If RNA summary is not provided, rna_reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()

    return task


def format_llava_next_prompt(instruction: str):
    system_msg = (
        "You are a helpful biomedical multimodal evaluation assistant. "
        "Follow the user's output format exactly."
    )

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return prompt

# ============================================================
# Load model
# ============================================================

print("\n[Load] Loading processor...")
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

print("[Load] Loading model...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    attn_implementation="sdpa",
)

# Text-only Task A라 image-token patch는 필수는 아니지만,
# Task C 코드와 호환되게 유지
processor.patch_size = model.config.vision_config.patch_size
processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
processor.num_additional_image_tokens = 1

print("[Patch] processor.patch_size:", processor.patch_size)
print("[Patch] processor.vision_feature_select_strategy:", processor.vision_feature_select_strategy)
print("[Patch] processor.num_additional_image_tokens:", processor.num_additional_image_tokens)
print("[Patch] model image_token_index:", getattr(model.config, "image_token_index", None))

model.eval()
target_device = next(model.parameters()).device

print("[Load] target_device:", target_device)
if torch.cuda.is_available():
    print("[Load] GPU name:", torch.cuda.get_device_name(0))
    print("[Load] bf16 supported:", torch.cuda.is_bf16_supported())

# ============================================================
# Inference
# ============================================================

def call_model_taskA_once(row, variant):
    instruction = build_taskA_instruction(row, variant)
    prompt = format_llava_next_prompt(instruction)

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(target_device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    answer_start = int(inputs["input_ids"].shape[-1])

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
    }

    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **gen_kwargs,
        )

    output_text = processor.decode(
        output[0][answer_start:],
        skip_special_tokens=True,
    ).strip()

    output_text = clean_response(output_text)

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


def call_model_taskA(row, variant, max_retries=MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_model_taskA_once(row, variant)

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

# ============================================================
# Metrics
# ============================================================

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

        rows.append(
            {
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
                "mean_rna_reliance_self_report": float(g["rna_reliance"].mean()),
            }
        )

        report_lines.append(f"\n\n===== {variant} =====\n")
        report_lines.append(
            classification_report(
                y_true,
                y_pred,
                labels=CLASSES,
                zero_division=0,
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        cm_csv = os.path.join(CONFUSION_DIR, f"confusion_{variant}{suffix}.csv")
        cm_df.to_csv(cm_csv)

    metrics = pd.DataFrame(rows)

    metrics_path = METRICS_CSV if suffix == "" else METRICS_CSV.replace(".csv", f"{suffix}.csv")
    reliance_path = RELIANCE_CSV if suffix == "" else RELIANCE_CSV.replace(".csv", f"{suffix}.csv")
    report_path = REPORT_TXT if suffix == "" else REPORT_TXT.replace(".txt", f"{suffix}.txt")

    metrics.to_csv(metrics_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = (
        metrics.set_index("variant")["balanced_accuracy"].to_dict()
        if len(metrics)
        else {}
    )

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

        "integration_gain_over_best_unimodal": None
        if full is None or not unimodal_scores
        else full - max(unimodal_scores),

        "clinical_shortcut_advantage_over_full": None
        if full is None or clinical is None
        else clinical - full,

        "clinical_necessity_full_minus_rna_only": None
        if full is None or rna is None
        else full - rna,

        "rna_necessity_full_minus_clinical_only": None
        if full is None or clinical is None
        else full - clinical,

        "image_summary_necessity_full_minus_clinical_rna": None
        if full is None or clinical_rna is None
        else full - clinical_rna,

        "clinical_counterfactual_sensitivity_full_minus_cf": None
        if full is None or cf_clinical is None
        else full - cf_clinical,

        "rna_counterfactual_sensitivity_full_minus_cf": None
        if full is None or cf_rna is None
        else full - cf_rna,
    }

    wide = res.pivot_table(
        index="case_barcode",
        columns="variant",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    for other in [
        "clinical_only",
        "rna_only",
        "image_summary_only",
        "clinical_rna",
        "counterfactual_clinical",
        "counterfactual_rna",
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
    print("confusion dir:", CONFUSION_DIR)

    return metrics

# ============================================================
# Status
# ============================================================

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
                keep="last",
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
            print(res_dedup.head().to_string(index=False))
        else:
            print("results file is empty.")
    else:
        print("\n[results file does not exist yet]")

    if os.path.exists(FAILED_JSONL) and os.path.getsize(FAILED_JSONL) > 0:
        fail = load_jsonl_safe(FAILED_JSONL)
        print("\nFailed rows:", len(fail))
        print(fail.tail().to_string(index=False))
    else:
        print("\nNo failed file or failed file is empty.")

# ============================================================
# Main
# ============================================================

def main_taskA_biomed_llava_next():
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
        "has_rna",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nTask A sample preview:")
    print(df[preview_cols].head().to_string(index=False))

    existing_res, done = (
        load_existing_done(RESULTS_JSONL) if RESUME else (pd.DataFrame(), set())
    )
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
            pred = call_model_taskA(row_dict, variant)

            out = {
                "model_id": MODEL_ID,
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

                **pred,
            }

            append_jsonl(RESULTS_JSONL, out)
            done.add(key)

        except Exception as e:
            fail = {
                "model_id": MODEL_ID,
                "task_setting": TASK_SETTING,
                "eval_mode": EVAL_MODE,
                "case_barcode": row_dict.get("case_barcode", None),
                "variant": variant,
                "true_label": row_dict.get(TARGET_COL, None),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

            append_jsonl(FAILED_JSONL, fail)

            print(
                "\nFAILED:",
                fail.get("case_barcode"),
                fail.get("variant"),
                fail.get("error"),
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"\n[Task A] processed this run: {idx}/{len(todo)} "
                f"| elapsed={elapsed / 60:.1f} min"
            )

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

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main_taskA_biomed_llava_next()
