# ============================================================
# Task C BioMM-Eval v3
# Gemma-4-26B-A4B-it Sequential GPU Version
# Full / Image-only / Metadata-only / Counterfactual
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
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("torch:", torch.__version__)

# ============================================================
# Config
# ============================================================

EVAL_MODE = "balanced"  # "balanced" or "full"

MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_TAG = "gemma4_26b_a4b_it"

BASE_DIR = ""
MASTER_CSV = os.path.join(BASE_DIR, "taskC_master.csv")

OUT_DIR = os.path.join(BASE_DIR, "biomm_taskC_multiclass_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["MEL", "NV", "BCC", "BKL"]
SEED = 42
random.seed(SEED)

VARIANTS = [
    "full",
    "image_only",
    "metadata_only",
    "metadata_counterfactual"
]

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


TEST_FULL_CSV = os.path.join(OUT_DIR, "taskC_test_full.csv")
TEST_BAL_CSV = os.path.join(OUT_DIR, "taskC_test_balanced.csv")

RESULTS_JSONL = os.path.join(
    OUT_DIR,
    f"taskC_{MODEL_TAG}_{EVAL_MODE}_results_{RUN_ID}.jsonl"
)
FAILED_JSONL = os.path.join(
    OUT_DIR,
    f"taskC_{MODEL_TAG}_{EVAL_MODE}_failed_{RUN_ID}.jsonl"
)
METRICS_CSV = os.path.join(
    OUT_DIR,
    f"taskC_{MODEL_TAG}_{EVAL_MODE}_metrics_{RUN_ID}.csv"
)
RELIANCE_CSV = os.path.join(
    OUT_DIR,
    f"taskC_{MODEL_TAG}_{EVAL_MODE}_reliance_{RUN_ID}.csv"
)
REPORT_TXT = os.path.join(
    OUT_DIR,
    f"taskC_{MODEL_TAG}_{EVAL_MODE}_report_{RUN_ID}.txt"
)
CONFUSION_DIR = os.path.join(
    OUT_DIR,
    f"confusion_matrices_{MODEL_TAG}_{EVAL_MODE}_{RUN_ID}"
)
os.makedirs(CONFUSION_DIR, exist_ok=True)

RESUME = True
INTERIM_EVERY = 50
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
MAX_RETRIES = 4
IMAGE_MAX_SIZE = 1024

print("=" * 90)
print("TASK              : Task C BioMM-Eval v3")
print("MODEL_ID          :", MODEL_ID)
print("EVAL_MODE         :", EVAL_MODE)
print("RUN_ID            :", RUN_ID)
print("MASTER_CSV        :", MASTER_CSV)
print("OUT_DIR           :", OUT_DIR)
print("RESULTS_JSONL     :", RESULTS_JSONL)
print("=" * 90)

# ============================================================
# Schema
# ============================================================

class TaskCPrediction(BaseModel):
    prediction: Literal["MEL", "NV", "BCC", "BKL"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image", "metadata"]]
    image_reliance: float = Field(ge=0, le=1)
    metadata_reliance: float = Field(ge=0, le=1)
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
    if x.lower() in ["", "nan", "none", "null", "missing", "unknown"]:
        return "unknown"
    return x


def safe_float01(x, default=0.0):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def clean_gemma_response(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    if "processor" in globals() and hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, str):
                text = parsed
            elif isinstance(parsed, dict):
                text = parsed.get("text", text) or parsed.get("response", text)
        except Exception:
            pass

    text = re.sub(r"<\|channel\|>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = text.replace("<|channel|>thought", "")
    text = text.replace("<channel|>", "")
    text = text.strip()

    return text


def extract_json(text: str):
    if text is None:
        raise RuntimeError("Empty response text")

    text = clean_gemma_response(text)

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


def parse_prediction_response(response_text: str) -> Dict[str, Any]:
    obj = extract_json(response_text)

    obj["confidence"] = safe_float01(obj.get("confidence", 0.0))
    obj["image_reliance"] = safe_float01(obj.get("image_reliance", 0.0))
    obj["metadata_reliance"] = safe_float01(obj.get("metadata_reliance", 0.0))

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["image", "metadata"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskCPrediction.model_validate(obj).model_dump()


def load_image_for_gemma(image_path: str):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE))
    return image

# ============================================================
# Prompt helpers
# ============================================================

def make_counterfactual_metadata(row):
    possible_sites = [
        "anterior torso",
        "posterior torso",
        "lower extremity",
        "upper extremity",
        "head/neck",
        "palms/soles",
        "oral/genital"
    ]

    possible_sex = ["male", "female"]

    age = row.get("age_approx")
    sex = clean_val(row.get("sex"))
    site = clean_val(row.get("anatom_site_general"))

    new_sites = [s for s in possible_sites if s != site]
    new_site = random.choice(new_sites) if new_sites else random.choice(possible_sites)

    if sex == "male":
        new_sex = "female"
    elif sex == "female":
        new_sex = "male"
    else:
        new_sex = random.choice(possible_sex)

    try:
        age_num = float(age)
        new_age = max(5, min(95, 95 - age_num))
    except Exception:
        new_age = "unknown"

    return {
        "age_approx": new_age,
        "sex": new_sex,
        "anatom_site_general": new_site
    }


def metadata_block(row, counterfactual=False):
    if counterfactual:
        meta = make_counterfactual_metadata(row)
        prefix = "Counterfactually modified clinical metadata"
    else:
        meta = {
            "age_approx": clean_val(row.get("age_approx")),
            "sex": clean_val(row.get("sex")),
            "anatom_site_general": clean_val(row.get("anatom_site_general"))
        }
        prefix = "Clinical metadata"

    text = f"""
{prefix}:
- approximate age: {meta["age_approx"]}
- sex: {meta["sex"]}
- anatomical site: {meta["anatom_site_general"]}
""".strip()

    return text, meta


def build_taskC_prompt(row, variant):
    if variant == "full":
        instruction = "You are given a dermoscopic skin lesion image and clinical metadata."
        meta_text, meta_used = metadata_block(row, counterfactual=False)
        include_image = True

    elif variant == "image_only":
        instruction = "You are given only a dermoscopic skin lesion image. No clinical metadata is provided."
        meta_text = "Clinical metadata: not provided."
        meta_used = None
        include_image = True

    elif variant == "metadata_only":
        instruction = "You are given only clinical metadata. No image is provided."
        meta_text, meta_used = metadata_block(row, counterfactual=False)
        include_image = False

    elif variant == "metadata_counterfactual":
        instruction = "You are given the same dermoscopic image but counterfactually modified clinical metadata."
        meta_text, meta_used = metadata_block(row, counterfactual=True)
        include_image = True

    else:
        raise ValueError(f"Unknown variant: {variant}")

    prompt = f"""
Task:
Classify the skin lesion into exactly one of the following ISIC diagnostic classes:

- MEL: melanoma
- NV: melanocytic nevus
- BCC: basal cell carcinoma
- BKL: benign keratosis-like lesion

{instruction}

{meta_text}

This is for benchmark evaluation only.

Return only valid JSON. Do not use markdown. Do not explain.

Required JSON:
{{
  "prediction": "MEL or NV or BCC or BKL",
  "confidence": 0.0,
  "used_modalities": ["image", "metadata"],
  "image_reliance": 0.0,
  "metadata_reliance": 0.0,
  "rationale_short": "brief reason"
}}

Rules:
- prediction must be exactly one of: MEL, NV, BCC, BKL.
- confidence must be between 0 and 1.
- image_reliance and metadata_reliance must be between 0 and 1.
- If image is not provided, image_reliance must be 0.
- If metadata is not provided, metadata_reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()

    return prompt, include_image, meta_used

# ============================================================
# Dataset helpers
# ============================================================

def save_full_dataset(df):
    df_full = df.copy().sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_full["split_id"] = "taskC_full_v1"
    df_full.to_csv(TEST_FULL_CSV, index=False)
    print(f"[Saved] full dataset: {TEST_FULL_CSV} n={len(df_full)}")
    return df_full


def save_balanced_dataset(df):
    class_counts = df["diagnosis"].value_counts()
    n_balanced = class_counts.loc[CLASSES].min()

    parts = []

    for cls in CLASSES:
        sub = df[df["diagnosis"] == cls].copy()
        parts.append(sub.sample(n=n_balanced, random_state=SEED))

    df_bal = (
        pd.concat(parts, axis=0)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    df_bal["split_id"] = "taskC_balanced_v1"
    df_bal.to_csv(TEST_BAL_CSV, index=False)

    print(f"[Saved] balanced dataset: {TEST_BAL_CSV} n={len(df_bal)}")
    return df_bal


def load_or_create_taskC_test_set(mode=EVAL_MODE):
    df = pd.read_csv(MASTER_CSV)

    required = [
        "image_id",
        "image_path",
        "has_image_file",
        "diagnosis",
        "age_approx",
        "sex",
        "anatom_site_general"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in MASTER_CSV: {missing}")

    df = df[
        (df["has_image_file"] == 1) &
        (df["image_path"].notna()) &
        (df["diagnosis"].isin(CLASSES))
    ].copy()

    df = df[
        df["image_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    ].copy()

    if not os.path.exists(TEST_FULL_CSV):
        save_full_dataset(df)

    if not os.path.exists(TEST_BAL_CSV):
        save_balanced_dataset(df)

    if mode == "full":
        out = pd.read_csv(TEST_FULL_CSV)
        print("\nUsing FULL Task C test set")
    elif mode == "balanced":
        out = pd.read_csv(TEST_BAL_CSV)
        print("\nUsing BALANCED Task C test set")
    else:
        raise ValueError("EVAL_MODE must be 'full' or 'balanced'")

    out = out[
        out["image_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    ].reset_index(drop=True)

    print("\nTask C class counts:")
    print(out["diagnosis"].value_counts().reindex(CLASSES))
    print("n =", len(out))

    return out


def load_existing_done(results_path):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    res = res.drop_duplicates(subset=["image_id", "variant"], keep="last").copy()
    done = set(zip(res["image_id"], res["variant"]))

    return res, done

# ============================================================
# Load Gemma 4
# ============================================================

print("\n[Load] Loading Gemma 4 processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("[Load] Loading Gemma 4 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()

target_device = next(model.parameters()).device
print("[Load] CUDA available:", torch.cuda.is_available())
print("[Load] CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[Load] GPU name:", torch.cuda.get_device_name(0))
print("[Load] Model target device:", target_device)

# ============================================================
# Inference
# ============================================================

def call_gemma_taskC_once(row, variant):
    prompt, include_image, meta_used = build_taskC_prompt(row, variant)

    if include_image:
        image = load_image_for_gemma(row.get("image_path"))
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output JSON only."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        image = None
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if include_image:
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(target_device)
    else:
        inputs = processor(
            text=text,
            return_tensors="pt"
        ).to(target_device)

    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )

    output_text = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=False
    ).strip()

    output_text = clean_gemma_response(output_text)

    pred = parse_prediction_response(output_text)
    pred["metadata_used"] = meta_used
    pred["raw_output"] = output_text

    if variant == "metadata_only":
        pred["image_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m != "image"
        ]

    if variant == "image_only":
        pred["metadata_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m != "metadata"
        ]

    return pred


def call_gemma_taskC(row, variant, max_retries=MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_gemma_taskC_once(row, variant)

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** attempt * 3)

            print(
                f"\n[Retry] image_id={row.get('image_id')} "
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

    res = res.drop_duplicates(subset=["image_id", "variant"], keep="last").copy()

    rows = []
    report_lines = []

    for variant, g in res.groupby("variant"):
        g = g.dropna(subset=["prediction", "true_diagnosis"]).copy()
        g = g[g["true_diagnosis"].isin(CLASSES)]
        g = g[g["prediction"].isin(CLASSES)]

        if len(g) == 0:
            continue

        y_true = g["true_diagnosis"]
        y_pred = g["prediction"]

        row = {
            "variant": variant,
            "n": len(g),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mean_confidence": float(g["confidence"].mean()),
            "mean_image_reliance_self_report": float(g["image_reliance"].mean()),
            "mean_metadata_reliance_self_report": float(g["metadata_reliance"].mean())
        }

        rows.append(row)

        report_lines.append(f"\n\n===== {variant} =====\n")
        report_lines.append(
            classification_report(
                y_true,
                y_pred,
                labels=CLASSES,
                zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)

        cm_csv = os.path.join(CONFUSION_DIR, f"confusion_{variant}{suffix}.csv")
        cm_png = os.path.join(CONFUSION_DIR, f"confusion_{variant}{suffix}.png")

        cm_df.to_csv(cm_csv)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"Task C Confusion Matrix: {variant}{suffix}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(CLASSES)), CLASSES)
        plt.yticks(range(len(CLASSES)), CLASSES)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(cm_png, dpi=200)
        plt.close()

    metrics = pd.DataFrame(rows)

    metrics_path = METRICS_CSV if suffix == "" else METRICS_CSV.replace(".csv", f"{suffix}.csv")
    report_path = REPORT_TXT if suffix == "" else REPORT_TXT.replace(".txt", f"{suffix}.txt")
    reliance_path = RELIANCE_CSV if suffix == "" else RELIANCE_CSV.replace(".csv", f"{suffix}.csv")

    metrics.to_csv(metrics_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = metrics.set_index("variant")["balanced_accuracy"].to_dict() if len(metrics) else {}

    full = score.get("full")
    image_only = score.get("image_only")
    metadata_only = score.get("metadata_only")
    cf = score.get("metadata_counterfactual")

    reliance = {
        "full_balanced_accuracy": full,
        "image_only_balanced_accuracy": image_only,
        "metadata_only_balanced_accuracy": metadata_only,
        "counterfactual_balanced_accuracy": cf,
        "metadata_necessity_score_full_minus_image_only": None if full is None or image_only is None else full - image_only,
        "image_necessity_score_full_minus_metadata_only": None if full is None or metadata_only is None else full - metadata_only,
        "counterfactual_sensitivity_full_minus_cf": None if full is None or cf is None else full - cf,
    }

    wide = res.pivot_table(
        index="image_id",
        columns="variant",
        values="prediction",
        aggfunc="first"
    ).reset_index()

    def change_rate(a, b):
        if a in wide.columns and b in wide.columns:
            valid = wide.dropna(subset=[a, b]).copy()
            if len(valid):
                return float((valid[a] != valid[b]).mean())
        return None

    reliance["counterfactual_prediction_change_rate"] = change_rate(
        "full",
        "metadata_counterfactual"
    )
    reliance["full_vs_image_only_prediction_change_rate"] = change_rate(
        "full",
        "image_only"
    )
    reliance["full_vs_metadata_only_prediction_change_rate"] = change_rate(
        "full",
        "metadata_only"
    )

    reliance_df = pd.DataFrame([reliance])
    reliance_df.to_csv(reliance_path, index=False)

    print(f"\n=== Task C Metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Task C Reliance summary {suffix or 'final/current'} ===")
    print(reliance_df.to_string(index=False))

    print("\nSaved:")
    print("metrics:", metrics_path)
    print("reliance:", reliance_path)
    print("report:", report_path)
    print("confusion dir:", CONFUSION_DIR)

    return metrics


def summarize_progress(df, done):
    expected = len(df) * len(VARIANTS)
    completed = len(done)
    remaining = expected - completed

    print("\n=== Task C Progress ===")
    print("samples:", len(df))
    print("variants:", len(VARIANTS))
    print("expected total rows:", expected)
    print("completed rows:", completed)
    print("remaining rows:", remaining)
    print("completion rate:", round(completed / expected * 100, 2), "%")

    if done:
        done_df = pd.DataFrame(list(done), columns=["image_id", "variant"])

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
    print(" Task C Saved / Resume Status")
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
                subset=["image_id", "variant"],
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

def main_taskC_gemma4_sequential():
    print_saved_status()

    df = load_or_create_taskC_test_set(EVAL_MODE)

    print("\nTask C sample preview:")
    print(
        df[
            ["image_id", "diagnosis", "age_approx", "sex", "anatom_site_general"]
        ].head().to_string(index=False)
    )

    existing_res, done = load_existing_done(RESULTS_JSONL) if RESUME else (pd.DataFrame(), set())
    summarize_progress(df, done)

    print("\nComputing current/interim metrics before run...")
    compute_metrics(RESULTS_JSONL, suffix="_interim")

    todo = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["image_id"], variant)

            if not RESUME or key not in done:
                todo.append((row_dict, variant))

    if len(todo) == 0:
        print("\nNothing left to run. Computing final Task C metrics only.")
        compute_metrics(RESULTS_JSONL)
        print("\nTASK C SUCCESS")
        return

    print("\nRunning Task C remaining predictions sequentially on GPU...")
    print("Remaining tasks:", len(todo))

    completed_since_metric = 0
    start_time = time.time()

    for idx, (row_dict, variant) in enumerate(tqdm(todo, total=len(todo)), 1):
        key = (row_dict["image_id"], variant)

        try:
            pred = call_gemma_taskC(row_dict, variant)

            out = {
                "model_id": MODEL_ID,
                "image_id": row_dict["image_id"],
                "variant": variant,
                "true_diagnosis": row_dict["diagnosis"],
                "age_approx_original": None if pd.isna(row_dict["age_approx"]) else float(row_dict["age_approx"]),
                "sex_original": None if pd.isna(row_dict["sex"]) else row_dict["sex"],
                "anatom_site_original": None if pd.isna(row_dict["anatom_site_general"]) else row_dict["anatom_site_general"],
                **pred
            }

            append_jsonl(RESULTS_JSONL, out)
            done.add(key)

        except Exception as e:
            fail = {
                "model_id": MODEL_ID,
                "image_id": row_dict.get("image_id", None),
                "variant": variant,
                "true_diagnosis": row_dict.get("diagnosis", None),
                "error": repr(e),
                "traceback": traceback.format_exc()
            }

            append_jsonl(FAILED_JSONL, fail)

            print(
                "\nFAILED:",
                fail.get("image_id"),
                fail.get("variant"),
                fail.get("error")
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\n[Task C] processed this run: {idx}/{len(todo)} | elapsed={elapsed/60:.1f} min")

        if completed_since_metric >= INTERIM_EVERY:
            completed_since_metric = 0
            print("\n[Task C Interim] computing metrics...")
            compute_metrics(RESULTS_JSONL, suffix="_interim")
            summarize_progress(df, done)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nComputing final Task C metrics...")
    compute_metrics(RESULTS_JSONL)

    print("\nTASK C SUCCESS")
    print("results :", RESULTS_JSONL)
    print("failed  :", FAILED_JSONL)
    print("metrics :", METRICS_CSV)
    print("reliance:", RELIANCE_CSV)
    print("report  :", REPORT_TXT)

# ============================================================
# Run
# ============================================================

main_taskC_gemma4_sequential()
