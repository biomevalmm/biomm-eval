# ============================================================
# Task C BioMM-Eval v3
# Qwen3.6-27B local Transformers Image-Text-to-Text version
# Full / Image-only / Metadata-only / Counterfactual
# Resume + retry + interim metrics + final metrics
# Multi-GPU device_map="auto" optimized
# ============================================================

import os, json, random, time, traceback, gc
from datetime import datetime
from typing import List, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------
# Config
# -------------------------
EVAL_MODE = "balanced"  # "balanced" or "full"

MODEL = "Qwen/Qwen3.6-27B"
MODEL_TAG = "qwen36_27b"

BASE_DIR = ""
MASTER_CSV = os.path.join(BASE_DIR, "taskC_master.csv")
OUT_DIR = os.path.join(BASE_DIR, "biomm_taskC_multiclass_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

CLASSES = ["MEL", "NV", "BCC", "BKL"]

VARIANTS = [
    "full",
    "image_only",
    "metadata_only",
    "metadata_counterfactual"
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

MAX_INPUT_TOKENS = 12000
MAX_NEW_TOKENS = 512
IMAGE_MAX_SIZE = 768
MAX_RETRIES = 4
INTERIM_EVERY = 50

# GPU memory 자동 설정: 사용 가능한 모든 GPU 사용
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    MAX_MEMORY = {i: "75GiB" for i in range(GPU_COUNT)}
else:
    GPU_COUNT = 0
    MAX_MEMORY = None

print("MODEL:", MODEL)
print("RUN_ID:", RUN_ID)
print("RESULTS:", RESULTS_JSONL)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", GPU_COUNT)
    for i in range(GPU_COUNT):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print("MAX_MEMORY:", MAX_MEMORY)

# -------------------------
# Load Qwen3.6 Image-Text-to-Text
# -------------------------
print("\nLoading Qwen3.6 model...")

processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

if MAX_MEMORY is not None:
    model_kwargs["max_memory"] = MAX_MEMORY

# flash_attention_2가 설치되어 있으면 사용, 아니면 자동 fallback
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        attn_implementation="flash_attention_2",
        **model_kwargs
    )
    print("Loaded with flash_attention_2.")
except Exception as e:
    print("flash_attention_2 load failed; fallback to default attention.")
    print("Reason:", repr(e))
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        **model_kwargs
    )

model.eval()

tokenizer = processor.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded.")

if hasattr(model, "hf_device_map"):
    print("\nDevice map:")
    print(model.hf_device_map)

# -------------------------
# Structured output schema
# -------------------------
class TaskCPrediction(BaseModel):
    prediction: Literal["MEL", "NV", "BCC", "BKL"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image", "metadata"]]
    image_reliance: float = Field(ge=0, le=1)
    metadata_reliance: float = Field(ge=0, le=1)
    rationale_short: str


# -------------------------
# IO helpers
# -------------------------
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl_safe(path):
    rows = []

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                print(f"[load_jsonl_safe] skipped broken JSONL line: {line_num}")

    return pd.DataFrame(rows)


# -------------------------
# General helpers
# -------------------------
def clean_val(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    if x.lower() in [
        "",
        "nan",
        "none",
        "null",
        "missing",
        "not reported",
        "not available",
        "not provided"
    ]:
        return "unknown"

    return x


def safe_float01(x, default=0.0):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def load_image(path, max_size=IMAGE_MAX_SIZE):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------
# Prompt helpers
# -------------------------
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


def build_prompt(row, variant):
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

Return ONLY a JSON object.

Valid JSON format:
{{
  "prediction": "MEL",
  "confidence": 0.0,
  "used_modalities": ["image", "metadata"],
  "image_reliance": 0.0,
  "metadata_reliance": 0.0,
  "rationale_short": "max 15 words"
}}

Rules:
- Output MUST be valid JSON.
- Do NOT include any text before or after the JSON.
- Do NOT include markdown or ```json.
- Do NOT include reasoning outside JSON.
- prediction MUST be exactly one of: MEL, NV, BCC, BKL.
- confidence must be between 0 and 1.
- image_reliance and metadata_reliance must be between 0 and 1.
- If a modality is not provided, its reliance must be 0.
- rationale_short MUST be at most 15 words.
- Do NOT include step-by-step reasoning.
""".strip()

    return prompt, include_image, meta_used


def parse_prediction_response(response_text):
    text = str(response_text).strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return TaskCPrediction.model_validate_json(text).model_dump()
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])

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

    raise RuntimeError(f"Could not parse JSON from response: {text[:1000]}")


def apply_qwen_chat_template(messages):
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def get_input_device():
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


# -------------------------
# Model call
# -------------------------
def call_model_once(row, variant):
    prompt, include_image, meta_used = build_prompt(row, variant)

    content = []
    image = None

    if include_image:
        image = load_image(row["image_path"])
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a strict JSON generator. "
                        "You MUST output ONLY a valid JSON object. "
                        "Do NOT include explanations, reasoning, or markdown."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": content
        }
    ]

    text = apply_qwen_chat_template(messages)

    if include_image:
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        )

    input_device = get_input_device()

    inputs = {
        k: v.to(input_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    obj = parse_prediction_response(response_text)
    obj["metadata_used"] = meta_used

    if variant == "metadata_only":
        obj["image_reliance"] = 0.0
        obj["used_modalities"] = [
            m for m in obj["used_modalities"]
            if m == "metadata"
        ]

    elif variant == "image_only":
        obj["metadata_reliance"] = 0.0
        obj["used_modalities"] = [
            m for m in obj["used_modalities"]
            if m == "image"
        ]

    return obj, response_text


def call_model(row, variant, max_retries=MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            obj, raw = call_model_once(row, variant)
            return obj

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            cleanup_cuda()
            wait = min(90, 2 ** attempt * 5)
            print(
                f"\n[CUDA OOM Retry] image_id={row.get('image_id')} "
                f"variant={variant} attempt={attempt + 1}/{max_retries} wait={wait}s"
            )
            time.sleep(wait)

        except Exception as e:
            last_error = e
            cleanup_cuda()
            wait = min(90, 2 ** attempt * 5)
            print(
                f"\n[Retry] image_id={row.get('image_id')} "
                f"variant={variant} attempt={attempt + 1}/{max_retries} "
                f"error={repr(e)} wait={wait}s"
            )
            time.sleep(wait)

    raise last_error


# -------------------------
# Dataset
# -------------------------
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


def load_or_create_test_sets(mode=EVAL_MODE):
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
        raise RuntimeError(f"Missing required columns: {missing}")

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
        print("\nUsing FULL test set")
    elif mode == "balanced":
        out = pd.read_csv(TEST_BAL_CSV)
        print("\nUsing BALANCED test set")
    else:
        raise ValueError("mode must be 'full' or 'balanced'")

    out = out[
        out["image_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    ].reset_index(drop=True)

    print(out["diagnosis"].value_counts())
    print("n =", len(out))

    return out


# -------------------------
# Resume / metrics
# -------------------------
def load_existing_done(results_path):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    needed = {"image_id", "variant"}

    if not needed.issubset(set(res.columns)):
        print("[Resume] Existing results missing required columns. Ignoring existing done set.")
        return res, set()

    res = res.drop_duplicates(subset=["image_id", "variant"], keep="last").copy()
    done = set(zip(res["image_id"], res["variant"]))

    return res, done


def compute_metrics(results_path, suffix=""):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        print("[Metrics] No results yet.")
        return None

    required = {"image_id", "variant", "prediction", "true_diagnosis"}

    if not required.issubset(set(res.columns)):
        print("[Metrics] Missing required columns:", required - set(res.columns))
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

        rows.append({
            "eval_mode": EVAL_MODE,
            "variant": variant,
            "n": len(g),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mean_confidence": float(g["confidence"].mean()) if "confidence" in g.columns else None,
            "mean_image_reliance_self_report": float(g["image_reliance"].mean()) if "image_reliance" in g.columns else None,
            "mean_metadata_reliance_self_report": float(g["metadata_reliance"].mean()) if "metadata_reliance" in g.columns else None
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

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)

        cm_csv = os.path.join(CONFUSION_DIR, f"confusion_{variant}{suffix}.csv")
        cm_png = os.path.join(CONFUSION_DIR, f"confusion_{variant}{suffix}.png")

        cm_df.to_csv(cm_csv)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix: {variant}{suffix}")
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
        "eval_mode": EVAL_MODE,
        "full_balanced_accuracy": full,
        "image_only_balanced_accuracy": image_only,
        "metadata_only_balanced_accuracy": metadata_only,
        "counterfactual_balanced_accuracy": cf,
        "metadata_necessity_score_full_minus_image_only": None if full is None or image_only is None else full - image_only,
        "image_necessity_score_full_minus_metadata_only": None if full is None or metadata_only is None else full - metadata_only,
        "counterfactual_sensitivity_full_minus_cf": None if full is None or cf is None else full - cf
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

    reliance["counterfactual_prediction_change_rate"] = change_rate("full", "metadata_counterfactual")
    reliance["full_vs_image_only_prediction_change_rate"] = change_rate("full", "image_only")
    reliance["full_vs_metadata_only_prediction_change_rate"] = change_rate("full", "metadata_only")

    reliance_df = pd.DataFrame([reliance])
    reliance_df.to_csv(reliance_path, index=False)

    print(f"\n=== Metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Reliance summary {suffix or 'final/current'} ===")
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

    print("\n=== Progress ===")
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
    print(" Saved / Resume Status")
    print("==============================")
    print("RUN_ID:", RUN_ID)
    print("RESULTS_JSONL:", RESULTS_JSONL)
    print("FAILED_JSONL:", FAILED_JSONL)

    res = load_jsonl_safe(RESULTS_JSONL)

    if len(res):
        res_dedup = res.drop_duplicates(subset=["image_id", "variant"], keep="last")
        print("\n[results file exists]")
        print("size bytes:", os.path.getsize(RESULTS_JSONL))
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
        print("\nNo results yet.")

    fail = load_jsonl_safe(FAILED_JSONL)

    if len(fail):
        print("\nFailed rows:", len(fail))
        print(fail.tail().to_string(index=False))
    else:
        print("\nNo failed file or failed file is empty.")


# -------------------------
# Main
# -------------------------
def main():
    print_saved_status()

    df = load_or_create_test_sets(EVAL_MODE)

    preview_cols = [
        "image_id",
        "diagnosis",
        "age_approx",
        "sex",
        "anatom_site_general",
        "image_path"
    ]

    print("\nSamples:", len(df))
    print(df[preview_cols].head().to_string(index=False))

    _, done = load_existing_done(RESULTS_JSONL)
    summarize_progress(df, done)

    print("\nComputing current/interim metrics before run...")
    compute_metrics(RESULTS_JSONL, suffix="_interim")

    expected_total = len(df) * len(VARIANTS)
    total_remaining = expected_total - len(done)

    if total_remaining <= 0:
        print("\nNothing left to run. Computing final metrics only.")
        compute_metrics(RESULTS_JSONL)
        print("\nSUCCESS")
        return

    print("\nRunning remaining predictions...")
    print("Remaining tasks:", total_remaining)

    completed_since_metric = 0
    pbar = tqdm(total=total_remaining)

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["image_id"], variant)

            if key in done:
                continue

            try:
                pred = call_model(row_dict, variant)

                age_val = row_dict.get("age_approx")
                try:
                    age_val = None if pd.isna(age_val) else float(age_val)
                except Exception:
                    age_val = None

                out = {
                    "eval_mode": EVAL_MODE,
                    "image_id": row_dict["image_id"],
                    "variant": variant,
                    "true_diagnosis": row_dict["diagnosis"],
                    "image_path": row_dict["image_path"],
                    "age_approx_original": age_val,
                    "sex_original": None if pd.isna(row_dict.get("sex")) else row_dict.get("sex"),
                    "anatom_site_original": None if pd.isna(row_dict.get("anatom_site_general")) else row_dict.get("anatom_site_general"),
                    **pred
                }

                append_jsonl(RESULTS_JSONL, out)
                done.add(key)

                pbar.update(1)
                completed_since_metric += 1

                if completed_since_metric >= INTERIM_EVERY:
                    completed_since_metric = 0
                    print("\n[Interim] computing metrics...")
                    compute_metrics(RESULTS_JSONL, suffix="_interim")
                    summarize_progress(df, done)

            except Exception as e:
                fail = {
                    "eval_mode": EVAL_MODE,
                    "image_id": row_dict.get("image_id", None),
                    "variant": variant,
                    "true_diagnosis": row_dict.get("diagnosis", None),
                    "image_path": row_dict.get("image_path", None),
                    "error": repr(e),
                    "traceback": traceback.format_exc()
                }

                append_jsonl(FAILED_JSONL, fail)
                print("\nFAILED:", row_dict.get("image_id", None), variant, repr(e))
                pbar.update(1)
                cleanup_cuda()
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
