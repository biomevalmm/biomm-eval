# ============================================================
# Task B BioMM-Eval v2
# Gemma-4-26B-A4B-it Sequential GPU Version
# Genome sequence + metadata
# FULL dataset, no class sampling
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
from transformers import AutoProcessor, AutoModelForCausalLM

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# ============================================================
# Config
# ============================================================

MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_TAG = "gemma4_26b_a4b_it"

BASE_DIR = ""
SEQ_CSV = os.path.join(BASE_DIR, "selected_sequences_saureus199_cacnes200.csv")

OUT_DIR = os.path.join(BASE_DIR, "biomm_taskB_v2_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


RESULTS_JSONL = os.path.join(OUT_DIR, f"taskB_{MODEL_TAG}_full_results_{RUN_ID}.jsonl")
FAILED_JSONL = os.path.join(OUT_DIR, f"taskB_{MODEL_TAG}_full_failed_{RUN_ID}.jsonl")
METRICS_CSV = os.path.join(OUT_DIR, f"taskB_{MODEL_TAG}_full_metrics_{RUN_ID}.csv")
RELIANCE_CSV = os.path.join(OUT_DIR, f"taskB_{MODEL_TAG}_full_reliance_{RUN_ID}.csv")
REPORT_TXT = os.path.join(OUT_DIR, f"taskB_{MODEL_TAG}_full_report_{RUN_ID}.txt")

CLASSES = [
    "Staphylococcus aureus",
    "Cutibacterium acnes"
]

SEED = 42
MAX_SEQUENCE_CHARS = 8000
PERTURB_RATE = 0.01

VARIANTS = [
    "full",
    "sequence_only",
    "metadata_only",
    "metadata_counterfactual",
    "sequence_perturbed"
]

RESUME = True
INTERIM_EVERY = 50
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
MAX_RETRIES = 4

random.seed(SEED)

print("=" * 90)
print("TASK          : Task B BioMM-Eval v2")
print("MODEL_ID      :", MODEL_ID)
print("RUN_ID        :", RUN_ID)
print("SEQ_CSV       :", SEQ_CSV)
print("RESULTS_JSONL :", RESULTS_JSONL)
print("=" * 90)

# ============================================================
# Schema
# ============================================================

class TaskBPrediction(BaseModel):
    prediction: Literal["Staphylococcus aureus", "Cutibacterium acnes"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["sequence", "metadata"]]
    sequence_reliance: float = Field(ge=0, le=1)
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

    if x.lower() in [
        "",
        "nan",
        "none",
        "null",
        "not collected",
        "missing",
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


def normalize_organism_name(x):
    x = clean_val(x)

    if "Staphylococcus aureus" in x:
        return "Staphylococcus aureus"

    if "Cutibacterium acnes" in x or "Propionibacterium acnes" in x:
        return "Cutibacterium acnes"

    return x


def clean_gemma_response(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    if hasattr(processor, "parse_response"):
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
    obj["sequence_reliance"] = safe_float01(obj.get("sequence_reliance", 0.0))
    obj["metadata_reliance"] = safe_float01(obj.get("metadata_reliance", 0.0))

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["sequence", "metadata"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskBPrediction.model_validate(obj).model_dump()

# ============================================================
# Sequence / metadata prompt helpers
# ============================================================

def sequence_context(seq, max_chars=MAX_SEQUENCE_CHARS):
    if pd.isna(seq):
        return "unknown", 0, 0

    seq = str(seq).upper()
    seq = "".join(c for c in seq if c in "ACGTN")

    full_len = len(seq)

    if full_len == 0:
        return "unknown", 0, 0

    if full_len <= max_chars:
        return seq, full_len, full_len

    part = max_chars // 3

    start = seq[:part]

    mid_start = max(0, full_len // 2 - part // 2)
    middle = seq[mid_start:mid_start + part]

    end = seq[-part:]

    context = (
        start
        + "\n...[MIDDLE_SEGMENT]...\n"
        + middle
        + "\n...[END_SEGMENT]...\n"
        + end
    )

    return context, full_len, len(start) + len(middle) + len(end)


def perturb_sequence(seq_text, mutation_rate=PERTURB_RATE):
    bases = ["A", "C", "G", "T"]
    out = []

    for c in seq_text:
        if c in bases and random.random() < mutation_rate:
            choices = [b for b in bases if b != c]
            out.append(random.choice(choices))
        else:
            out.append(c)

    return "".join(out)


def metadata_block(row, counterfactual=False):
    host = clean_val(row.get("host"))
    host_disease = clean_val(row.get("host_disease"))
    isolation_source = clean_val(row.get("isolation_source"))
    body_site = clean_val(row.get("body_site"))
    geo = clean_val(row.get("geo_loc_name"))
    strain = clean_val(row.get("strain"))
    isolate = clean_val(row.get("isolate"))
    assembly_level = clean_val(row.get("assembly_level"))
    refseq_category = clean_val(row.get("refseq_category"))
    submitter = clean_val(row.get("submitter"))

    if counterfactual:
        host_choices = ["Homo sapiens", "human", "environment", "mouse", "unknown"]
        disease_choices = [
            "acne",
            "skin infection",
            "wound infection",
            "healthy skin",
            "bloodstream infection",
            "unknown"
        ]
        source_choices = [
            "skin",
            "acne lesion",
            "wound",
            "blood",
            "nasal swab",
            "environment",
            "unknown"
        ]
        body_choices = [
            "skin",
            "face",
            "back",
            "wound",
            "blood",
            "nasal cavity",
            "unknown"
        ]

        host = random.choice([x for x in host_choices if x != host])
        host_disease = random.choice([x for x in disease_choices if x != host_disease])
        isolation_source = random.choice([x for x in source_choices if x != isolation_source])
        body_site = random.choice([x for x in body_choices if x != body_site])

    return f"""
Metadata:
- host: {host}
- host_disease: {host_disease}
- isolation_source: {isolation_source}
- body_site: {body_site}
- geo_loc_name: {geo}
- strain: {strain}
- isolate: {isolate}
- assembly_level: {assembly_level}
- refseq_category: {refseq_category}
- submitter: {submitter}
""".strip()


def build_taskB_prompt(row, variant):
    seq_ctx, full_len, used_len = sequence_context(
        row.get("joined_sequence", ""),
        MAX_SEQUENCE_CHARS
    )

    if variant == "full":
        instruction = "You are given long genome DNA sequence segments and biological metadata."
        blocks = [
            metadata_block(row, counterfactual=False),
            f"""
DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- sampling_strategy: beginning + middle + end segments

{seq_ctx}
""".strip()
        ]

    elif variant == "sequence_only":
        instruction = "You are given only long genome DNA sequence segments. No metadata is provided."
        blocks = [
            f"""
DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- sampling_strategy: beginning + middle + end segments

{seq_ctx}
""".strip()
        ]

    elif variant == "metadata_only":
        instruction = "You are given only biological metadata. No DNA sequence is provided."
        blocks = [
            metadata_block(row, counterfactual=False)
        ]

    elif variant == "metadata_counterfactual":
        instruction = "You are given original long genome DNA sequence segments but counterfactually modified metadata."
        blocks = [
            metadata_block(row, counterfactual=True),
            f"""
DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- sampling_strategy: beginning + middle + end segments

{seq_ctx}
""".strip()
        ]

    elif variant == "sequence_perturbed":
        instruction = "You are given perturbed long genome DNA sequence segments and original biological metadata."
        perturbed = perturb_sequence(seq_ctx, PERTURB_RATE)
        blocks = [
            metadata_block(row, counterfactual=False),
            f"""
Perturbed DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- perturbation_rate: {PERTURB_RATE}
- sampling_strategy: beginning + middle + end segments

{perturbed}
""".strip()
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    joined = "\n\n".join(blocks)

    prompt = f"""
Task:
Classify the bacterial genome into exactly one organism.

Allowed labels:
- Staphylococcus aureus
- Cutibacterium acnes

{instruction}

{joined}

This is for benchmark evaluation only.

Return only valid JSON. Do not use markdown. Do not explain.

Required JSON:
{{
  "prediction": "Staphylococcus aureus or Cutibacterium acnes",
  "confidence": 0.0,
  "used_modalities": ["sequence", "metadata"],
  "sequence_reliance": 0.0,
  "metadata_reliance": 0.0,
  "rationale_short": "brief reason"
}}

Rules:
- prediction must be exactly one of: Staphylococcus aureus, Cutibacterium acnes.
- confidence must be between 0 and 1.
- sequence_reliance and metadata_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()

    return prompt, full_len, used_len

# ============================================================
# Dataset
# ============================================================

def load_full_data():
    if not os.path.exists(SEQ_CSV):
        raise FileNotFoundError(f"Missing file: {SEQ_CSV}")

    df = pd.read_csv(SEQ_CSV)

    required = [
        "assembly_accession",
        "organism_name",
        "joined_sequence",
        "joined_seq_len"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["label"] = df["organism_name"].apply(normalize_organism_name)
    df = df[df["label"].isin(CLASSES)].copy()

    df = df[df["joined_sequence"].notna()].copy()
    df["joined_seq_len_numeric"] = pd.to_numeric(df["joined_seq_len"], errors="coerce")
    df = df[df["joined_seq_len_numeric"] > 1000].copy()

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("Using FULL Task B dataset, no class sampling")
    print("Class counts:")
    print(df["label"].value_counts().to_string())
    print("n =", len(df))

    return df


def load_existing_done(results_path):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    res = res.drop_duplicates(
        subset=["assembly_accession", "variant"],
        keep="last"
    ).copy()

    done = set(zip(res["assembly_accession"], res["variant"]))

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

def call_gemma_taskB_once(row, variant):
    prompt, full_len, used_len = build_taskB_prompt(row, variant)

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
    pred["raw_output"] = output_text

    if variant == "metadata_only":
        pred["sequence_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m != "sequence"
        ]

    if variant == "sequence_only":
        pred["metadata_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m != "metadata"
        ]

    pred["full_sequence_length"] = full_len
    pred["provided_sequence_length"] = used_len

    return pred


def call_gemma_taskB(row, variant, max_retries=MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_gemma_taskB_once(row, variant)

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** attempt * 3)

            print(
                f"\n[Retry] accession={row.get('assembly_accession')} "
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

    res = res.drop_duplicates(
        subset=["assembly_accession", "variant"],
        keep="last"
    ).copy()

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
            "variant": variant,
            "n": len(g),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mean_confidence": float(g["confidence"].mean()),
            "mean_sequence_reliance_self_report": float(g["sequence_reliance"].mean()),
            "mean_metadata_reliance_self_report": float(g["metadata_reliance"].mean()),
            "mean_provided_sequence_length": float(g["provided_sequence_length"].mean()),
            "mean_full_sequence_length": float(g["full_sequence_length"].mean())
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
    seq = score.get("sequence_only")
    meta = score.get("metadata_only")
    cf = score.get("metadata_counterfactual")
    pert = score.get("sequence_perturbed")

    reliance = {
        "full_balanced_accuracy": full,
        "sequence_only_balanced_accuracy": seq,
        "metadata_only_balanced_accuracy": meta,
        "metadata_counterfactual_balanced_accuracy": cf,
        "sequence_perturbed_balanced_accuracy": pert,
        "metadata_necessity_score_full_minus_sequence_only": None if full is None or seq is None else full - seq,
        "sequence_necessity_score_full_minus_metadata_only": None if full is None or meta is None else full - meta,
        "metadata_counterfactual_sensitivity_full_minus_cf": None if full is None or cf is None else full - cf,
        "sequence_robustness_drop_full_minus_perturbed": None if full is None or pert is None else full - pert
    }

    wide = res.pivot_table(
        index="assembly_accession",
        columns="variant",
        values="prediction",
        aggfunc="first"
    ).reset_index()

    for other in [
        "sequence_only",
        "metadata_only",
        "metadata_counterfactual",
        "sequence_perturbed"
    ]:
        key = f"full_vs_{other}_prediction_change_rate"

        if "full" in wide.columns and other in wide.columns:
            valid = wide.dropna(subset=["full", other]).copy()
            reliance[key] = float((valid["full"] != valid[other]).mean()) if len(valid) else None
        else:
            reliance[key] = None

    reliance_df = pd.DataFrame([reliance])
    reliance_df.to_csv(reliance_path, index=False)

    print(f"\n=== Task B Metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Task B Reliance summary {suffix or 'final/current'} ===")
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

    print("\n=== Task B Progress ===")
    print("samples:", len(df))
    print("variants:", len(VARIANTS))
    print("expected total rows:", expected)
    print("completed rows:", completed)
    print("remaining rows:", remaining)
    print("completion rate:", round(completed / expected * 100, 2), "%")

    if done:
        done_df = pd.DataFrame(list(done), columns=["assembly_accession", "variant"])

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
    print(" Task B Saved / Resume Status")
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
                subset=["assembly_accession", "variant"],
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

def main_taskB_gemma4_sequential():
    print_saved_status()

    df = load_full_data()

    print("\nTask B sample preview:")
    preview_cols = [
        "assembly_accession",
        "label",
        "organism_name",
        "host",
        "host_disease",
        "isolation_source",
        "body_site",
        "joined_seq_len"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))

    existing_res, done = load_existing_done(RESULTS_JSONL) if RESUME else (pd.DataFrame(), set())
    summarize_progress(df, done)

    print("\nComputing current/interim metrics before run...")
    compute_metrics(RESULTS_JSONL, suffix="_interim")

    todo = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["assembly_accession"], variant)

            if not RESUME or key not in done:
                todo.append((row_dict, variant))

    if len(todo) == 0:
        print("\nNothing left to run. Computing final Task B metrics only.")
        compute_metrics(RESULTS_JSONL)
        print("\nTASK B SUCCESS")
        return

    print("\nRunning Task B remaining predictions sequentially on GPU...")
    print("Remaining tasks:", len(todo))

    completed_since_metric = 0
    start_time = time.time()

    for idx, (row_dict, variant) in enumerate(tqdm(todo, total=len(todo)), 1):
        try:
            pred = call_gemma_taskB(row_dict, variant)

            out = {
                "model_id": MODEL_ID,
                "assembly_accession": row_dict["assembly_accession"],
                "variant": variant,
                "true_label": row_dict["label"],
                "organism_name_raw": row_dict.get("organism_name"),
                "joined_seq_len": None if pd.isna(row_dict.get("joined_seq_len")) else int(float(row_dict.get("joined_seq_len"))),
                "host": None if pd.isna(row_dict.get("host", np.nan)) else row_dict.get("host"),
                "host_disease": None if pd.isna(row_dict.get("host_disease", np.nan)) else row_dict.get("host_disease"),
                "isolation_source": None if pd.isna(row_dict.get("isolation_source", np.nan)) else row_dict.get("isolation_source"),
                "body_site": None if pd.isna(row_dict.get("body_site", np.nan)) else row_dict.get("body_site"),
                **pred
            }

            append_jsonl(RESULTS_JSONL, out)
            done.add((row_dict["assembly_accession"], variant))

        except Exception as e:
            fail = {
                "model_id": MODEL_ID,
                "assembly_accession": row_dict.get("assembly_accession", None),
                "variant": variant,
                "true_label": row_dict.get("label", None),
                "error": repr(e),
                "traceback": traceback.format_exc()
            }

            append_jsonl(FAILED_JSONL, fail)

            print(
                "\nFAILED:",
                fail.get("assembly_accession"),
                fail.get("variant"),
                fail.get("error")
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\n[Task B] processed this run: {idx}/{len(todo)} | elapsed={elapsed/60:.1f} min")

        if completed_since_metric >= INTERIM_EVERY:
            completed_since_metric = 0
            print("\n[Task B Interim] computing metrics...")
            compute_metrics(RESULTS_JSONL, suffix="_interim")
            summarize_progress(df, done)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nComputing final Task B metrics...")
    compute_metrics(RESULTS_JSONL)

    print("\nTASK B SUCCESS")
    print("results :", RESULTS_JSONL)
    print("failed  :", FAILED_JSONL)
    print("metrics :", METRICS_CSV)
    print("reliance:", RELIANCE_CSV)
    print("report  :", REPORT_TXT)

# ============================================================
# Run
# ============================================================

main_taskB_gemma4_sequential()
