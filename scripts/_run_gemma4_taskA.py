# ============================================================
# Task A BioMM-Eval
# Gemma-4-26B-A4B-it Sequential GPU Version (JSON-stable)
# ============================================================

import os, re, json, time, random, traceback
from datetime import datetime
from typing import List, Literal, Any, Dict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

# -------------------------
# Config
# -------------------------
MODEL_ID = "google/gemma-4-26B-A4B-it"

BASE_DIR = ""

OUT_DIR = os.path.join(BASE_DIR, "biomm_taskA_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_FULL_CSV = os.path.join(OUT_DIR, "taskA_test_full.csv")

TARGET_COL = "sample_type"
CLASSES = ["Primary Tumor", "Metastatic"]

MAX_NEW_TOKENS = 256
MAX_RETRIES = 3
DO_SAMPLE = False

VARIANTS = [
    "full", "clinical_only", "rna_only",
    "image_summary_only", "clinical_rna"
]

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_JSONL = os.path.join(OUT_DIR, f"taskA_gemma4_results_{RUN_ID}.jsonl")

# -------------------------
# Schema
# -------------------------
class TaskAPrediction(BaseModel):
    prediction: Literal["Primary Tumor", "Metastatic"]
    confidence: float
    used_modalities: List[str]
    image_reliance: float
    clinical_reliance: float
    rna_reliance: float
    rationale_short: str

# -------------------------
# Helpers
# -------------------------
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def extract_json(text):
    text = text.strip()

    start = text.find("{")
    if start < 0:
        raise RuntimeError("No JSON found")

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj

# -------------------------
# Prompt
# -------------------------
def build_prompt(row, variant):
    return f"""
Predict TCGA-SKCM sample type.

Allowed:
Primary Tumor, Metastatic

Clinical:
age: {row.get("age_at_diagnosis")}
gender: {row.get("gender")}

RNA:
mean: {row.get("expr_mean")}

Return JSON only:

{{
 "prediction": "",
 "confidence": 0.0,
 "used_modalities": [],
 "image_reliance": 0.0,
 "clinical_reliance": 0.0,
 "rna_reliance": 0.0,
 "rationale_short": ""
}}
"""

# -------------------------
# Load model
# -------------------------
print("Loading Gemma 4...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# -------------------------
# Inference
# -------------------------
def run_model(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False   # 🔥 핵심
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE
        )

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response

# -------------------------
# Main
# -------------------------
df = pd.read_csv(TEST_FULL_CSV)

print("Start inference...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    for variant in VARIANTS:
        for attempt in range(MAX_RETRIES):
            try:
                prompt = build_prompt(row, variant)
                output = run_model(prompt)

                obj = extract_json(output)

                append_jsonl(RESULTS_JSONL, {
                    "case_barcode": row["case_barcode"],
                    "variant": variant,
                    "true_label": row[TARGET_COL],
                    **obj
                })
                break

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print("FAILED:", row["case_barcode"], variant, e)
                time.sleep(2 ** attempt)

print("DONE")
