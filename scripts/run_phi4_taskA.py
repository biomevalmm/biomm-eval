import os, json, time, random, traceback, argparse, re
from datetime import datetime
from typing import List, Literal, Dict, Any

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM

from taskA_prompts import build_taskA_prompt
from taskA_utils import append_jsonl, load_jsonl_safe, clean_model_response, extract_json, safe_float01
from taskA_metrics import compute_metrics


CLASSES = ["Primary Tumor", "Metastatic"]
VARIANTS = [
    "full",
    "clinical_only",
    "rna_only",
    "image_summary_only",
    "clinical_rna",
    "counterfactual_clinical",
    "counterfactual_rna",
]


class TaskAPrediction(BaseModel):
    prediction: Literal["Primary Tumor", "Metastatic"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image_summary", "clinical", "rna"]]
    image_reliance: float = Field(ge=0, le=1)
    clinical_reliance: float = Field(ge=0, le=1)
    rna_reliance: float = Field(ge=0, le=1)
    rationale_short: str


def clean_phi_response(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    # Remove Phi reasoning traces if present.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = text.replace("<|im_sep|>", "")
    text = text.replace("<nothink>", "")

    return clean_model_response(text).strip()


def fallback_extract_prediction(response_text: str) -> Dict[str, Any]:
    text = clean_phi_response(response_text)
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
    text = clean_phi_response(response_text)

    try:
        obj = extract_json(text)
        bad_pred_values = {
            "Primary Tumor or Metastatic",
            "Primary Tumor/Metastatic",
            "Primary or Metastatic",
            "primary tumor or metastatic",
        }
        if str(obj.get("prediction", "")).strip() in bad_pred_values:
            return TaskAPrediction.model_validate(fallback_extract_prediction(text)).model_dump()

    except Exception:
        return TaskAPrediction.model_validate(fallback_extract_prediction(text)).model_dump()

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


def load_test_set(split_csv):
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    df = pd.read_csv(split_csv)

    required = [
        "case_barcode",
        "sample_type",
        "has_image",
        "has_rna",
        "age_at_diagnosis",
        "gender",
        "expr_mean",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df[df["sample_type"].isin(CLASSES)].copy().reset_index(drop=True)

    print("Loaded Task A split:", split_csv)
    print(df["sample_type"].value_counts().reindex(CLASSES))
    print("n =", len(df))

    return df


def load_existing_done(results_path):
    if not os.path.exists(results_path):
        return set()

    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return set()

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()
    return set(zip(res["case_barcode"], res["variant"]))


def normalize_modalities(pred, variant):
    if variant == "clinical_only":
        pred["image_reliance"] = 0.0
        pred["rna_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m == "clinical"]

    elif variant == "rna_only":
        pred["image_reliance"] = 0.0
        pred["clinical_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m == "rna"]

    elif variant == "image_summary_only":
        pred["clinical_reliance"] = 0.0
        pred["rna_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m == "image_summary"]

    elif variant == "clinical_rna":
        pred["image_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m in ["clinical", "rna"]]

    return pred


def build_phi_prompt(prompt: str) -> str:
    # Phi-4 reasoning models may emit reasoning unless explicitly discouraged.
    # We keep the model response constrained to JSON for metric parsing.
    return (
        "<|im_start|>user<|im_sep|>\n"
        "<nothink>\n"
        f"{prompt}\n\n"
        "Return only a valid JSON object. Do not include markdown, explanation, or extra text."
        "<|im_end|>\n"
        "<|im_start|>assistant<|im_sep|>\n"
    )


def generate_phi_response(model, processor, prompt_text, target_device, max_new_tokens):
    inputs = processor(
        text=prompt_text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(target_device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_tokens = generated_ids[0, input_len:]

    output_text = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    ).strip()

    return clean_phi_response(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, default="./examples/taskA_test_full.csv")
    parser.add_argument("--out_dir", type=str, default="./outputs/taskA_phi4_reasoning_vision")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-4-reasoning-vision-15B")
    parser.add_argument("--task_setting", type=str, default="shortcut_controlled")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_results", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"taskA_phi4_reasoning_vision_{run_id}"

    if args.resume_results is not None:
        results_jsonl = args.resume_results
        run_prefix = os.path.basename(results_jsonl).replace("_results.jsonl", "")
    else:
        results_jsonl = os.path.join(args.out_dir, f"{run_prefix}_results.jsonl")

    failed_jsonl = os.path.join(args.out_dir, f"{run_prefix}_failed.jsonl")

    print("=" * 80)
    print("Task A Phi-4-Reasoning-Vision evaluation")
    print("model_id:", args.model_id)
    print("split_csv:", args.split_csv)
    print("results:", results_jsonl)
    print("failed :", failed_jsonl)
    print("=" * 80)

    df = load_test_set(args.split_csv)
    done = load_existing_done(results_jsonl)

    print("\nLoading processor/model...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model.eval()

    target_device = next(model.parameters()).device
    print("CUDA available:", torch.cuda.is_available())
    print("target device:", target_device)

    todo = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for variant in VARIANTS:
            key = (row_dict["case_barcode"], variant)
            if key not in done:
                todo.append((row_dict, variant))

    print("Already completed:", len(done))
    print("Remaining evaluations:", len(todo))

    for row_dict, variant in tqdm(todo):
        success = False
        last_error = None

        for attempt in range(1, args.max_retries + 1):
            try:
                prompt = build_taskA_prompt(
                    row_dict,
                    variant=variant,
                    task_setting=args.task_setting,
                )

                phi_prompt = build_phi_prompt(prompt)

                output_text = generate_phi_response(
                    model=model,
                    processor=processor,
                    prompt_text=phi_prompt,
                    target_device=target_device,
                    max_new_tokens=args.max_new_tokens,
                )

                pred = parse_prediction_response(output_text)
                pred = normalize_modalities(pred, variant)
                pred["raw_output"] = clean_phi_response(output_text)

                out = {
                    "model_id": args.model_id,
                    "case_barcode": row_dict["case_barcode"],
                    "variant": variant,
                    "true_label": row_dict["sample_type"],
                    "sample_type": row_dict.get("sample_type"),
                    "gender": None if pd.isna(row_dict.get("gender")) else row_dict.get("gender"),
                    "age_at_diagnosis": None if pd.isna(row_dict.get("age_at_diagnosis")) else row_dict.get("age_at_diagnosis"),
                    **pred,
                }

                append_jsonl(results_jsonl, out)
                success = True
                break

            except Exception as e:
                last_error = e

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                time.sleep(1)

        if not success:
            fail = {
                "model_id": args.model_id,
                "case_barcode": row_dict.get("case_barcode"),
                "variant": variant,
                "true_label": row_dict.get("sample_type"),
                "error": repr(last_error),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(failed_jsonl, fail)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compute_metrics(
        results_path=results_jsonl,
        out_dir=args.out_dir,
        run_prefix=run_prefix,
    )

    print("\nDone.")
    print("results:", results_jsonl)
    print("failed :", failed_jsonl)


if __name__ == "__main__":
    main()
