import os, re, json, time, random, traceback, argparse, hashlib
from datetime import datetime
from typing import List, Literal, Any, Dict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

CLASSES = ["MEL", "NV", "BCC", "BKL"]

VARIANTS = [
    "full",
    "image_only",
    "metadata_only",
    "metadata_counterfactual",
]


class TaskCPrediction(BaseModel):
    prediction: Literal["MEL", "NV", "BCC", "BKL"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["image", "metadata"]]
    image_reliance: float = Field(ge=0, le=1)
    metadata_reliance: float = Field(ge=0, le=1)
    rationale_short: str


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    if x.lower() in ["", "nan", "none", "null", "missing", "unknown", "not provided"]:
        return "unknown"
    return x


def safe_float01(x, default=0.0):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def clean_deepseek_response(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace("<|User|>", "")
    text = text.replace("<|Assistant|>", "")
    text = text.replace("<｜User｜>", "")
    text = text.replace("<｜Assistant｜>", "")
    text = text.replace("</s>", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def extract_json(text: str):
    if text is None:
        raise RuntimeError("Empty response text")

    text = clean_deepseek_response(text)
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
    text = clean_deepseek_response(response_text)
    upper = text.upper()

    found = None
    for c in CLASSES:
        if c in upper:
            found = c
            break

    if found is None:
        aliases = {
            "MELANOMA": "MEL",
            "MELANOCYTIC NEVUS": "NV",
            "NEVUS": "NV",
            "BASAL CELL CARCINOMA": "BCC",
            "BENIGN KERATOSIS": "BKL",
            "KERATOSIS": "BKL",
        }
        for k, v in aliases.items():
            if k in upper:
                found = v
                break

    if found is None:
        raise RuntimeError(f"Could not find Task C prediction in response: {text[:500]}")

    return {
        "prediction": found,
        "confidence": 0.5,
        "used_modalities": ["image", "metadata"],
        "image_reliance": 0.5,
        "metadata_reliance": 0.5,
        "rationale_short": text[:500],
    }


def parse_prediction_response(response_text: str) -> Dict[str, Any]:
    text = clean_deepseek_response(response_text)

    try:
        obj = extract_json(text)
    except Exception:
        return TaskCPrediction.model_validate(
            fallback_extract_prediction(text)
        ).model_dump()

    pred_raw = str(obj.get("prediction", "")).strip().upper()

    aliases = {
        "MELANOMA": "MEL",
        "MEL": "MEL",
        "NV": "NV",
        "NEVUS": "NV",
        "MELANOCYTIC NEVUS": "NV",
        "BCC": "BCC",
        "BASAL CELL CARCINOMA": "BCC",
        "BKL": "BKL",
        "BENIGN KERATOSIS": "BKL",
        "BENIGN KERATOSIS-LIKE LESION": "BKL",
    }

    if pred_raw in aliases:
        obj["prediction"] = aliases[pred_raw]

    obj["confidence"] = safe_float01(obj.get("confidence", 0.0), default=0.5)
    obj["image_reliance"] = safe_float01(obj.get("image_reliance", 0.0), default=0.5)
    obj["metadata_reliance"] = safe_float01(obj.get("metadata_reliance", 0.0), default=0.5)

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["image", "metadata"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskCPrediction.model_validate(obj).model_dump()


def stable_rng(seed: int, key: str) -> random.Random:
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def normalize_label(x):
    x = clean_val(x).upper()
    aliases = {
        "MEL": "MEL",
        "MELANOMA": "MEL",
        "NV": "NV",
        "NEVUS": "NV",
        "MELANOCYTIC NEVUS": "NV",
        "BCC": "BCC",
        "BASAL CELL CARCINOMA": "BCC",
        "BKL": "BKL",
        "BENIGN KERATOSIS": "BKL",
        "BENIGN KERATOSIS-LIKE LESION": "BKL",
    }
    return aliases.get(x, x)


def resolve_column(df, preferred, candidates, required=True):
    if preferred and preferred in df.columns:
        return preferred

    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise RuntimeError(f"Could not resolve required column. Tried: {[preferred] + candidates}")

    return None


def metadata_block(row, counterfactual=False, rng=None):
    age = clean_val(row.get("age_approx", row.get("age", row.get("age_at_diagnosis"))))
    sex = clean_val(row.get("sex", row.get("gender")))
    site = clean_val(row.get("anatom_site_general", row.get("anatomical_site", row.get("body_site"))))
    lesion_id = clean_val(row.get("lesion_id"))
    image_id = clean_val(row.get("image_id", row.get("isic_id")))

    if counterfactual:
        if rng is None:
            rng = random

        age_choices = ["20", "35", "50", "65", "80", "unknown"]
        sex_choices = ["male", "female", "unknown"]
        site_choices = [
            "head/neck",
            "upper extremity",
            "lower extremity",
            "anterior torso",
            "posterior torso",
            "palms/soles",
            "oral/genital",
            "unknown",
        ]

        age = rng.choice([x for x in age_choices if x != age] or age_choices)
        sex = rng.choice([x for x in sex_choices if x != sex] or sex_choices)
        site = rng.choice([x for x in site_choices if x != site] or site_choices)

        prefix = "Counterfactually modified clinical metadata"
    else:
        prefix = "Clinical metadata"

    return f"""
{prefix}:
- age: {age}
- sex: {sex}
- anatomical_site: {site}
- lesion_id: {lesion_id}
- image_id: {image_id}
""".strip()


def build_taskC_prompt(row, variant, seed=42):
    sample_id = str(row.get("image_id", row.get("isic_id", row.get("lesion_id", "unknown"))))
    rng = stable_rng(seed, f"{sample_id}:{variant}")

    if variant == "full":
        instruction = "You are given a dermoscopic lesion image and clinical metadata."
        image_needed = True
        meta_text = metadata_block(row, counterfactual=False, rng=rng)

    elif variant == "image_only":
        instruction = "You are given only a dermoscopic lesion image. No clinical metadata is provided."
        image_needed = True
        meta_text = "Clinical metadata: not provided."

    elif variant == "metadata_only":
        instruction = "You are given only clinical metadata. No dermoscopic image is provided."
        image_needed = False
        meta_text = metadata_block(row, counterfactual=False, rng=rng)

    elif variant == "metadata_counterfactual":
        instruction = "You are given the original dermoscopic lesion image but counterfactually modified clinical metadata."
        image_needed = True
        meta_text = metadata_block(row, counterfactual=True, rng=rng)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    image_note = "Image: provided.\n\n" if image_needed else "Image: not provided.\n\n"

    prompt = f"""
Task:
Classify the skin lesion into exactly one diagnosis.

Allowed labels:
- MEL
- NV
- BCC
- BKL

Label meanings:
- MEL: melanoma
- NV: melanocytic nevus
- BCC: basal cell carcinoma
- BKL: benign keratosis-like lesion

{instruction}

{image_note}{meta_text}

This is for benchmark evaluation only, not clinical use.

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
- image_reliance and metadata_reliance must each be between 0 and 1.
- If image is not provided, image_reliance must be 0.
- If metadata is not provided, metadata_reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()

    return prompt, image_needed


def normalize_modalities(pred, variant):
    if variant == "image_only":
        pred["metadata_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m == "image"]

    elif variant == "metadata_only":
        pred["image_reliance"] = 0.0
        pred["used_modalities"] = [m for m in pred["used_modalities"] if m == "metadata"]

    return pred


def load_fixed_taskC_split(
    csv_path,
    label_col=None,
    image_col=None,
    image_root=None,
    limit=None,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing fixed Task C CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    label_col = resolve_column(
        df,
        preferred=label_col,
        candidates=["dx", "label", "diagnosis", "target", "true_label"],
        required=True,
    )

    image_col = resolve_column(
        df,
        preferred=image_col,
        candidates=["image_path", "path", "filepath", "file_path", "jpg_path", "image"],
        required=True,
    )

    df["_label"] = df[label_col].apply(normalize_label)
    df = df[df["_label"].isin(CLASSES)].copy()

    df["_image_path"] = df[image_col].apply(
        lambda p: os.path.join(image_root, str(p))
        if image_root and not os.path.isabs(str(p))
        else str(p)
    )

    df = df[df["_image_path"].notna()].copy()
    df = df[df["_image_path"].apply(lambda p: isinstance(p, str) and len(p.strip()) > 0)].copy()
    df = df.reset_index(drop=True)

    if limit is not None and limit > 0:
        df = df.head(limit).copy().reset_index(drop=True)

    print("Loaded fixed Task C split:", csv_path)
    print("label_col:", label_col)
    print("image_col:", image_col)
    print("image_root:", image_root)
    print("Class counts:")
    print(df["_label"].value_counts().reindex(CLASSES).to_string())
    print("n =", len(df))

    return df, label_col, image_col


def load_existing_done(results_path):
    if not os.path.exists(results_path):
        return pd.DataFrame(), set()

    res = load_jsonl_safe(results_path)
    if len(res) == 0:
        return pd.DataFrame(), set()

    if not {"sample_id", "variant"}.issubset(set(res.columns)):
        return pd.DataFrame(), set()

    res = res.drop_duplicates(
        subset=["sample_id", "variant"],
        keep="last",
    ).copy()

    done = set(zip(res["sample_id"], res["variant"]))
    return res, done


def build_deepseek_conversation(row, variant, seed=42):
    prompt, image_needed = build_taskC_prompt(row, variant, seed=seed)

    if image_needed:
        image_path = row.get("_image_path")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [image_path],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]
    else:
        image_path = None
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

    return conversation, image_path


def call_deepseek_taskC_once(
    model,
    processor,
    tokenizer,
    row,
    variant,
    max_new_tokens=256,
    seed=42,
    use_cache=False,
):
    conversation, image_path = build_deepseek_conversation(
        row=row,
        variant=variant,
        seed=seed,
    )

    pil_images = load_pil_images(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    )

    target_device = next(model.parameters()).device
    prepare_inputs = prepare_inputs.to(target_device)

    with torch.inference_mode():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )

    response_text = tokenizer.decode(
        outputs[0].cpu().tolist(),
        skip_special_tokens=True,
    ).strip()

    pred = parse_prediction_response(response_text)
    pred = normalize_modalities(pred, variant)
    pred["raw_output"] = response_text
    return pred


def call_deepseek_taskC(
    model,
    processor,
    tokenizer,
    row,
    variant,
    max_retries=4,
    max_new_tokens=256,
    seed=42,
    use_cache=False,
):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_deepseek_taskC_once(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                row=row,
                variant=variant,
                max_new_tokens=max_new_tokens,
                seed=seed,
                use_cache=use_cache,
            )
        except Exception as e:
            last_error = e
            wait = min(60, 2 ** (attempt - 1) * 3)
            print(
                f"\n[Retry] sample={row.get('_sample_id')} "
                f"variant={variant} "
                f"attempt={attempt}/{max_retries} "
                f"error={repr(e)} "
                f"wait={wait}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(wait)

    raise last_error


def compute_taskC_metrics(results_path, out_dir, run_prefix, suffix=""):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        print("[Metrics] No valid rows.")
        return None

    res = res.drop_duplicates(subset=["sample_id", "variant"], keep="last").copy()

    rows = []
    report_lines = []

    metrics_csv = os.path.join(out_dir, f"{run_prefix}_metrics{suffix}.csv")
    reliance_csv = os.path.join(out_dir, f"{run_prefix}_reliance{suffix}.csv")
    report_txt = os.path.join(out_dir, f"{run_prefix}_report{suffix}.txt")

    confusion_dir = os.path.join(out_dir, f"confusion_matrices_{run_prefix}{suffix}")
    os.makedirs(confusion_dir, exist_ok=True)

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
            "mean_image_reliance_self_report": float(g["image_reliance"].mean()),
            "mean_metadata_reliance_self_report": float(g["metadata_reliance"].mean()),
        })

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
        cm_df.to_csv(os.path.join(confusion_dir, f"{variant}_confusion.csv"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_csv, index=False)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = metrics.set_index("variant")["balanced_accuracy"].to_dict() if len(metrics) else {}

    full = score.get("full")
    image_only = score.get("image_only")
    metadata_only = score.get("metadata_only")
    cf = score.get("metadata_counterfactual")

    unimodal_scores = [x for x in [image_only, metadata_only] if x is not None]

    reliance = {
        "full_balanced_accuracy": full,
        "image_only_balanced_accuracy": image_only,
        "metadata_only_balanced_accuracy": metadata_only,
        "metadata_counterfactual_balanced_accuracy": cf,
        "integration_gain_over_best_unimodal": None if full is None or not unimodal_scores else full - max(unimodal_scores),
        "image_necessity_full_minus_metadata_only": None if full is None or metadata_only is None else full - metadata_only,
        "metadata_necessity_full_minus_image_only": None if full is None or image_only is None else full - image_only,
        "metadata_counterfactual_sensitivity_full_minus_cf": None if full is None or cf is None else full - cf,
    }

    wide = res.pivot_table(
        index="sample_id",
        columns="variant",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    for other in ["image_only", "metadata_only", "metadata_counterfactual"]:
        key = f"full_vs_{other}_prediction_change_rate"
        if "full" in wide.columns and other in wide.columns:
            valid = wide.dropna(subset=["full", other]).copy()
            reliance[key] = float((valid["full"] != valid[other]).mean()) if len(valid) else None
        else:
            reliance[key] = None

    reliance_df = pd.DataFrame([reliance])
    reliance_df.to_csv(reliance_csv, index=False)

    print(f"\n=== Task C Metrics {suffix or 'final/current'} ===")
    print(metrics.to_string(index=False))

    print(f"\n=== Task C Reliance summary {suffix or 'final/current'} ===")
    print(reliance_df.to_string(index=False))

    print("\nSaved:")
    print("metrics :", metrics_csv)
    print("reliance:", reliance_csv)
    print("report  :", report_txt)
    print("confusion:", confusion_dir)

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

    if expected > 0:
        print("completion rate:", round(completed / expected * 100, 2), "%")

    if done:
        done_df = pd.DataFrame(list(done), columns=["sample_id", "variant"])
        print("\nCompleted by variant:")
        print(
            done_df["variant"]
            .value_counts()
            .reindex(VARIANTS)
            .fillna(0)
            .astype(int)
            .to_string()
        )


def print_saved_status(results_jsonl, failed_jsonl, run_prefix):
    print("\n==============================")
    print(" Task C Saved / Resume Status")
    print("==============================")
    print("RUN_PREFIX:", run_prefix)
    print("RESULTS_JSONL:", results_jsonl)
    print("FAILED_JSONL:", failed_jsonl)

    if os.path.exists(results_jsonl):
        print("\n[results file exists]")
        print("size bytes:", os.path.getsize(results_jsonl))

        if os.path.getsize(results_jsonl) > 0:
            res = load_jsonl_safe(results_jsonl)
            res_dedup = res.drop_duplicates(
                subset=["sample_id", "variant"],
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

    if os.path.exists(failed_jsonl) and os.path.getsize(failed_jsonl) > 0:
        fail = load_jsonl_safe(failed_jsonl)
        print("\nFailed rows:", len(fail))
        print(fail.tail().to_string(index=False))
    else:
        print("\nNo failed file or failed file is empty.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixed_csv",
        type=str,
        required=True,
        help="Fixed Task C CSV. The row order is preserved; no shuffle is performed.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root directory for relative image paths.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default=None,
        help="Optional label column. Defaults to auto-detect: dx, label, diagnosis, target, true_label.",
    )
    parser.add_argument(
        "--image_col",
        type=str,
        default=None,
        help="Optional image path column. Defaults to auto-detect: image_path, path, filepath, file_path, jpg_path, image.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/taskC_deepseek_vl2",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/deepseek-vl2",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="deepseek_vl2",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume_results",
        type=str,
        default=None,
        help="Existing results JSONL to resume from.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--interim_every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit. Uses the first N rows of the fixed CSV.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
    )

    args = parser.parse_args()

    print("torch:", torch.__version__)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"taskC_{args.model_tag}_fixed_{run_id}"

    if args.resume_results is not None:
        results_jsonl = args.resume_results
        run_prefix = os.path.basename(results_jsonl).replace("_results.jsonl", "")
    else:
        results_jsonl = os.path.join(args.out_dir, f"{run_prefix}_results.jsonl")

    failed_jsonl = os.path.join(args.out_dir, f"{run_prefix}_failed.jsonl")

    print("=" * 90)
    print("TASK          : Task C BioMM-Eval fixed split")
    print("MODEL_ID      :", args.model_id)
    print("RUN_PREFIX    :", run_prefix)
    print("FIXED_CSV     :", args.fixed_csv)
    print("IMAGE_ROOT    :", args.image_root)
    print("RESULTS_JSONL :", results_jsonl)
    print("FAILED_JSONL  :", failed_jsonl)
    print("=" * 90)

    print_saved_status(results_jsonl, failed_jsonl, run_prefix)

    df, label_col, image_col = load_fixed_taskC_split(
        csv_path=args.fixed_csv,
        label_col=args.label_col,
        image_col=args.image_col,
        image_root=args.image_root,
        limit=args.limit,
    )

    id_col = resolve_column(
        df,
        preferred=None,
        candidates=["image_id", "isic_id", "lesion_id", "sample_id"],
        required=False,
    )

    if id_col is None:
        df["_sample_id"] = df.index.astype(str)
    else:
        df["_sample_id"] = df[id_col].astype(str)

    print("\nTask C sample preview:")
    preview_cols = [
        "_sample_id",
        "_label",
        "_image_path",
        "age_approx",
        "sex",
        "anatom_site_general",
        "lesion_id",
        "image_id",
        "isic_id",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))

    resume = not args.no_resume
    existing_res, done = (
        load_existing_done(results_jsonl) if resume else (pd.DataFrame(), set())
    )

    summarize_progress(df, done)

    print("\nLoading DeepSeek-VL2 processor...")
    processor = DeepseekVLV2Processor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    print("Loading DeepSeek-VL2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    target_device = next(model.parameters()).device

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    print("target device:", target_device)

    print("\nComputing current/interim metrics before run...")
    compute_taskC_metrics(
        results_path=results_jsonl,
        out_dir=args.out_dir,
        run_prefix=run_prefix,
        suffix="_interim",
    )

    todo = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["_sample_id"], variant)

            if not resume or key not in done:
                todo.append((row_dict, variant))

    if len(todo) == 0:
        print("\nNothing left to run. Computing final Task C metrics only.")
        compute_taskC_metrics(
            results_path=results_jsonl,
            out_dir=args.out_dir,
            run_prefix=run_prefix,
        )
        print("\nTASK C SUCCESS")
        return

    print("\nRunning Task C remaining predictions sequentially on GPU...")
    print("Remaining tasks:", len(todo))

    completed_since_metric = 0
    start_time = time.time()

    for idx, (row_dict, variant) in enumerate(tqdm(todo, total=len(todo)), 1):
        try:
            pred = call_deepseek_taskC(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                row=row_dict,
                variant=variant,
                max_retries=args.max_retries,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                use_cache=args.use_cache,
            )

            out = {
                "model_id": args.model_id,
                "sample_id": row_dict["_sample_id"],
                "variant": variant,
                "true_label": row_dict["_label"],
                "label_raw": row_dict.get(label_col),
                "image_path": row_dict.get("_image_path"),
                "age_approx": None if pd.isna(row_dict.get("age_approx", np.nan)) else row_dict.get("age_approx"),
                "sex": None if pd.isna(row_dict.get("sex", np.nan)) else row_dict.get("sex"),
                "anatom_site_general": None if pd.isna(row_dict.get("anatom_site_general", np.nan)) else row_dict.get("anatom_site_general"),
                "lesion_id": None if pd.isna(row_dict.get("lesion_id", np.nan)) else row_dict.get("lesion_id"),
                **pred,
            }

            append_jsonl(results_jsonl, out)
            done.add((row_dict["_sample_id"], variant))

        except Exception as e:
            fail = {
                "model_id": args.model_id,
                "sample_id": row_dict.get("_sample_id", None),
                "variant": variant,
                "true_label": row_dict.get("_label", None),
                "image_path": row_dict.get("_image_path", None),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

            append_jsonl(failed_jsonl, fail)

            print(
                "\nFAILED:",
                fail.get("sample_id"),
                fail.get("variant"),
                fail.get("error"),
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"\n[Task C] processed this run: {idx}/{len(todo)} "
                f"| elapsed={elapsed / 60:.1f} min"
            )

        if args.interim_every > 0 and completed_since_metric >= args.interim_every:
            completed_since_metric = 0
            print("\n[Task C Interim] computing metrics...")
            compute_taskC_metrics(
                results_path=results_jsonl,
                out_dir=args.out_dir,
                run_prefix=run_prefix,
                suffix="_interim",
            )
            summarize_progress(df, done)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nComputing final Task C metrics...")
    compute_taskC_metrics(
        results_path=results_jsonl,
        out_dir=args.out_dir,
        run_prefix=run_prefix,
    )

    print("\nTASK C SUCCESS")
    print("results :", results_jsonl)
    print("failed  :", failed_jsonl)


if __name__ == "__main__":
    main()
