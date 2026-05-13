import os, re, json, time, random, traceback, argparse
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
    classification_report,
)


CLASSES = [
    "Staphylococcus aureus",
    "Cutibacterium acnes",
]

VARIANTS = [
    "full",
    "sequence_only",
    "metadata_only",
    "metadata_counterfactual",
    "sequence_perturbed",
]


class TaskBPrediction(BaseModel):
    prediction: Literal["Staphylococcus aureus", "Cutibacterium acnes"]
    confidence: float = Field(ge=0, le=1)
    used_modalities: List[Literal["sequence", "metadata"]]
    sequence_reliance: float = Field(ge=0, le=1)
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

    if x.lower() in [
        "",
        "nan",
        "none",
        "null",
        "not collected",
        "missing",
        "not provided",
        "unknown",
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


def clean_phi_response(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<nothink>", "").replace("</nothink>", "")
    text = text.replace("<|im_start|>", "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|im_sep|>", "")
    text = text.replace("<|end|>", "")
    text = text.replace("</s>", "")

    return text.strip()


def extract_json(text: str):
    if text is None:
        raise RuntimeError("Empty response text")

    text = clean_phi_response(text)

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
    text = clean_phi_response(response_text)
    upper = text.upper()

    if "STAPHYLOCOCCUS AUREUS" in upper or "S. AUREUS" in upper or "SAUREUS" in upper:
        pred = "Staphylococcus aureus"
    elif (
        "CUTIBACTERIUM ACNES" in upper
        or "PROPIONIBACTERIUM ACNES" in upper
        or "C. ACNES" in upper
        or "CACNES" in upper
    ):
        pred = "Cutibacterium acnes"
    else:
        raise RuntimeError(f"Could not find Task B prediction in response: {text[:500]}")

    return {
        "prediction": pred,
        "confidence": 0.5,
        "used_modalities": ["sequence", "metadata"],
        "sequence_reliance": 0.5,
        "metadata_reliance": 0.5,
        "rationale_short": text[:500],
    }


def parse_prediction_response(response_text: str) -> Dict[str, Any]:
    text = clean_phi_response(response_text)

    try:
        obj = extract_json(text)

        bad_pred_values = {
            "Staphylococcus aureus or Cutibacterium acnes",
            "Staphylococcus aureus/Cutibacterium acnes",
            "S. aureus or C. acnes",
        }

        if str(obj.get("prediction", "")).strip() in bad_pred_values:
            return TaskBPrediction.model_validate(
                fallback_extract_prediction(text)
            ).model_dump()

    except Exception:
        return TaskBPrediction.model_validate(
            fallback_extract_prediction(text)
        ).model_dump()

    pred_raw = str(obj.get("prediction", "")).strip()

    if pred_raw in ["S. aureus", "Staph aureus", "SA", "saureus"]:
        obj["prediction"] = "Staphylococcus aureus"
    elif pred_raw in ["C. acnes", "P. acnes", "Propionibacterium acnes", "cacnes"]:
        obj["prediction"] = "Cutibacterium acnes"

    obj["confidence"] = safe_float01(obj.get("confidence", 0.0), default=0.5)
    obj["sequence_reliance"] = safe_float01(obj.get("sequence_reliance", 0.0), default=0.5)
    obj["metadata_reliance"] = safe_float01(obj.get("metadata_reliance", 0.0), default=0.5)

    if "used_modalities" not in obj or not isinstance(obj["used_modalities"], list):
        obj["used_modalities"] = []

    obj["used_modalities"] = [
        m for m in obj["used_modalities"]
        if m in ["sequence", "metadata"]
    ]

    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]

    return TaskBPrediction.model_validate(obj).model_dump()


def sequence_context(seq, max_chars):
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


def perturb_sequence(seq_text, mutation_rate, rng):
    bases = ["A", "C", "G", "T"]
    out = []

    for c in seq_text:
        if c in bases and rng.random() < mutation_rate:
            choices = [b for b in bases if b != c]
            out.append(rng.choice(choices))
        else:
            out.append(c)

    return "".join(out)


def deterministic_rng(seed: int, key: str):
    h = abs(hash((seed, key))) % (2**32)
    return random.Random(h)


def metadata_block(row, counterfactual=False, rng=None):
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
        if rng is None:
            rng = random

        host_choices = ["Homo sapiens", "human", "environment", "mouse", "unknown"]
        disease_choices = [
            "acne",
            "skin infection",
            "wound infection",
            "healthy skin",
            "bloodstream infection",
            "unknown",
        ]
        source_choices = [
            "skin",
            "acne lesion",
            "wound",
            "blood",
            "nasal swab",
            "environment",
            "unknown",
        ]
        body_choices = [
            "skin",
            "face",
            "back",
            "wound",
            "blood",
            "nasal cavity",
            "unknown",
        ]

        host = rng.choice([x for x in host_choices if x != host] or host_choices)
        host_disease = rng.choice([x for x in disease_choices if x != host_disease] or disease_choices)
        isolation_source = rng.choice([x for x in source_choices if x != isolation_source] or source_choices)
        body_site = rng.choice([x for x in body_choices if x != body_site] or body_choices)

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


def build_taskB_prompt(row, variant, max_sequence_chars, perturb_rate, seed):
    accession = str(row.get("assembly_accession", "unknown"))
    rng = deterministic_rng(seed, f"{accession}:{variant}")

    seq_ctx, full_len, used_len = sequence_context(
        row.get("joined_sequence", ""),
        max_sequence_chars,
    )

    if variant == "full":
        instruction = "You are given long genome DNA sequence segments and biological metadata."
        blocks = [
            metadata_block(row, counterfactual=False, rng=rng),
            f"""
DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- sampling_strategy: beginning + middle + end segments

{seq_ctx}
""".strip(),
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
""".strip(),
        ]

    elif variant == "metadata_only":
        instruction = "You are given only biological metadata. No DNA sequence is provided."
        blocks = [
            metadata_block(row, counterfactual=False, rng=rng),
        ]

    elif variant == "metadata_counterfactual":
        instruction = "You are given original long genome DNA sequence segments but counterfactually modified metadata."
        blocks = [
            metadata_block(row, counterfactual=True, rng=rng),
            f"""
DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- sampling_strategy: beginning + middle + end segments

{seq_ctx}
""".strip(),
        ]

    elif variant == "sequence_perturbed":
        instruction = "You are given perturbed long genome DNA sequence segments and original biological metadata."
        perturbed = perturb_sequence(seq_ctx, mutation_rate=perturb_rate, rng=rng)

        blocks = [
            metadata_block(row, counterfactual=False, rng=rng),
            f"""
Perturbed DNA sequence context:
- original_joined_sequence_length: {full_len}
- sequence_characters_provided: {used_len}
- perturbation_rate: {perturb_rate}
- sampling_strategy: beginning + middle + end segments

{perturbed}
""".strip(),
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

This is for benchmark evaluation only, not clinical use.

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
- If sequence is not provided, sequence_reliance must be 0.
- If metadata is not provided, metadata_reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()

    return prompt, full_len, used_len


def load_fixed_taskB_split(seq_csv, limit=None):
    if not os.path.exists(seq_csv):
        raise FileNotFoundError(f"Missing file: {seq_csv}")

    df = pd.read_csv(seq_csv)

    required = [
        "assembly_accession",
        "organism_name",
        "joined_sequence",
        "joined_seq_len",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["label"] = df["organism_name"].apply(normalize_organism_name)
    df = df[df["label"].isin(CLASSES)].copy()

    df = df[df["joined_sequence"].notna()].copy()
    df["joined_seq_len_numeric"] = pd.to_numeric(df["joined_seq_len"], errors="coerce")
    df = df[df["joined_seq_len_numeric"] > 1000].copy()

    # IMPORTANT:
    # Do not shuffle here.
    # The CSV is treated as the fixed benchmark subset/split, same style as Task A.
    df = df.reset_index(drop=True)

    if limit is not None and limit > 0:
        df = df.head(limit).copy().reset_index(drop=True)

    print("Loaded fixed Task B split:", seq_csv)
    print("Class counts:")
    print(df["label"].value_counts().reindex(CLASSES).to_string())
    print("n =", len(df))

    return df


def load_existing_done(results_path):
    if not os.path.exists(results_path):
        return pd.DataFrame(), set()

    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        return pd.DataFrame(), set()

    if not {"assembly_accession", "variant"}.issubset(set(res.columns)):
        return pd.DataFrame(), set()

    res = res.drop_duplicates(
        subset=["assembly_accession", "variant"],
        keep="last",
    ).copy()

    done = set(zip(res["assembly_accession"], res["variant"]))

    return res, done


def normalize_modalities(pred, variant):
    if variant == "metadata_only":
        pred["sequence_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m == "metadata"
        ]

    elif variant == "sequence_only":
        pred["metadata_reliance"] = 0.0
        pred["used_modalities"] = [
            m for m in pred["used_modalities"] if m == "sequence"
        ]

    return pred


def build_phi_prompt(prompt):
    system_prompt = (
        "You are Phi, a biomedical genome classification evaluation assistant. "
        "Use NOTHINK mode. Output only valid JSON. Do not explain."
    )

    text = (
        "<|im_start|>system<|im_sep|>"
        + system_prompt
        + "<|im_end|>"
        + "<|im_start|>user<|im_sep|>"
        + prompt
        + "<|im_end|>"
        + "<|im_start|>assistant<|im_sep|><nothink>"
    )

    return text


def call_phi_taskB_once(
    model,
    processor,
    row,
    variant,
    target_device,
    max_new_tokens,
    max_sequence_chars,
    perturb_rate,
    seed,
    use_cache,
):
    prompt, full_len, used_len = build_taskB_prompt(
        row=row,
        variant=variant,
        max_sequence_chars=max_sequence_chars,
        perturb_rate=perturb_rate,
        seed=seed,
    )

    text = build_phi_prompt(prompt)

    inputs = processor(
        text=text,
        return_tensors="pt",
    ).to(target_device)

    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": use_cache,
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    output_text = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=False,
    ).strip()

    output_text = clean_phi_response(output_text)

    pred = parse_prediction_response(output_text)
    pred = normalize_modalities(pred, variant)
    pred["raw_output"] = output_text
    pred["full_sequence_length"] = full_len
    pred["provided_sequence_length"] = used_len

    return pred


def call_phi_taskB(
    model,
    processor,
    row,
    variant,
    target_device,
    max_retries,
    max_new_tokens,
    max_sequence_chars,
    perturb_rate,
    seed,
    use_cache,
):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_phi_taskB_once(
                model=model,
                processor=processor,
                row=row,
                variant=variant,
                target_device=target_device,
                max_new_tokens=max_new_tokens,
                max_sequence_chars=max_sequence_chars,
                perturb_rate=perturb_rate,
                seed=seed,
                use_cache=use_cache,
            )

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** (attempt - 1) * 3)

            print(
                f"\n[Retry] accession={row.get('assembly_accession')} "
                f"variant={variant} "
                f"attempt={attempt}/{max_retries} "
                f"error={repr(e)} "
                f"wait={wait}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(wait)

    raise last_error


def compute_taskB_metrics(results_path, out_dir, run_prefix, suffix=""):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        print("[Metrics] No valid rows.")
        return None

    res = res.drop_duplicates(
        subset=["assembly_accession", "variant"],
        keep="last",
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
            "mean_full_sequence_length": float(g["full_sequence_length"].mean()),
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

    metrics = pd.DataFrame(rows)

    suffix_part = suffix if suffix else ""
    metrics_path = os.path.join(out_dir, f"{run_prefix}_metrics{suffix_part}.csv")
    reliance_path = os.path.join(out_dir, f"{run_prefix}_reliance{suffix_part}.csv")
    report_path = os.path.join(out_dir, f"{run_prefix}_report{suffix_part}.txt")

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
        "sequence_robustness_drop_full_minus_perturbed": None if full is None or pert is None else full - pert,
    }

    wide = res.pivot_table(
        index="assembly_accession",
        columns="variant",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    for other in [
        "sequence_only",
        "metadata_only",
        "metadata_counterfactual",
        "sequence_perturbed",
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

    if expected > 0:
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


def print_saved_status(results_jsonl, failed_jsonl, run_prefix):
    print("\n==============================")
    print(" Task B Saved / Resume Status")
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
                subset=["assembly_accession", "variant"],
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
        "--seq_csv",
        type=str,
        required=True,
        help="Fixed Task B CSV containing assembly_accession, organism_name, joined_sequence, joined_seq_len, and metadata columns.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/taskB_phi4_reasoning_vision",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-4-reasoning-vision-15B",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="phi4_reasoning_vision_15b",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional fixed run id for reproducibility/resume.",
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
        help="Disable resume behavior.",
    )
    parser.add_argument(
        "--max_sequence_chars",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--perturb_rate",
        type=float,
        default=0.01,
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
        help="Use KV cache during generation.",
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
    run_prefix = f"taskB_{args.model_tag}_fixed_{run_id}"

    if args.resume_results is not None:
        results_jsonl = args.resume_results
        run_prefix = os.path.basename(results_jsonl).replace("_results.jsonl", "")
    else:
        results_jsonl = os.path.join(args.out_dir, f"{run_prefix}_results.jsonl")

    failed_jsonl = os.path.join(args.out_dir, f"{run_prefix}_failed.jsonl")

    print("=" * 90)
    print("TASK          : Task B BioMM-Eval fixed split")
    print("MODEL_ID      :", args.model_id)
    print("RUN_PREFIX    :", run_prefix)
    print("SEQ_CSV       :", args.seq_csv)
    print("RESULTS_JSONL :", results_jsonl)
    print("FAILED_JSONL  :", failed_jsonl)
    print("=" * 90)

    print_saved_status(results_jsonl, failed_jsonl, run_prefix)

    df = load_fixed_taskB_split(
        seq_csv=args.seq_csv,
        limit=args.limit,
    )

    print("\nTask B sample preview:")
    preview_cols = [
        "assembly_accession",
        "label",
        "organism_name",
        "host",
        "host_disease",
        "isolation_source",
        "body_site",
        "joined_seq_len",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))

    resume = not args.no_resume

    existing_res, done = (
        load_existing_done(results_jsonl) if resume else (pd.DataFrame(), set())
    )

    summarize_progress(df, done)

    print("\nLoading Phi processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    print("Loading Phi model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        use_cache=args.use_cache,
    )
    model.eval()

    target_device = next(model.parameters()).device

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    print("target device:", target_device)

    print("\nComputing current/interim metrics before run...")
    compute_taskB_metrics(
        results_path=results_jsonl,
        out_dir=args.out_dir,
        run_prefix=run_prefix,
        suffix="_interim",
    )

    todo = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        for variant in VARIANTS:
            key = (row_dict["assembly_accession"], variant)

            if not resume or key not in done:
                todo.append((row_dict, variant))

    if len(todo) == 0:
        print("\nNothing left to run. Computing final Task B metrics only.")
        compute_taskB_metrics(
            results_path=results_jsonl,
            out_dir=args.out_dir,
            run_prefix=run_prefix,
        )
        print("\nTASK B SUCCESS")
        return

    print("\nRunning Task B remaining predictions sequentially on GPU...")
    print("Remaining tasks:", len(todo))

    completed_since_metric = 0
    start_time = time.time()

    for idx, (row_dict, variant) in enumerate(tqdm(todo, total=len(todo)), 1):
        try:
            pred = call_phi_taskB(
                model=model,
                processor=processor,
                row=row_dict,
                variant=variant,
                target_device=target_device,
                max_retries=args.max_retries,
                max_new_tokens=args.max_new_tokens,
                max_sequence_chars=args.max_sequence_chars,
                perturb_rate=args.perturb_rate,
                seed=args.seed,
                use_cache=args.use_cache,
            )

            joined_seq_len = row_dict.get("joined_seq_len")

            out = {
                "model_id": args.model_id,
                "assembly_accession": row_dict["assembly_accession"],
                "variant": variant,
                "true_label": row_dict["label"],
                "organism_name_raw": row_dict.get("organism_name"),
                "joined_seq_len": (
                    None
                    if pd.isna(joined_seq_len)
                    else int(float(joined_seq_len))
                ),
                "host": (
                    None
                    if pd.isna(row_dict.get("host", np.nan))
                    else row_dict.get("host")
                ),
                "host_disease": (
                    None
                    if pd.isna(row_dict.get("host_disease", np.nan))
                    else row_dict.get("host_disease")
                ),
                "isolation_source": (
                    None
                    if pd.isna(row_dict.get("isolation_source", np.nan))
                    else row_dict.get("isolation_source")
                ),
                "body_site": (
                    None
                    if pd.isna(row_dict.get("body_site", np.nan))
                    else row_dict.get("body_site")
                ),
                **pred,
            }

            append_jsonl(results_jsonl, out)
            done.add((row_dict["assembly_accession"], variant))

        except Exception as e:
            fail = {
                "model_id": args.model_id,
                "assembly_accession": row_dict.get("assembly_accession", None),
                "variant": variant,
                "true_label": row_dict.get("label", None),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

            append_jsonl(failed_jsonl, fail)

            print(
                "\nFAILED:",
                fail.get("assembly_accession"),
                fail.get("variant"),
                fail.get("error"),
            )

        completed_since_metric += 1

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"\n[Task B] processed this run: {idx}/{len(todo)} "
                f"| elapsed={elapsed / 60:.1f} min"
            )

        if args.interim_every > 0 and completed_since_metric >= args.interim_every:
            completed_since_metric = 0
            print("\n[Task B Interim] computing metrics...")
            compute_taskB_metrics(
                results_path=results_jsonl,
                out_dir=args.out_dir,
                run_prefix=run_prefix,
                suffix="_interim",
            )
            summarize_progress(df, done)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nComputing final Task B metrics...")
    compute_taskB_metrics(
        results_path=results_jsonl,
        out_dir=args.out_dir,
        run_prefix=run_prefix,
    )

    print("\nTASK B SUCCESS")
    print("results :", results_jsonl)
    print("failed  :", failed_jsonl)


if __name__ == "__main__":
    main()
