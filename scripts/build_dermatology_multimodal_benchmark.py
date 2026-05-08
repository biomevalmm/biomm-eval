#!/usr/bin/env python3
"""
Build and optionally evaluate the BioMM-Eval dermatology multimodal benchmark.

Task C:
  Dermoscopic image + clinical metadata lesion diagnosis

Classes:
  - MEL: melanoma
  - NV: melanocytic nevus
  - BCC: basal cell carcinoma
  - BKL: benign keratosis-like lesion

Main outputs:
  isic_dermatology_multimodal_benchmark.csv
  taskC_test_full.csv
  taskC_test_balanced.csv
  taskC_<model>_<mode>_results_<run_id>.jsonl
  taskC_<model>_<mode>_metrics_<run_id>.csv
  taskC_<model>_<mode>_reliance_<run_id>.csv
"""

import argparse
import base64
import json
import os
import random
import shutil
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


ISIC_2019_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_Input.zip"
)
ISIC_2019_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_GroundTruth.csv"
)
ISIC_2019_META_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_Metadata.csv"
)

CLASSES = ["MEL", "NV", "BCC", "BKL"]

VARIANTS = [
    "full",
    "image_only",
    "metadata_only",
    "metadata_counterfactual",
]

DEFAULT_SEED = 42


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Using cached file: {out_path}")
        return out_path

    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    return out_path


def normalize_text(x: object) -> str | float:
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x.lower() in {"", "unknown", "nan", "na", "n/a", "null"}:
        return np.nan

    return x


def clean_val(x: object) -> str:
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    if x.lower() in {"", "unknown", "nan", "na", "n/a", "null"}:
        return "unknown"

    return x


def find_image_path(image_id: str, image_dir: Path) -> str | float:
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return str(candidate)

    return np.nan


def extract_images(zip_path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)

    n_existing = len(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if n_existing >= 1000:
        print(f"Images already extracted: {n_existing}")
        return

    print("Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(image_dir)

    for subdir in [p for p in image_dir.iterdir() if p.is_dir()]:
        image_files = [
            p
            for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if image_files:
            print(f"Flattening nested image folder: {subdir}")

        for src in image_files:
            dst = image_dir / src.name
            if not dst.exists():
                shutil.move(str(src), str(dst))


def build_isic_master_table(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    meta_path = download_file(
        ISIC_2019_META_URL,
        args.raw_dir / "ISIC_2019_Training_Metadata.csv",
    )
    labels_path = download_file(
        ISIC_2019_LABELS_URL,
        args.raw_dir / "ISIC_2019_Training_GroundTruth.csv",
    )

    if args.download_images:
        zip_path = download_file(
            ISIC_2019_IMAGES_URL,
            args.raw_dir / "ISIC_2019_Training_Input.zip",
        )
        extract_images(zip_path, args.image_dir)

    meta_df = pd.read_csv(meta_path)
    labels_df = pd.read_csv(labels_path)

    meta_df = meta_df.rename(
        columns={
            "image": "image_id",
            "age_approx": "age_approx",
            "sex": "sex",
            "anatom_site_general": "anatom_site_general",
            "lesion_id": "common_lesion_id",
        }
    )

    for col in meta_df.columns:
        if col != "age_approx":
            meta_df[col] = meta_df[col].map(normalize_text)

    meta_df["age_approx"] = pd.to_numeric(meta_df["age_approx"], errors="coerce")

    labels_df = labels_df.rename(columns={"image": "image_id"})

    diagnosis_cols = [c for c in labels_df.columns if c != "image_id"]
    if not diagnosis_cols:
        raise RuntimeError("No diagnosis columns found in ISIC label file.")

    labels_df["diagnosis"] = labels_df[diagnosis_cols].idxmax(axis=1)
    labels_df["diagnosis_score"] = labels_df[diagnosis_cols].max(axis=1)

    malignant_set = {"MEL", "BCC", "SCC"}
    labels_df["is_malignant"] = labels_df["diagnosis"].isin(malignant_set).astype(int)

    labels_long = labels_df[
        ["image_id", "diagnosis", "diagnosis_score", "is_malignant"]
    ].copy()

    master = meta_df.merge(labels_long, on="image_id", how="inner")

    if args.download_images:
        master["image_path"] = master["image_id"].map(
            lambda image_id: find_image_path(image_id, args.image_dir)
        )
        master["has_image_file"] = master["image_path"].notna().astype(int)
    else:
        master["image_path"] = np.nan
        master["has_image_file"] = 0

    metadata_cols = [
        c
        for c in [
            "age_approx",
            "sex",
            "anatom_site_general",
            "common_lesion_id",
        ]
        if c in master.columns
    ]

    master["n_metadata_fields_present"] = master[metadata_cols].notna().sum(axis=1)

    master = master.sort_values("image_id").reset_index(drop=True)

    meta_df.to_csv(args.output_dir / "isic_2019_metadata_raw.csv", index=False)
    labels_df.to_csv(args.output_dir / "isic_2019_labels_raw.csv", index=False)

    master_path = args.output_dir / args.output_filename
    master.to_csv(master_path, index=False)

    subset = master[master["diagnosis"].isin(CLASSES)].copy()
    subset.to_csv(
        args.output_dir / "isic_dermatology_four_class_subset.csv",
        index=False,
    )

    summary = {
        "n_rows_master": len(master),
        "n_rows_four_class_subset": len(subset),
        "n_unique_diagnoses": master["diagnosis"].nunique(),
        "n_with_image_files": int(master["has_image_file"].sum()),
        "metadata_completeness_mean": float(master["n_metadata_fields_present"].mean()),
    }

    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)

    print("\nSaved benchmark table:", master_path)
    print("Master shape:", master.shape)
    print("Four-class subset shape:", subset.shape)
    print(master["diagnosis"].value_counts().to_string())

    return master


def create_test_splits(args: argparse.Namespace, master: pd.DataFrame | None = None) -> None:
    if master is None:
        master_path = args.output_dir / args.output_filename
        if not master_path.exists():
            raise FileNotFoundError(f"Missing benchmark table: {master_path}")
        master = pd.read_csv(master_path)

    required = [
        "image_id",
        "image_path",
        "has_image_file",
        "diagnosis",
        "age_approx",
        "sex",
        "anatom_site_general",
    ]

    missing = [c for c in required if c not in master.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = master[
        (master["has_image_file"] == 1)
        & master["image_path"].notna()
        & master["diagnosis"].isin(CLASSES)
    ].copy()

    df = df[
        df["image_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    ].copy()

    if df.empty:
        raise RuntimeError("No usable Task C rows with existing image files.")

    full = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    full["split_id"] = "taskC_full_v1"

    full_path = args.output_dir / "taskC_test_full.csv"
    full.to_csv(full_path, index=False)

    class_counts = full["diagnosis"].value_counts()
    n_balanced = int(class_counts.loc[CLASSES].min())

    parts = []
    for cls in CLASSES:
        sub = full[full["diagnosis"] == cls].copy()
        parts.append(sub.sample(n=n_balanced, random_state=args.seed))

    balanced = (
        pd.concat(parts, axis=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )
    balanced["split_id"] = "taskC_balanced_v1"

    balanced_path = args.output_dir / "taskC_test_balanced.csv"
    balanced.to_csv(balanced_path, index=False)

    print("\nSaved full split:", full_path, "n=", len(full))
    print(full["diagnosis"].value_counts().to_string())

    print("\nSaved balanced split:", balanced_path, "n=", len(balanced))
    print(balanced["diagnosis"].value_counts().to_string())


def image_to_data_url(path: str, max_size: int = 640, quality: int = 85) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))

    tmp_path = Path("/tmp/biomm_taskc_tmp.jpg")
    image.save(tmp_path, format="JPEG", quality=quality)

    with tmp_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64}"


def make_counterfactual_metadata(row: pd.Series, rng: random.Random) -> dict[str, Any]:
    possible_sites = [
        "anterior torso",
        "posterior torso",
        "lower extremity",
        "upper extremity",
        "head/neck",
        "palms/soles",
        "oral/genital",
    ]

    possible_sex = ["male", "female"]

    age = row.get("age_approx")
    sex = clean_val(row.get("sex"))
    site = clean_val(row.get("anatom_site_general"))

    new_sites = [s for s in possible_sites if s != site]
    new_site = rng.choice(new_sites)

    if sex == "male":
        new_sex = "female"
    elif sex == "female":
        new_sex = "male"
    else:
        new_sex = rng.choice(possible_sex)

    try:
        age_num = float(age)
        new_age = max(5, min(95, 95 - age_num))
    except Exception:
        new_age = "unknown"

    return {
        "age_approx": new_age,
        "sex": new_sex,
        "anatom_site_general": new_site,
    }


def metadata_block(
    row: pd.Series,
    counterfactual: bool,
    rng: random.Random,
) -> tuple[str, dict[str, Any] | None]:
    if counterfactual:
        meta = make_counterfactual_metadata(row, rng)
        prefix = "Counterfactually modified clinical metadata"
    else:
        meta = {
            "age_approx": clean_val(row.get("age_approx")),
            "sex": clean_val(row.get("sex")),
            "anatom_site_general": clean_val(row.get("anatom_site_general")),
        }
        prefix = "Clinical metadata"

    text = f"""
{prefix}:
- approximate age: {meta["age_approx"]}
- sex: {meta["sex"]}
- anatomical site: {meta["anatom_site_general"]}
""".strip()

    return text, meta


def build_prompt(
    row: pd.Series,
    variant: str,
    rng: random.Random,
) -> tuple[str, bool, dict[str, Any] | None]:
    if variant == "full":
        instruction = "You are given a dermoscopic skin lesion image and clinical metadata."
        meta_text, meta_used = metadata_block(row, counterfactual=False, rng=rng)
        include_image = True

    elif variant == "image_only":
        instruction = (
            "You are given only a dermoscopic skin lesion image. "
            "No clinical metadata is provided."
        )
        meta_text = "Clinical metadata: not provided."
        meta_used = None
        include_image = True

    elif variant == "metadata_only":
        instruction = "You are given only clinical metadata. No image is provided."
        meta_text, meta_used = metadata_block(row, counterfactual=False, rng=rng)
        include_image = False

    elif variant == "metadata_counterfactual":
        instruction = (
            "You are given the same dermoscopic image but counterfactually "
            "modified clinical metadata."
        )
        meta_text, meta_used = metadata_block(row, counterfactual=True, rng=rng)
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

Output requirements:
- Return only valid JSON.
- prediction must be exactly one of: MEL, NV, BCC, BKL.
- confidence must be between 0 and 1.
- image_reliance and metadata_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance should be 0.

JSON fields:
prediction
confidence
used_modalities
image_reliance
metadata_reliance
rationale_short
""".strip()

    return prompt, include_image, meta_used


def safe_float01(x: object, default: float = 0.0) -> float:
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return default


def parse_prediction_response(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        obj = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise RuntimeError(f"Could not parse JSON from response: {text[:500]}")

    prediction = obj.get("prediction")
    if prediction not in CLASSES:
        raise RuntimeError(f"Invalid prediction: {prediction}")

    used_modalities = obj.get("used_modalities", [])
    if not isinstance(used_modalities, list):
        used_modalities = []

    used_modalities = [
        x for x in used_modalities if x in {"image", "metadata"}
    ]

    return {
        "prediction": prediction,
        "confidence": safe_float01(obj.get("confidence", 0)),
        "used_modalities": used_modalities,
        "image_reliance": safe_float01(obj.get("image_reliance", 0)),
        "metadata_reliance": safe_float01(obj.get("metadata_reliance", 0)),
        "rationale_short": str(obj.get("rationale_short", ""))[:500],
    }


def call_openai_model(
    client: Any,
    model: str,
    row: pd.Series,
    variant: str,
    rng: random.Random,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt, include_image, meta_used = build_prompt(row, variant, rng)

    content = [{"type": "input_text", "text": prompt}]

    if include_image:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(row["image_path"]),
            }
        )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "taskC_prediction",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prediction": {"type": "string", "enum": CLASSES},
                        "confidence": {"type": "number"},
                        "used_modalities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["image", "metadata"],
                            },
                        },
                        "image_reliance": {"type": "number"},
                        "metadata_reliance": {"type": "number"},
                        "rationale_short": {"type": "string"},
                    },
                    "required": [
                        "prediction",
                        "confidence",
                        "used_modalities",
                        "image_reliance",
                        "metadata_reliance",
                        "rationale_short",
                    ],
                },
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )

    out = parse_prediction_response(response.output_text)
    out["metadata_used"] = meta_used

    if variant == "image_only":
        out["metadata_reliance"] = 0.0
        out["used_modalities"] = [
            m for m in out["used_modalities"] if m != "metadata"
        ]

    if variant == "metadata_only":
        out["image_reliance"] = 0.0
        out["used_modalities"] = [
            m for m in out["used_modalities"] if m != "image"
        ]

    return out


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_done(results_path: Path) -> set[tuple[str, str]]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()

    res = pd.read_json(results_path, lines=True)
    if res.empty:
        return set()

    res = res.drop_duplicates(subset=["image_id", "variant"], keep="last")
    return set(zip(res["image_id"], res["variant"]))


def compute_metrics(
    results_path: Path,
    metrics_path: Path,
    reliance_path: Path,
    report_path: Path,
    confusion_dir: Path,
) -> None:
    if not results_path.exists() or results_path.stat().st_size == 0:
        print("[Metrics] No results yet.")
        return

    confusion_dir.mkdir(parents=True, exist_ok=True)

    res = pd.read_json(results_path, lines=True)
    res = res.drop_duplicates(subset=["image_id", "variant"], keep="last")

    rows = []
    report_lines = []

    for variant, group in res.groupby("variant"):
        group = group.dropna(subset=["prediction", "true_diagnosis"]).copy()
        group = group[group["true_diagnosis"].isin(CLASSES)]
        group = group[group["prediction"].isin(CLASSES)]

        if group.empty:
            continue

        y_true = group["true_diagnosis"]
        y_pred = group["prediction"]

        rows.append(
            {
                "variant": variant,
                "n": len(group),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "mean_confidence": float(group["confidence"].mean()),
                "mean_image_reliance_self_report": float(
                    group["image_reliance"].mean()
                ),
                "mean_metadata_reliance_self_report": float(
                    group["metadata_reliance"].mean()
                ),
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
        cm_df.to_csv(confusion_dir / f"confusion_{variant}.csv")

        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix: {variant}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(CLASSES)), CLASSES)
        plt.yticks(range(len(CLASSES)), CLASSES)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(confusion_dir / f"confusion_{variant}.png", dpi=200)
        plt.close()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    score = metrics.set_index("variant")["balanced_accuracy"].to_dict()

    full = score.get("full")
    image_only = score.get("image_only")
    metadata_only = score.get("metadata_only")
    cf = score.get("metadata_counterfactual")

    reliance = {
        "full_balanced_accuracy": full,
        "image_only_balanced_accuracy": image_only,
        "metadata_only_balanced_accuracy": metadata_only,
        "metadata_counterfactual_balanced_accuracy": cf,
        "metadata_necessity_score_full_minus_image_only": (
            None if full is None or image_only is None else full - image_only
        ),
        "image_necessity_score_full_minus_metadata_only": (
            None if full is None or metadata_only is None else full - metadata_only
        ),
        "counterfactual_sensitivity_full_minus_cf": (
            None if full is None or cf is None else full - cf
        ),
    }

    wide = res.pivot_table(
        index="image_id",
        columns="variant",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    for other in ["image_only", "metadata_only", "metadata_counterfactual"]:
        key = f"full_vs_{other}_prediction_change_rate"

        if "full" in wide.columns and other in wide.columns:
            valid = wide.dropna(subset=["full", other]).copy()
            reliance[key] = (
                float((valid["full"] != valid[other]).mean()) if len(valid) else None
            )
        else:
            reliance[key] = None

    pd.DataFrame([reliance]).to_csv(reliance_path, index=False)

    print("\n=== Task C metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== Task C reliance summary ===")
    print(pd.DataFrame([reliance]).to_string(index=False))


def run_evaluation(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing API key. Pass --api-key or set OPENAI_API_KEY."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if args.eval_mode == "full":
        split_path = args.output_dir / "taskC_test_full.csv"
    elif args.eval_mode == "balanced":
        split_path = args.output_dir / "taskC_test_balanced.csv"
    else:
        raise ValueError("--eval-mode must be full or balanced.")

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run without --skip-build first."
        )

    df = pd.read_csv(split_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    rng = random.Random(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_").replace("-", "").replace(".", "")

    eval_dir = args.output_dir / "biomm_taskC_outputs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_path = eval_dir / f"taskC_{model_tag}_{args.eval_mode}_results_{run_id}.jsonl"
    failed_path = eval_dir / f"taskC_{model_tag}_{args.eval_mode}_failed_{run_id}.jsonl"
    metrics_path = eval_dir / f"taskC_{model_tag}_{args.eval_mode}_metrics_{run_id}.csv"
    reliance_path = eval_dir / f"taskC_{model_tag}_{args.eval_mode}_reliance_{run_id}.csv"
    report_path = eval_dir / f"taskC_{model_tag}_{args.eval_mode}_report_{run_id}.txt"
    confusion_dir = eval_dir / f"confusion_matrices_{model_tag}_{args.eval_mode}_{run_id}"

    done = load_existing_done(results_path)

    print("\nRunning Task C evaluation")
    print("model:", args.model)
    print("split:", split_path)
    print("samples:", len(df))
    print("variants:", VARIANTS)
    print(df["diagnosis"].value_counts().to_string())

    total = len(df) * len(VARIANTS) - len(done)
    pbar = tqdm(total=max(0, total))

    completed_since_metric = 0

    for _, row in df.iterrows():
        for variant in VARIANTS:
            key = (row["image_id"], variant)
            if key in done:
                continue

            try:
                last_error = None
                pred = None

                for attempt in range(args.max_retries):
                    try:
                        pred = call_openai_model(
                            client=client,
                            model=args.model,
                            row=row,
                            variant=variant,
                            rng=rng,
                            max_output_tokens=args.max_output_tokens,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        wait = min(90, 2**attempt * 5)
                        print(
                            f"\n[Retry] image_id={row.get('image_id')} "
                            f"variant={variant} attempt={attempt + 1}/{args.max_retries} "
                            f"wait={wait}s error={repr(exc)}"
                        )
                        time.sleep(wait)

                if pred is None:
                    raise last_error

                out = {
                    "image_id": row["image_id"],
                    "variant": variant,
                    "true_diagnosis": row["diagnosis"],
                    "age_approx_original": (
                        None
                        if pd.isna(row.get("age_approx"))
                        else float(row.get("age_approx"))
                    ),
                    "sex_original": (
                        None if pd.isna(row.get("sex")) else row.get("sex")
                    ),
                    "anatom_site_original": (
                        None
                        if pd.isna(row.get("anatom_site_general"))
                        else row.get("anatom_site_general")
                    ),
                    **pred,
                }

                append_jsonl(results_path, out)
                done.add(key)
                pbar.update(1)
                completed_since_metric += 1

                time.sleep(args.sleep_seconds)

                if completed_since_metric >= args.metric_interval:
                    completed_since_metric = 0
                    compute_metrics(
                        results_path,
                        metrics_path.with_name(metrics_path.stem + "_interim.csv"),
                        reliance_path.with_name(reliance_path.stem + "_interim.csv"),
                        report_path.with_name(report_path.stem + "_interim.txt"),
                        confusion_dir,
                    )

            except Exception as exc:
                append_jsonl(
                    failed_path,
                    {
                        "image_id": row.get("image_id"),
                        "variant": variant,
                        "true_diagnosis": row.get("diagnosis"),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print("\nFAILED:", row.get("image_id"), variant, repr(exc))
                pbar.update(1)
                time.sleep(max(2.0, args.sleep_seconds))

    pbar.close()

    compute_metrics(
        results_path,
        metrics_path,
        reliance_path,
        report_path,
        confusion_dir,
    )

    print("\nSUCCESS")
    print("results:", results_path)
    print("failed:", failed_path)
    print("metrics:", metrics_path)
    print("reliance:", reliance_path)
    print("report:", report_path)
    print("confusion:", confusion_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally evaluate the BioMM-Eval Task C "
            "dermatology image-metadata benchmark."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/isic_dermatology"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("downloads/isic2019"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/isic_dermatology/images"),
    )
    parser.add_argument(
        "--output-filename",
        default="isic_dermatology_multimodal_benchmark.csv",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download and extract the ISIC image zip.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset construction and only run split/eval steps.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run model evaluation after building splits.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["full", "balanced"],
        default="balanced",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
    )
    parser.add_argument(
        "--api-key",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for debugging evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=700,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--metric-interval",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    master = None

    if not args.skip_build:
        master = build_isic_master_table(args)

    create_test_splits(args, master=master)

    if args.run_eval:
        run_evaluation(args)


if __name__ == "__main__":
    main()
