#!/usr/bin/env python3
"""
Usage:

python scripts/build_tcga_multimodal_benchmark.py \
  --output-dir data/tcga_skcm \
  --raw-dir downloads/gdc \
  --task-setting shortcut_controlled \
  --eval-mode full
"""

"""
Build the BioMM-Eval TCGA multimodal benchmark.

Default project:
  TCGA-SKCM

Main outputs:
  tcga_skcm_multimodal_benchmark.csv
  taskA_test_full.csv
  taskA_test_balanced.csv
  tcga_skcm_multimodal_benchmark_prompts.jsonl

This script:
  1. Queries TCGA/GDC clinical metadata.
  2. Downloads and summarizes open RNA expression quantification files.
  3. Queries IDC-derived pathology/image availability metadata.
  4. Merges clinical, RNA, and image-summary modalities.
  5. Builds fixed full and balanced evaluation splits.
  6. Constructs intervention-style prompt records for BioMM-Eval Task A.
"""

import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


GDC_API = "https://api.gdc.cancer.gov"

DEFAULT_PROJECT_ID = "TCGA-SKCM"
DEFAULT_IDC_COLLECTION_ID = "tcga_skcm"
DEFAULT_SEED = 42

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


def gdc_post(endpoint: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    url = f"{GDC_API}/{endpoint.lstrip('/')}"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def download_gdc_file(file_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{GDC_API}/data/{file_id}"
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.findall(r'filename="?([^"]+)"?', content_disposition)
    filename = match[0] if match else f"{file_id}.dat"

    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    with out_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return out_path


def extract_case_barcode(value: object) -> str | None:
    if pd.isna(value):
        return None

    match = re.match(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
        str(value).strip(),
        flags=re.IGNORECASE,
    )

    return match.group(1).upper() if match else None


def clean_text(value: object) -> str | float:
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() in {
        "",
        "unknown",
        "not reported",
        "not available",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
    }:
        return np.nan

    return value


def clean_prompt_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    value = str(value).strip()

    if value.lower() in {
        "",
        "unknown",
        "not reported",
        "not available",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
    }:
        return "unknown"

    return value


def read_expression_tsv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None

    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
        compression=compression,
        header=None,
    )


def fetch_clinical_table(project_id: str) -> pd.DataFrame:
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [project_id],
            },
        },
        "fields": ",".join(
            [
                "case_id",
                "submitter_id",
                "demographic.gender",
                "demographic.race",
                "demographic.ethnicity",
                "demographic.vital_status",
                "diagnoses.primary_diagnosis",
                "diagnoses.tumor_stage",
                "diagnoses.ajcc_pathologic_stage",
                "diagnoses.ajcc_clinical_stage",
                "diagnoses.age_at_diagnosis",
                "samples.sample_type",
                "samples.tissue_type",
                "samples.tumor_descriptor",
            ]
        ),
        "format": "JSON",
        "size": "5000",
    }

    cases = gdc_post("cases", payload)["data"]["hits"]

    rows = []

    for case in cases:
        diagnoses = case.get("diagnoses", [{}])
        diagnosis = diagnoses[0] if diagnoses else {}

        demographic = case.get("demographic", {})

        samples = case.get("samples", [{}])
        sample = samples[0] if samples else {}

        rows.append(
            {
                "case_barcode": extract_case_barcode(case.get("submitter_id")),
                "case_id": case.get("case_id"),
                "gender": clean_text(demographic.get("gender")),
                "race": clean_text(demographic.get("race")),
                "ethnicity": clean_text(demographic.get("ethnicity")),
                "vital_status": clean_text(demographic.get("vital_status")),
                "primary_diagnosis": clean_text(diagnosis.get("primary_diagnosis")),
                "tumor_stage": clean_text(diagnosis.get("tumor_stage")),
                "ajcc_pathologic_stage": clean_text(diagnosis.get("ajcc_pathologic_stage")),
                "ajcc_clinical_stage": clean_text(diagnosis.get("ajcc_clinical_stage")),
                "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
                "sample_type": clean_text(sample.get("sample_type")),
                "tissue_type": clean_text(sample.get("tissue_type")),
                "tumor_descriptor": clean_text(sample.get("tumor_descriptor")),
            }
        )

    clinical = pd.DataFrame(rows)
    clinical = clinical[clinical["case_barcode"].notna()].copy()
    clinical = clinical.drop_duplicates("case_barcode").reset_index(drop=True)

    return clinical


def workflow_priority(workflow: object) -> int:
    workflow = str(workflow).lower()

    score = 0

    if "star - counts" in workflow:
        score += 50
    if "htseq - counts" in workflow:
        score += 40
    if "fpkm-uq" in workflow:
        score += 30
    if "fpkm" in workflow:
        score += 20
    if "counts" in workflow:
        score += 10

    return score


def query_rna_inventory(project_id: str) -> pd.DataFrame:
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [project_id],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": ["Gene Expression Quantification"],
                    },
                },
            ],
        },
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "data_type",
                "data_format",
                "access",
                "experimental_strategy",
                "analysis.workflow_type",
                "cases.submitter_id",
                "cases.case_id",
                "file_size",
                "updated_datetime",
            ]
        ),
        "format": "JSON",
        "size": "5000",
    }

    hits = gdc_post("files", payload)["data"]["hits"]

    if not hits:
        raise RuntimeError(f"No RNA expression files found for project: {project_id}")

    inventory = pd.json_normalize(hits)
    inventory["file_size"] = pd.to_numeric(
        inventory.get("file_size"),
        errors="coerce",
    )
    inventory["workflow"] = inventory.get("analysis.workflow_type", "").fillna("")
    inventory["access"] = inventory.get("access", "").fillna("")
    inventory["file_name"] = inventory.get("file_name", "").fillna("")
    inventory["workflow_priority"] = inventory["workflow"].map(workflow_priority)
    inventory["open_priority"] = (inventory["access"].str.lower() == "open").astype(int)

    inventory = inventory.sort_values(
        ["open_priority", "workflow_priority", "file_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    usable_rows = []

    for _, row in inventory.iterrows():
        cases = row.get("cases", [])
        if isinstance(cases, list) and len(cases) > 0 and "submitter_id" in cases[0]:
            usable_rows.append(row)

    if not usable_rows:
        raise RuntimeError("RNA files found, but none had usable case submitter IDs.")

    return pd.DataFrame(usable_rows).reset_index(drop=True)


def summarize_rna_files(
    inventory: pd.DataFrame,
    raw_rna_dir: Path,
    max_rna_files: int | None,
) -> pd.DataFrame:
    rows = []
    used = 0

    for _, row in inventory.iterrows():
        if max_rna_files is not None and used >= max_rna_files:
            break

        access = str(row.get("access", "")).lower()
        if access != "open":
            continue

        case_info = row.get("cases", [])
        if not isinstance(case_info, list) or not case_info:
            continue

        case_barcode = extract_case_barcode(case_info[0].get("submitter_id"))
        if case_barcode is None:
            continue

        file_id = row["file_id"]
        file_name = row["file_name"]
        workflow = row["workflow"]

        try:
            local_path = download_gdc_file(file_id, raw_rna_dir)
            df = read_expression_tsv(local_path)

            numeric_col = None

            for col in df.columns[::-1]:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.notna().sum() > max(100, len(df) * 0.3):
                    numeric_col = col
                    break

            if numeric_col is None:
                continue

            values = pd.to_numeric(df[numeric_col], errors="coerce").dropna()

            if len(values) == 0:
                continue

            rows.append(
                {
                    "case_barcode": case_barcode,
                    "expr_workflow": workflow,
                    "expr_file_name": file_name,
                    "expr_n_genes": int(values.shape[0]),
                    "expr_mean": float(values.mean()),
                    "expr_std": float(values.std()),
                    "expr_median": float(values.median()),
                    "expr_q25": float(values.quantile(0.25)),
                    "expr_q75": float(values.quantile(0.75)),
                    "expr_min": float(values.min()),
                    "expr_max": float(values.max()),
                    "expr_nonzero_fraction": float((values > 0).mean()),
                }
            )

            used += 1

            if used % 25 == 0:
                print(f"Processed {used} RNA files")

        except Exception as exc:
            print(f"Skipped RNA file {file_name}: {exc}")

    rna = pd.DataFrame(rows)

    if rna.empty:
        return pd.DataFrame(
            columns=[
                "case_barcode",
                "expr_workflow",
                "expr_file_name",
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
        )

    numeric_cols = [
        c
        for c in rna.columns
        if c.startswith("expr_") and c not in {"expr_workflow", "expr_file_name"}
    ]

    agg = {c: "mean" for c in numeric_cols}
    agg["expr_workflow"] = "first"
    agg["expr_file_name"] = "first"

    rna = rna.groupby("case_barcode", as_index=False).agg(agg)

    return rna


def fetch_idc_image_summary(collection_id: str, install_idc_index: bool = True) -> pd.DataFrame:
    if install_idc_index:
        subprocess.run(
            ["python", "-m", "pip", "-q", "install", "-U", "idc-index"],
            check=True,
        )

    from idc_index import IDCClient

    client = IDCClient.client()
    raw = client.sql_query(f"SELECT * FROM index WHERE collection_id='{collection_id}'")
    raw["case_barcode"] = raw["PatientID"].apply(extract_case_barcode)

    agg = {}

    if "SeriesInstanceUID" in raw.columns:
        agg["SeriesInstanceUID"] = "nunique"
    if "StudyInstanceUID" in raw.columns:
        agg["StudyInstanceUID"] = "nunique"

    image = raw.groupby("case_barcode", dropna=True).agg(agg).reset_index()

    rename_map = {}

    if "SeriesInstanceUID" in image.columns:
        rename_map["SeriesInstanceUID"] = "n_series"
    if "StudyInstanceUID" in image.columns:
        rename_map["StudyInstanceUID"] = "n_studies"

    image = image.rename(columns=rename_map)

    if "Modality" in raw.columns:
        modalities = (
            raw.groupby("case_barcode")["Modality"]
            .apply(lambda s: "; ".join(sorted(set(map(str, s.dropna().tolist())))))
            .reset_index(name="image_modalities")
        )
        image = image.merge(modalities, on="case_barcode", how="left")

    return image


def normalize_task_a_label(value: object) -> str | float:
    value = clean_prompt_value(value)

    if value.lower() == "primary tumor":
        return "Primary Tumor"
    if value.lower() == "metastatic":
        return "Metastatic"

    return np.nan


def merge_modalities(
    clinical: pd.DataFrame,
    rna: pd.DataFrame,
    image: pd.DataFrame,
) -> pd.DataFrame:
    master = clinical.merge(rna, on="case_barcode", how="left")
    master = master.merge(image, on="case_barcode", how="left")

    master["has_image"] = master["n_series"].notna().astype(int) if "n_series" in master.columns else 0
    master["has_rna"] = master["expr_mean"].notna().astype(int) if "expr_mean" in master.columns else 0

    master["sample_type"] = master["sample_type"].apply(normalize_task_a_label)
    master = master[master["sample_type"].isin(CLASSES)].copy()

    preferred = [
        "case_barcode",
        "case_id",
        "sample_type",
        "primary_diagnosis",
        "tumor_stage",
        "ajcc_pathologic_stage",
        "ajcc_clinical_stage",
        "gender",
        "race",
        "ethnicity",
        "vital_status",
        "age_at_diagnosis",
        "tissue_type",
        "tumor_descriptor",
        "has_image",
        "n_series",
        "n_studies",
        "image_modalities",
        "has_rna",
        "expr_workflow",
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

    preferred = [c for c in preferred if c in master.columns]
    remaining = [c for c in master.columns if c not in preferred]

    master = master[preferred + remaining]
    master = master.sort_values("case_barcode").reset_index(drop=True)

    return master


def build_full_split(master: pd.DataFrame, require_all_modalities: bool = True) -> pd.DataFrame:
    df = master.copy()

    if require_all_modalities:
        df = df[(df["has_image"] == 1) & (df["has_rna"] == 1)].copy()

    df = df[df["sample_type"].isin(CLASSES)].copy()
    df = df.sort_values("case_barcode").reset_index(drop=True)

    return df


def build_balanced_split(full_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    parts = []

    class_counts = full_df["sample_type"].value_counts()
    n = int(class_counts.min())

    for label in CLASSES:
        sub = full_df[full_df["sample_type"] == label].copy()
        parts.append(sub.sample(n=n, random_state=seed))

    balanced = (
        pd.concat(parts, axis=0)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return balanced


def clinical_block(row: pd.Series, task_setting: str, counterfactual: bool, rng: random.Random) -> str:
    age = row.get("age_at_diagnosis", np.nan)
    gender = clean_prompt_value(row.get("gender"))
    race = clean_prompt_value(row.get("race"))
    vital_status = clean_prompt_value(row.get("vital_status"))
    primary_diagnosis = clean_prompt_value(row.get("primary_diagnosis"))
    tumor_stage = clean_prompt_value(row.get("tumor_stage"))
    ajcc_pathologic_stage = clean_prompt_value(row.get("ajcc_pathologic_stage"))
    tissue_type = clean_prompt_value(row.get("tissue_type"))
    tumor_descriptor = clean_prompt_value(row.get("tumor_descriptor"))

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
        ajcc_pathologic_stage = rng.choice(
            [s for s in stage_choices if s != ajcc_pathologic_stage]
        )

        tumor_stage_choices = ["stage i", "stage ii", "stage iii", "stage iv", "unknown"]
        tumor_stage = rng.choice([s for s in tumor_stage_choices if s != tumor_stage])

        if task_setting == "shortcut_present":
            descriptor_choices = ["Primary", "Metastatic", "Recurrence", "unknown"]
            tumor_descriptor = rng.choice(
                [s for s in descriptor_choices if s != tumor_descriptor]
            )

    if task_setting == "shortcut_present":
        return f"""
Clinical metadata:
- age_at_diagnosis: {age}
- gender: {gender}
- race: {race}
- vital_status: {vital_status}
- primary_diagnosis: {primary_diagnosis}
- tumor_stage: {tumor_stage}
- ajcc_pathologic_stage: {ajcc_pathologic_stage}
- tissue_type: {tissue_type}
- tumor_descriptor: {tumor_descriptor}
""".strip()

    if task_setting == "shortcut_controlled":
        return f"""
Clinical metadata:
- age_at_diagnosis: {age}
- gender: {gender}
- race: {race}
- vital_status: {vital_status}
- primary_diagnosis: {primary_diagnosis}
- tumor_stage: {tumor_stage}
- ajcc_pathologic_stage: {ajcc_pathologic_stage}
""".strip()

    raise ValueError("task_setting must be 'shortcut_present' or 'shortcut_controlled'.")


def rna_block(row: pd.Series, counterfactual: bool, rng: random.Random) -> str:
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

    for col in cols:
        value = row.get(col, np.nan)
        values[col] = None if pd.isna(value) else float(value)

    if counterfactual:
        for col in ["expr_mean", "expr_std", "expr_median", "expr_q25", "expr_q75", "expr_max"]:
            if values.get(col) is not None:
                values[col] = values[col] * rng.choice([0.5, 0.7, 1.3, 1.5])

        if values.get("expr_nonzero_fraction") is not None:
            values["expr_nonzero_fraction"] = max(
                0.0,
                min(1.0, 1.0 - values["expr_nonzero_fraction"]),
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


def image_summary_block(row: pd.Series) -> str:
    return f"""
Pathology/image availability summary:
- has_image: {clean_prompt_value(row.get("has_image"))}
- n_series: {clean_prompt_value(row.get("n_series"))}
- n_studies: {clean_prompt_value(row.get("n_studies"))}
- image_modalities: {clean_prompt_value(row.get("image_modalities"))}
""".strip()


def build_prompt(
    row: pd.Series,
    variant: str,
    task_setting: str,
    rng: random.Random,
) -> str:
    if variant == "full":
        instruction = (
            "You are given pathology/image availability metadata, "
            "clinical metadata, and RNA expression summary."
        )
        blocks = [
            image_summary_block(row),
            clinical_block(row, task_setting, counterfactual=False, rng=rng),
            rna_block(row, counterfactual=False, rng=rng),
        ]

    elif variant == "clinical_only":
        instruction = "You are given only clinical metadata."
        blocks = [clinical_block(row, task_setting, counterfactual=False, rng=rng)]

    elif variant == "rna_only":
        instruction = "You are given only RNA expression summary."
        blocks = [rna_block(row, counterfactual=False, rng=rng)]

    elif variant == "image_summary_only":
        instruction = (
            "You are given only pathology/image availability metadata. "
            "No pixel-level pathology image is provided."
        )
        blocks = [image_summary_block(row)]

    elif variant == "clinical_rna":
        instruction = "You are given clinical metadata and RNA expression summary."
        blocks = [
            clinical_block(row, task_setting, counterfactual=False, rng=rng),
            rna_block(row, counterfactual=False, rng=rng),
        ]

    elif variant == "counterfactual_clinical":
        instruction = (
            "You are given pathology/image availability metadata, "
            "counterfactually modified clinical metadata, and original RNA expression summary."
        )
        blocks = [
            image_summary_block(row),
            clinical_block(row, task_setting, counterfactual=True, rng=rng),
            rna_block(row, counterfactual=False, rng=rng),
        ]

    elif variant == "counterfactual_rna":
        instruction = (
            "You are given pathology/image availability metadata, "
            "original clinical metadata, and perturbed RNA expression summary."
        )
        blocks = [
            image_summary_block(row),
            clinical_block(row, task_setting, counterfactual=False, rng=rng),
            rna_block(row, counterfactual=True, rng=rng),
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    joined_blocks = "\n\n".join(blocks)

    return f"""
Task:
Predict the TCGA-SKCM sample type.

Allowed labels:
- Primary Tumor
- Metastatic

{instruction}

{joined_blocks}

Output requirements:
- Return only valid JSON.
- prediction must be exactly "Primary Tumor" or "Metastatic".
- confidence must be between 0 and 1.
- image_reliance, clinical_reliance, and rna_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance should be 0.

JSON fields:
prediction
confidence
used_modalities
image_reliance
clinical_reliance
rna_reliance
rationale_short
""".strip()


def build_prompt_records(
    df: pd.DataFrame,
    task_setting: str,
    eval_mode: str,
    variants: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    records = []

    for _, row in df.iterrows():
        for variant in variants:
            prompt = build_prompt(row, variant, task_setting, rng)

            records.append(
                {
                    "sample_id": f"{row['case_barcode']}::{variant}",
                    "task": "tcga_multimodal_sample_type",
                    "task_setting": task_setting,
                    "eval_mode": eval_mode,
                    "case_barcode": row["case_barcode"],
                    "variant": variant,
                    "true_label": row["sample_type"],
                    "prompt": prompt,
                    "has_image": int(row.get("has_image", 0)),
                    "has_rna": int(row.get("has_rna", 0)),
                    "gender": None if pd.isna(row.get("gender")) else row.get("gender"),
                    "age_at_diagnosis": None
                    if pd.isna(row.get("age_at_diagnosis"))
                    else row.get("age_at_diagnosis"),
                    "tumor_descriptor_raw": None
                    if pd.isna(row.get("tumor_descriptor"))
                    else row.get("tumor_descriptor"),
                    "tissue_type_raw": None
                    if pd.isna(row.get("tissue_type"))
                    else row.get("tissue_type"),
                }
            )

    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_tcga_multimodal_benchmark(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    raw_rna_dir = args.raw_dir / "rna"
    raw_rna_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching clinical metadata...")
    clinical = fetch_clinical_table(args.project_id)
    clinical_path = args.output_dir / "tcga_skcm_clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    print(f"Saved: {clinical_path} shape={clinical.shape}")

    print("\nQuerying RNA expression inventory...")
    rna_inventory = query_rna_inventory(args.project_id)
    rna_inventory_path = args.output_dir / "tcga_skcm_rna_file_inventory.csv"
    rna_inventory.to_csv(rna_inventory_path, index=False)
    print(f"Saved: {rna_inventory_path} shape={rna_inventory.shape}")

    print("\nDownloading and summarizing RNA expression files...")
    rna = summarize_rna_files(
        inventory=rna_inventory,
        raw_rna_dir=raw_rna_dir,
        max_rna_files=args.max_rna_files,
    )
    rna_path = args.output_dir / "tcga_skcm_rna_summary.csv"
    rna.to_csv(rna_path, index=False)
    print(f"Saved: {rna_path} shape={rna.shape}")

    print("\nFetching IDC pathology/image metadata...")
    image = fetch_idc_image_summary(
        collection_id=args.idc_collection_id,
        install_idc_index=not args.skip_idc_install,
    )
    image_path = args.output_dir / "tcga_skcm_image_summary.csv"
    image.to_csv(image_path, index=False)
    print(f"Saved: {image_path} shape={image.shape}")

    print("\nMerging modalities...")
    master = merge_modalities(clinical=clinical, rna=rna, image=image)

    benchmark_path = args.output_dir / args.output_filename
    master.to_csv(benchmark_path, index=False)
    print(f"Saved benchmark table: {benchmark_path} shape={master.shape}")

    full_split = build_full_split(
        master,
        require_all_modalities=not args.allow_missing_modalities,
    )
    full_path = args.output_dir / "taskA_test_full.csv"
    full_split.to_csv(full_path, index=False)
    print(f"Saved full split: {full_path} shape={full_split.shape}")
    print(full_split["sample_type"].value_counts().to_string())

    balanced_split = build_balanced_split(full_split, seed=args.seed)
    balanced_path = args.output_dir / "taskA_test_balanced.csv"
    balanced_split.to_csv(balanced_path, index=False)
    print(f"Saved balanced split: {balanced_path} shape={balanced_split.shape}")
    print(balanced_split["sample_type"].value_counts().to_string())

    prompt_source = full_split if args.eval_mode == "full" else balanced_split

    prompt_records = build_prompt_records(
        df=prompt_source,
        task_setting=args.task_setting,
        eval_mode=args.eval_mode,
        variants=args.variants,
        seed=args.seed,
    )

    prompts_path = args.output_dir / args.prompts_filename
    write_jsonl(prompts_path, prompt_records)

    print(f"\nSaved prompt JSONL: {prompts_path}")
    print(f"Prompt rows: {len(prompt_records)}")
    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the BioMM-Eval TCGA multimodal benchmark from "
            "TCGA/GDC clinical metadata, RNA expression summaries, and IDC image metadata."
        )
    )

    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help="TCGA/GDC project ID.",
    )
    parser.add_argument(
        "--idc-collection-id",
        default=DEFAULT_IDC_COLLECTION_ID,
        help="IDC collection ID.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("downloads/gdc"),
        help="Directory for downloaded raw GDC files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tcga_skcm"),
        help="Directory for generated benchmark files.",
    )
    parser.add_argument(
        "--output-filename",
        default="tcga_skcm_multimodal_benchmark.csv",
        help="Merged benchmark CSV filename.",
    )
    parser.add_argument(
        "--prompts-filename",
        default="tcga_skcm_multimodal_benchmark_prompts.jsonl",
        help="Prompt JSONL filename.",
    )
    parser.add_argument(
        "--max-rna-files",
        type=int,
        default=None,
        help="Optional cap on downloaded RNA files. Default downloads all usable open files.",
    )
    parser.add_argument(
        "--task-setting",
        choices=["shortcut_present", "shortcut_controlled"],
        default="shortcut_controlled",
        help="Whether to include shortcut-prone clinical fields in prompts.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["full", "balanced"],
        default="full",
        help="Which split to use for prompt JSONL generation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Prompt intervention variants to generate.",
    )
    parser.add_argument(
        "--allow-missing-modalities",
        action="store_true",
        help="Do not require both image metadata and RNA summary in taskA_test_full.csv.",
    )
    parser.add_argument(
        "--skip-idc-install",
        action="store_true",
        help="Skip automatic installation/update of idc-index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for balanced split and counterfactual prompt generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    build_tcga_multimodal_benchmark(parse_args())
