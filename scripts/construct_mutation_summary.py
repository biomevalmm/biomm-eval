#!/usr/bin/env python3

"""
Construct TCGA mutation summary table from GDC MAF files.

Example:
    python scripts/construct_mutation_summary.py \
        --project-id TCGA-SKCM \
        --base-dir ./data
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =========================================================
# Config
# =========================================================

GDC_API = "https://api.gdc.cancer.gov"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =========================================================
# Helpers
# =========================================================

def gdc_post(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    """POST request to GDC API."""
    url = f"{GDC_API}/{endpoint.lstrip('/')}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def download_gdc_file(file_id: str, out_dir: Path) -> Path:
    """Download a GDC file."""
    url = f"{GDC_API}/data/{file_id}"

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.findall(r'filename="?([^"]+)"?', content_disposition)

    filename = match[0] if match else f"{file_id}.dat"
    out_path = out_dir / filename

    if out_path.exists():
        logging.info("Already exists: %s", out_path)
        return out_path

    logging.info("Downloading: %s", filename)

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return out_path


def normalize_case_barcode(barcode: str) -> str | None:
    """Normalize TCGA barcode to case-level barcode."""
    if pd.isna(barcode):
        return None

    barcode = str(barcode).strip()

    match = re.match(
        r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
        barcode,
        flags=re.I,
    )

    return match.group(1).upper() if match else None


def read_maf(path: Path) -> pd.DataFrame:
    """Read MAF file."""
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )


# =========================================================
# Candidate Search
# =========================================================

def build_maf_query(project_id: str) -> dict:
    fields = [
        "file_id",
        "file_name",
        "data_type",
        "data_format",
        "access",
        "experimental_strategy",
        "analysis.workflow_type",
        "file_size",
        "updated_datetime",
    ]

    filters = {
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
                    "value": ["Masked Somatic Mutation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                    "value": ["open"],
                },
            },
        ],
    }

    return {
        "filters": filters,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "100",
    }


def priority_score(row: pd.Series) -> int:
    """Ranking heuristic for selecting good MAF files."""
    name = str(row.get("file_name", "")).lower()
    workflow = str(row.get("analysis.workflow_type", "")).lower()

    score = 0

    if "mc3" in name:
        score += 50

    if "ensemble" in name:
        score += 10

    if "masked" in name:
        score += 5

    if "somatic" in name:
        score += 5

    if "aliquot" in name:
        score -= 10

    if "ensemble" in workflow:
        score += 5

    return score


# =========================================================
# Summary Construction
# =========================================================

def construct_case_summary(maf_df: pd.DataFrame) -> pd.DataFrame:
    """Construct case-level mutation summary."""

    maf_df = maf_df.copy()

    maf_df["case_barcode"] = (
        maf_df["Tumor_Sample_Barcode"]
        .map(normalize_case_barcode)
    )

    maf_df = maf_df[
        maf_df["case_barcode"].notna()
    ].copy()

    variant_col = "Variant_Classification"

    maf_df["is_nonsilent"] = ~(
        maf_df[variant_col]
        .fillna("")
        .str.lower()
        .isin(["silent", "synonymous", "synonymous_snv"])
    )

    summary_df = (
        maf_df
        .groupby("case_barcode")
        .agg(
            n_mutations=("case_barcode", "size"),
            n_nonsilent=("is_nonsilent", "sum"),
            n_unique_genes=("Hugo_Symbol", "nunique"),
        )
        .reset_index()
    )

    return summary_df


# =========================================================
# Main
# =========================================================

def main(args):

    base_dir = Path(args.base_dir)

    raw_gdc_dir = base_dir / "downloads" / "gdc"
    raw_gdc_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Searching candidate MAF files...")

    payload = build_maf_query(args.project_id)

    maf_json = gdc_post("files", payload)

    maf_hits = maf_json["data"]["hits"]

    maf_meta_df = pd.json_normalize(maf_hits)

    maf_meta_df["file_size"] = (
        pd.to_numeric(
            maf_meta_df["file_size"],
            errors="coerce",
        )
        .fillna(0)
    )

    maf_meta_df["priority"] = (
        maf_meta_df.apply(priority_score, axis=1)
    )

    maf_meta_df = maf_meta_df.sort_values(
        ["priority", "file_size", "updated_datetime"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    results = []

    top_k = min(args.top_k, len(maf_meta_df))

    for i in range(top_k):

        row = maf_meta_df.iloc[i]

        try:
            local_path = download_gdc_file(
                row["file_id"],
                raw_gdc_dir,
            )

            maf_df = read_maf(local_path)

            n_cases = (
                maf_df["Tumor_Sample_Barcode"]
                .dropna()
                .astype(str)
                .map(normalize_case_barcode)
                .dropna()
                .nunique()
            )

            results.append({
                "file_name": row["file_name"],
                "local_path": local_path,
                "n_cases": n_cases,
            })

            logging.info(
                "[%d/%d] %s -> %d cases",
                i + 1,
                top_k,
                row["file_name"],
                n_cases,
            )

        except Exception as e:
            logging.warning("FAILED: %s", e)

    scan_df = pd.DataFrame(results)

    scan_df = scan_df.sort_values(
        "n_cases",
        ascending=False,
    )

    best = scan_df.iloc[0]

    logging.info("Selected best MAF: %s", best["file_name"])

    best_maf = read_maf(best["local_path"])

    summary_df = construct_case_summary(best_maf)

    out_path = base_dir / "mutation_summary.csv"

    summary_df.to_csv(out_path, index=False)

    logging.info("Saved summary: %s", out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    main(args)
