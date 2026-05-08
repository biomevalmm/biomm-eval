#!/usr/bin/env python3
"""
Build Task B BioMM-Eval dataset from the curated genome CSV.

Input:
  selected_sequences_saureus199_cacnes200.csv

Output:
  taskB_samples.jsonl

Each JSONL row contains:
  - sample_id
  - assembly_accession
  - true_label
  - variant
  - prompt
  - metadata fields
  - sequence length info
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


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

DEFAULT_MAX_SEQUENCE_CHARS = 100_000
DEFAULT_PERTURB_RATE = 0.01
DEFAULT_SEED = 42


def clean_val(x: object) -> str:
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    if x.lower() in {
        "",
        "nan",
        "none",
        "null",
        "missing",
        "not collected",
        "not provided",
    }:
        return "unknown"

    return x


def normalize_organism_name(x: object) -> str:
    x = clean_val(x)

    if "Staphylococcus aureus" in x:
        return "Staphylococcus aureus"

    if "Cutibacterium acnes" in x or "Propionibacterium acnes" in x:
        return "Cutibacterium acnes"

    return x


def sequence_context(seq: object, max_chars: int) -> tuple[str, int, int]:
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
    middle = seq[mid_start : mid_start + part]

    end = seq[-part:]

    context = (
        start
        + "\n...[MIDDLE_SEGMENT]...\n"
        + middle
        + "\n...[END_SEGMENT]...\n"
        + end
    )

    used_len = len(start) + len(middle) + len(end)
    return context, full_len, used_len


def perturb_sequence(seq_text: str, mutation_rate: float, rng: random.Random) -> str:
    bases = ["A", "C", "G", "T"]
    out = []

    for c in seq_text:
        if c in bases and rng.random() < mutation_rate:
            choices = [b for b in bases if b != c]
            out.append(rng.choice(choices))
        else:
            out.append(c)

    return "".join(out)


def metadata_block(row: pd.Series, counterfactual: bool, rng: random.Random) -> str:
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

        host = rng.choice([x for x in host_choices if x != host])
        host_disease = rng.choice([x for x in disease_choices if x != host_disease])
        isolation_source = rng.choice([x for x in source_choices if x != isolation_source])
        body_site = rng.choice([x for x in body_choices if x != body_site])

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


def build_prompt(
    row: pd.Series,
    variant: str,
    max_sequence_chars: int,
    perturb_rate: float,
    rng: random.Random,
) -> tuple[str, int, int]:
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
""".strip()
        ]

    elif variant == "metadata_only":
        instruction = "You are given only biological metadata. No DNA sequence is provided."
        blocks = [metadata_block(row, counterfactual=False, rng=rng)]

    elif variant == "metadata_counterfactual":
        instruction = (
            "You are given original long genome DNA sequence segments "
            "but counterfactually modified metadata."
        )
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
        instruction = (
            "You are given perturbed long genome DNA sequence segments "
            "and original biological metadata."
        )
        perturbed = perturb_sequence(seq_ctx, perturb_rate, rng)
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

    joined_blocks = "\n\n".join(blocks)

    prompt = f"""
Task:
Classify the bacterial genome into exactly one organism.

Allowed labels:
- Staphylococcus aureus
- Cutibacterium acnes

{instruction}

{joined_blocks}

Output requirements:
- Return only valid JSON.
- prediction must be exactly "Staphylococcus aureus" or "Cutibacterium acnes".
- confidence must be between 0 and 1.
- sequence_reliance and metadata_reliance must each be between 0 and 1.
- If a modality is not provided, its reliance should be 0.

JSON fields:
prediction
confidence
used_modalities
sequence_reliance
metadata_reliance
rationale_short
""".strip()

    return prompt, full_len, used_len


def load_curated_data(input_csv: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_cols = [
        "assembly_accession",
        "organism_name",
        "joined_sequence",
        "joined_seq_len",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["true_label"] = df["organism_name"].apply(normalize_organism_name)
    df = df[df["true_label"].isin(CLASSES)].copy()

    df = df[df["joined_sequence"].notna()].copy()
    df["joined_seq_len_numeric"] = pd.to_numeric(
        df["joined_seq_len"],
        errors="coerce",
    )
    df = df[df["joined_seq_len_numeric"] > 1000].copy()

    df = df.drop_duplicates(subset=["assembly_accession"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_task_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    df = load_curated_data(args.input_csv, args.seed)

    if args.n_per_class is not None:
        parts = []
        for cls in CLASSES:
            sub = df[df["true_label"] == cls].copy()
            n = min(args.n_per_class, len(sub))
            parts.append(sub.sample(n=n, random_state=args.seed))

        df = (
            pd.concat(parts, axis=0)
            .sample(frac=1, random_state=args.seed)
            .reset_index(drop=True)
        )

    rows = []

    for _, row in df.iterrows():
        for variant in args.variants:
            prompt, full_len, used_len = build_prompt(
                row=row,
                variant=variant,
                max_sequence_chars=args.max_sequence_chars,
                perturb_rate=args.perturb_rate,
                rng=rng,
            )

            assembly_accession = clean_val(row.get("assembly_accession"))

            out = {
                "sample_id": f"{assembly_accession}::{variant}",
                "assembly_accession": assembly_accession,
                "variant": variant,
                "true_label": row["true_label"],
                "organism_name_raw": clean_val(row.get("organism_name")),
                "prompt": prompt,
                "full_sequence_length": full_len,
                "provided_sequence_length": used_len,
                "joined_seq_len": int(float(row.get("joined_seq_len"))),
                "host": clean_val(row.get("host")),
                "host_disease": clean_val(row.get("host_disease")),
                "isolation_source": clean_val(row.get("isolation_source")),
                "body_site": clean_val(row.get("body_site")),
                "geo_loc_name": clean_val(row.get("geo_loc_name")),
                "strain": clean_val(row.get("strain")),
                "isolate": clean_val(row.get("isolate")),
                "assembly_level": clean_val(row.get("assembly_level")),
                "refseq_category": clean_val(row.get("refseq_category")),
                "submitter": clean_val(row.get("submitter")),
            }

            rows.append(out)

    write_jsonl(args.output_jsonl, rows)

    print("Saved:", args.output_jsonl)
    print("Input samples:", len(df))
    print("Variants:", len(args.variants))
    print("Total task rows:", len(rows))
    print()
    print("Class counts:")
    print(df["true_label"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task B BioMM-Eval JSONL dataset from curated genome CSV."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to selected_sequences_saureus199_cacnes200.csv",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output path for taskB_samples.jsonl",
    )
    parser.add_argument(
        "--max-sequence-chars",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_CHARS,
    )
    parser.add_argument(
        "--perturb-rate",
        type=float,
        default=DEFAULT_PERTURB_RATE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Optional balanced sample size per class. Default uses full dataset.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
    )

    return parser.parse_args()


if __name__ == "__main__":
    build_task_dataset(parse_args())
