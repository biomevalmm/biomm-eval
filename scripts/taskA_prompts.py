import random
import numpy as np
import pandas as pd
from taskA_utils import clean_val


def clinical_block(row, task_setting="shortcut_controlled", counterfactual=False):
    age = row.get("age_at_diagnosis", np.nan)
    gender = clean_val(row.get("gender"))
    race = clean_val(row.get("race"))
    vital = clean_val(row.get("vital_status"))
    stage = clean_val(row.get("ajcc_pathologic_stage"))
    tumor_stage = clean_val(row.get("tumor_stage"))
    primary_diagnosis = clean_val(row.get("primary_diagnosis"))
    tissue_type = clean_val(row.get("tissue_type"))
    tumor_descriptor = clean_val(row.get("tumor_descriptor"))

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
        stage = random.choice([s for s in stage_choices if s != stage])

        tumor_stage_choices = ["stage i", "stage ii", "stage iii", "stage iv", "unknown"]
        tumor_stage = random.choice([s for s in tumor_stage_choices if s != tumor_stage])

        if task_setting == "shortcut_present":
            tumor_descriptor_choices = ["Primary", "Metastatic", "Recurrence", "unknown"]
            tumor_descriptor = random.choice([s for s in tumor_descriptor_choices if s != tumor_descriptor])

    common = f"""
Clinical metadata:
- age_at_diagnosis: {age}
- gender: {gender}
- race: {race}
- vital_status: {vital}
- primary_diagnosis: {primary_diagnosis}
- tumor_stage: {tumor_stage}
- ajcc_pathologic_stage: {stage}
""".strip()

    if task_setting == "shortcut_present":
        common += f"""
- tissue_type: {tissue_type}
- tumor_descriptor: {tumor_descriptor}
""".rstrip()

    return common


def rna_block(row, counterfactual=False):
    cols = [
        "expr_n_genes", "expr_mean", "expr_std", "expr_median",
        "expr_q25", "expr_q75", "expr_min", "expr_max",
        "expr_nonzero_fraction",
    ]

    values = {}
    for c in cols:
        v = row.get(c, np.nan)
        values[c] = None if pd.isna(v) else float(v)

    if counterfactual:
        for c in ["expr_mean", "expr_std", "expr_median", "expr_q25", "expr_q75", "expr_max"]:
            if values.get(c) is not None:
                values[c] *= random.choice([0.5, 0.7, 1.3, 1.5])

        if values.get("expr_nonzero_fraction") is not None:
            values["expr_nonzero_fraction"] = max(0.0, min(1.0, 1.0 - values["expr_nonzero_fraction"]))

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


def image_summary_block(row):
    return f"""
Image availability metadata:
- has_image: {clean_val(row.get("has_image"))}
- n_series: {clean_val(row.get("n_series"))}
- n_studies: {clean_val(row.get("n_studies"))}
- image_modalities: {clean_val(row.get("image_modalities"))}
""".strip()


def build_taskA_prompt(row, variant, task_setting="shortcut_controlled"):
    if variant == "full":
        instruction = "You are given image availability metadata, clinical metadata, and RNA expression summary."
        blocks = [image_summary_block(row), clinical_block(row, task_setting), rna_block(row)]

    elif variant == "clinical_only":
        instruction = "You are given only clinical metadata."
        blocks = [clinical_block(row, task_setting)]

    elif variant == "rna_only":
        instruction = "You are given only RNA expression summary."
        blocks = [rna_block(row)]

    elif variant == "image_summary_only":
        instruction = "You are given only image availability metadata. No pixel-level pathology image is provided."
        blocks = [image_summary_block(row)]

    elif variant == "clinical_rna":
        instruction = "You are given clinical metadata and RNA expression summary."
        blocks = [clinical_block(row, task_setting), rna_block(row)]

    elif variant == "counterfactual_clinical":
        instruction = "You are given image availability metadata, counterfactually modified clinical metadata, and original RNA expression summary."
        blocks = [image_summary_block(row), clinical_block(row, task_setting, counterfactual=True), rna_block(row)]

    elif variant == "counterfactual_rna":
        instruction = "You are given image availability metadata, original clinical metadata, and perturbed RNA expression summary."
        blocks = [image_summary_block(row), clinical_block(row, task_setting), rna_block(row, counterfactual=True)]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    joined = "\n\n".join(blocks)

    return f"""
You are a strict JSON generator. Output JSON only.

Task:
Predict the TCGA-SKCM sample type.

Allowed labels:
- Primary Tumor
- Metastatic

{instruction}

{joined}

This is for benchmark evaluation only, not clinical use.

Return only valid JSON. Do not use markdown. Do not explain.

Required JSON:
{{
  "prediction": "Primary Tumor or Metastatic",
  "confidence": 0.0,
  "used_modalities": ["image_summary", "clinical", "rna"],
  "image_reliance": 0.0,
  "clinical_reliance": 0.0,
  "rna_reliance": 0.0,
  "rationale_short": "brief reason"
}}

Rules:
- prediction must be exactly one of: Primary Tumor, Metastatic.
- confidence must be between 0 and 1.
- image_reliance, clinical_reliance, and rna_reliance must each be between 0 and 1.
- If image_summary is not provided, image_reliance must be 0.
- If clinical metadata is not provided, clinical_reliance must be 0.
- If RNA summary is not provided, rna_reliance must be 0.
- used_modalities must include only actually provided modalities.
- Return JSON only.
- The response must start with {{ and end with }}.
""".strip()
