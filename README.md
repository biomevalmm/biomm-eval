# BioMM-Eval

BioMM-Eval is a multimodal biomedical evaluation framework for analyzing modality reliance, shortcut sensitivity, counterfactual robustness, and integration behavior in multimodal foundation models.

The benchmark evaluates multimodal reasoning across:
- clinical metadata
- genomic sequence context
- RNA expression summaries
- image-derived information
- dermoscopic images

using controlled multimodal intervention settings including:
- unimodal ablations
- multimodal integration
- metadata counterfactuals
- sequence perturbations
- prediction-change diagnostics


## Supported tasks

- Task A: TCGA-SKCM multimodal classification
- Task B: bacterial genome classification
- Task C: dermoscopic lesion classification



## Evaluated models

- GPT-5.4-mini
- Gemini-3-Flash
- DeepSeek-V4-Flash
- DeepSeek-VL2
- Qwen3.6-27B
- Phi-4-Reasoning-Vision-15B
- Molmo2-8B
- biomed-LLaVA-NeXT-Llama3-8B


## Dataset generation

BioMM-Eval provides scripts for constructing the three benchmark datasets used in the paper.

### Environment

```bash
pip install pandas numpy requests scikit-learn tqdm pillow matplotlib openai
````

Some tasks require external data downloads from public biomedical repositories.

---

### Task A: TCGA-SKCM multimodal benchmark

This script builds the TCGA-SKCM clinical + RNA + image-summary benchmark and generates prompt records for intervention-style evaluation.

```bash
python scripts/build_tcga_multimodal_benchmark.py \
  --output-dir data/tcga_skcm \
  --raw-dir downloads/gdc \
  --task-setting shortcut_controlled \
  --eval-mode full
```

Main outputs:

```text
data/tcga_skcm/
  tcga_skcm_multimodal_benchmark.csv
  taskA_test_full.csv
  taskA_test_balanced.csv
  tcga_skcm_multimodal_benchmark_prompts.jsonl
```

Useful options:

```bash
--eval-mode balanced
--max-rna-files 50
--allow-missing-modalities
--skip-idc-install
```

---

### Task B: bacterial genome + metadata benchmark

This script evaluates bacterial genome sequence + metadata classification under multiple intervention settings.

Before running, update the configuration block in:

```text
scripts/build_genome_metadata_benchmark.py
```

Required input CSV:

```text
selected_sequences_saureus199_cacnes200.csv
```

Required columns include:

```text
assembly_accession
organism_name
joined_sequence
joined_seq_len
```

Run:

```bash
python scripts/build_genome_metadata_benchmark.py
```

Main outputs:

```text
taskB_<model>_results_<run_id>.jsonl
taskB_<model>_failed_<run_id>.jsonl
taskB_<model>_metrics_<run_id>.csv
taskB_<model>_reliance_<run_id>.csv
taskB_<model>_report_<run_id>.txt
```

Evaluated variants:

```text
full
sequence_only
metadata_only
metadata_counterfactual
sequence_perturbed
```

Note: set the API key via the script configuration or adapt the script to read from an environment variable such as `OPENAI_API_KEY`.

---

### Task C: ISIC dermatology multimodal benchmark

This script builds the ISIC 2019 dermoscopic image + metadata benchmark. It can also optionally run model evaluation.

Build metadata-only benchmark files:

```bash
python scripts/build_dermatology_multimodal_benchmark.py \
  --output-dir data/isic_dermatology \
  --raw-dir downloads/isic2019
```

Download and extract images:

```bash
python scripts/build_dermatology_multimodal_benchmark.py \
  --output-dir data/isic_dermatology \
  --raw-dir downloads/isic2019 \
  --image-dir data/isic_dermatology/images \
  --download-images
```

Run evaluation:

```bash
python scripts/build_dermatology_multimodal_benchmark.py \
  --output-dir data/isic_dermatology \
  --raw-dir downloads/isic2019 \
  --image-dir data/isic_dermatology/images \
  --download-images \
  --run-eval \
  --eval-mode balanced \
  --model gpt-5.4-mini
```

Main outputs:

```text
data/isic_dermatology/
  isic_dermatology_multimodal_benchmark.csv
  taskC_test_full.csv
  taskC_test_balanced.csv
  taskC_<model>_<mode>_results_<run_id>.jsonl
  taskC_<model>_<mode>_metrics_<run_id>.csv
  taskC_<model>_<mode>_reliance_<run_id>.csv
```

Evaluated variants:

```text
full
image_only
metadata_only
metadata_counterfactual
```

---

## Reproducibility notes

All benchmark construction scripts use a fixed random seed by default:

```text
seed = 42
```

Balanced splits and counterfactual variants are generated deterministically given the same input data and seed.

Large raw files, downloaded images, and model outputs are not committed to the repository. They should be regenerated using the scripts above.


