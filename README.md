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

This anonymized repository may receive incremental updates during the review period for documentation and reproducibility improvements.
