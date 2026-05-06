import os
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from taskA_utils import load_jsonl_safe


CLASSES = ["Primary Tumor", "Metastatic"]


def compute_metrics(results_path, out_dir, run_prefix, suffix=""):
    res = load_jsonl_safe(results_path)

    if len(res) == 0:
        print("[Metrics] No valid rows.")
        return None

    res = res.drop_duplicates(subset=["case_barcode", "variant"], keep="last").copy()

    rows = []
    report_lines = []

    for variant, g in res.groupby("variant"):
        g = g.dropna(subset=["prediction", "true_label"]).copy()
        g = g[g["true_label"].isin(CLASSES)]
        g = g[g["prediction"].isin(CLASSES)]

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
            "mean_clinical_reliance_self_report": float(g["clinical_reliance"].mean()),
            "mean_rna_reliance_self_report": float(g["rna_reliance"].mean()),
        })

        report_lines.append(f"\n\n===== {variant} =====\n")
        report_lines.append(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))

    metrics = pd.DataFrame(rows)

    metrics_path = os.path.join(out_dir, f"{run_prefix}_metrics{suffix}.csv")
    report_path = os.path.join(out_dir, f"{run_prefix}_report{suffix}.txt")

    metrics.to_csv(metrics_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n=== Task A Metrics {suffix or 'final'} ===")
    print(metrics.to_string(index=False))
    print("Saved metrics:", metrics_path)
    print("Saved report :", report_path)

    return metrics
