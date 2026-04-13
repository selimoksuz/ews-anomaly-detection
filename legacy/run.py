"""
EWS Anomaly Detection - Main runner.

Usage:
    python legacy/run.py

Steps:
  1. Generate synthetic data (train + inference)
  2. Train ensemble model on normal data
  3. Score inference data
  4. Evaluate against known anomaly labels
  5. Print alert report with human-readable explanations
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate

from legacy.config import ALL_FEATURES
from legacy.model import ExplainableEnsemble
from scripts.generate_data import generate_inference_data, generate_normal_data


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def evaluate(results, labels):
    """Evaluate detection performance against known labels."""
    merged = results.merge(labels, on="customer_id")

    print_header("DETECTION PERFORMANCE")

    for band in ["KIRMIZI", "TURUNCU", "SARI"]:
        in_band = merged[merged["alert_band"] == band]
        if len(in_band) == 0:
            continue
        n_total = len(in_band)
        n_true = in_band["is_anomaly"].sum()
        precision = n_true / n_total * 100 if n_total > 0 else 0

        by_type = in_band[in_band["is_anomaly"]].groupby("anomaly_type").size()
        type_str = ", ".join([f"{t}: {c}" for t, c in by_type.items()])

        print(f"\n  {band}:")
        print(f"    Alert sayisi: {n_total}")
        print(f"    Gercek anomali: {n_true} (precision: %{precision:.1f})")
        print(f"    Tip dagilimi: {type_str}")

    # Overall recall by anomaly type
    print(f"\n  RECALL (anomalilerin yakalanma orani):")
    all_anomalies = merged[merged["is_anomaly"]]
    alerted = merged[merged["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]
    caught = alerted[alerted["is_anomaly"]]

    for atype in ["A_UNIVARIATE", "B_MULTIVARIATE", "C_SUBTLE_DRIFT"]:
        total = len(all_anomalies[all_anomalies["anomaly_type"] == atype])
        found = len(caught[caught["anomaly_type"] == atype])
        pct = found / total * 100 if total > 0 else 0
        print(f"    {atype}: {found}/{total} (%{pct:.1f})")

    total_anom = len(all_anomalies)
    total_caught = len(caught)
    print(f"    TOPLAM: {total_caught}/{total_anom} (%{total_caught/total_anom*100:.1f})")


def print_alert_report(results, top_n=15):
    """Print top alerts with human-readable explanations."""
    print_header("ALERT RAPORU (Top 15)")

    alerts = results[results["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])].head(top_n)

    for _, row in alerts.iterrows():
        band_icon = {"KIRMIZI": "***", "TURUNCU": "**", "SARI": "*"}.get(row["alert_band"], "")
        print(f"\n  {band_icon} {row['customer_id']}  |  Skor: {row['anomaly_score']}  |  {row['alert_band']}")
        print(f"     AE: {row['ae_score']}  IF: {row['if_score']}  MD: {row['md_score']}")
        print(f"     Neden:")

        for feat, detail in row["detay"].items():
            print(
                f"       - {detail['label']}: "
                f"{detail['beklenen']} -> {detail['gerceklesen']} "
                f"(degisim: %{detail['degisim_pct']}, katki: %{detail['katki_pct']})"
            )


def print_score_distribution(results):
    """Print score distribution summary."""
    print_header("SKOR DAGILIMI")

    bands = results["alert_band"].value_counts()
    total = len(results)

    table = []
    for band in ["KIRMIZI", "TURUNCU", "SARI", "NORMAL"]:
        n = bands.get(band, 0)
        pct = n / total * 100
        table.append([band, n, f"%{pct:.1f}"])

    print(tabulate(table, headers=["Bant", "Musteri Sayisi", "Oran"], tablefmt="simple"))

    print(f"\n  Skor istatistikleri:")
    print(f"    Min:    {results['anomaly_score'].min()}")
    print(f"    Median: {results['anomaly_score'].median()}")
    print(f"    P95:    {results['anomaly_score'].quantile(0.95):.1f}")
    print(f"    P99:    {results['anomaly_score'].quantile(0.99):.1f}")
    print(f"    Max:    {results['anomaly_score'].max()}")


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # ── 1. Generate Data ──
    print_header("VERI URETIMI")
    train_df = generate_normal_data()
    print(f"  Training: {len(train_df)} customers x {len(ALL_FEATURES)} features")

    inf_df, labels = generate_inference_data()
    print(f"  Inference: {len(inf_df)} customers, {labels['is_anomaly'].sum()} anomalies")

    train_df.to_csv("data/training_data.csv", index=False)
    inf_df.to_csv("data/inference_data.csv", index=False)
    labels.to_csv("data/anomaly_labels.csv", index=False)

    # ── 2. Train Model ──
    print_header("MODEL EGITIMI")
    model = ExplainableEnsemble()
    model.fit(train_df)

    # ── 3. Score Inference Data ──
    print_header("SKORLAMA")
    results = model.predict(inf_df)
    print(f"  {len(results)} customer scored")

    # ── 4. Results ──
    print_score_distribution(results)
    evaluate(results, labels)
    print_alert_report(results)

    # ── 5. Save ──
    output = results.drop(columns=["detay"])
    output.to_csv("output/alert_results.csv", index=False)
    print(f"\n  Sonuclar kaydedildi: output/alert_results.csv")

    # Save detailed top alerts
    top_alerts = results[results["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])].head(50)
    detail_rows = []
    for _, row in top_alerts.iterrows():
        for feat, d in row["detay"].items():
            detail_rows.append({
                "customer_id": row["customer_id"],
                "anomaly_score": row["anomaly_score"],
                "alert_band": row["alert_band"],
                "feature": feat,
                "label": d["label"],
                "beklenen": d["beklenen"],
                "gerceklesen": d["gerceklesen"],
                "degisim_pct": d["degisim_pct"],
                "katki_pct": d["katki_pct"],
            })
    pd.DataFrame(detail_rows).to_csv("output/alert_details.csv", index=False)
    print(f"  Detaylar kaydedildi: output/alert_details.csv")


if __name__ == "__main__":
    main()
