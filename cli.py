"""
EWS Anomaly Detection CLI.

Usage:
    python cli.py setup              Oracle tablolarini olustur
    python cli.py load               Sentetik veri uret ve Oracle'a yukle
    python cli.py train              Modeli egit (Oracle'dan oku)
    python cli.py score              Skorla ve sonuclari Oracle'a yaz
    python cli.py run                Train + Score tek seferde
    python cli.py test               Sentetik veri ile full test (Oracle'siz)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

from engine.pipeline import EWSPipeline


def setup_logging(log_dir="logs", level="INFO"):
    Path(log_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{log_dir}/ews_{ts}.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_setup():
    pipe = EWSPipeline()
    pipe.setup()


def cmd_load():
    from generate_data import generate_training_data, generate_scoring_data
    pipe = EWSPipeline()

    print("Generating synthetic data...")
    train_df = generate_training_data()
    scoring_df, _ = generate_scoring_data()
    pipe.load_data(train_df, scoring_df)


def cmd_train():
    pipe = EWSPipeline()
    pipe.train()


def cmd_score():
    pipe = EWSPipeline()
    results = pipe.score()
    _print_summary(results)


def cmd_run():
    pipe = EWSPipeline()
    results = pipe.run()
    _print_summary(results)


def cmd_test():
    """Oracle'siz full test — sentetik veri ile."""
    from generate_data import generate_training_data, generate_scoring_data
    from engine.config_loader import load_config, get_feature_list
    from engine.models import AnomalyModels
    from engine.scorer import AnomalyScorer

    config = load_config()
    features = get_feature_list(config)

    print("Generating data...")
    train_df = generate_training_data()
    scoring_df, labels = generate_scoring_data()

    train_only = train_df[train_df["split_flag"] == "TRAIN"]
    test_only = train_df[train_df["split_flag"] == "TEST"]

    print(f"Train: {len(train_only)}, Test: {len(test_only)}, Scoring: {len(scoring_df)}")

    print("Training models...")
    models = AnomalyModels(config)
    models.fit(train_only[features].fillna(0).values)

    print("Scoring...")
    scorer = AnomalyScorer(config, models)
    results = scorer.score(scoring_df)

    _print_summary(results)

    # Validation
    ev = results.merge(labels, on="customer_id")
    al = ev[ev["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]

    print("\n=== VALIDATION ===")
    for t in ["A_UNIVARIATE", "B_MULTIVARIATE", "C_SUBTLE_DRIFT"]:
        tot = (ev["anomaly_type"] == t).sum()
        c = (al["anomaly_type"] == t).sum()
        pct = c / tot * 100 if tot > 0 else 0
        print(f"  {t}: {c}/{tot} (%{pct:.0f})")

    total_a = ev["is_anomaly"].sum()
    total_c = al["is_anomaly"].sum()
    print(f"  TOPLAM: {total_c}/{total_a} (%{total_c / total_a * 100:.0f})")


def _print_summary(results):
    print(f"\n=== SCORING SUMMARY ===")
    print(f"Total: {len(results)}")
    for band in ["KIRMIZI", "TURUNCU", "SARI", "NORMAL"]:
        cnt = (results["alert_band"] == band).sum()
        print(f"  {band}: {cnt}")

    print(f"\nTop 5:")
    for _, row in results.head(5).iterrows():
        print(f"  {row['customer_id']} | {row['anomaly_score']} | {row['alert_band']} | flags: {row['uni_flag_count']}")
        for feat, d in row["detay"].items():
            ico = "UP" if d["degisim_pct"] > 0 else "DN"
            print(f"    {d['label']}: {d['beklenen']}->{d['gerceklesen']} ({ico}%{abs(d['degisim_pct']):.0f})")


COMMANDS = {
    "setup": cmd_setup,
    "load": cmd_load,
    "train": cmd_train,
    "score": cmd_score,
    "run": cmd_run,
    "test": cmd_test,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    setup_logging()
    logging.info(f"CLI command: {cmd}")
    COMMANDS[cmd]()
