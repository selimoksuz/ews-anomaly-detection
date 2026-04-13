"""
EWS Anomaly Detection CLI.

Usage:
    python cli.py setup
    python cli.py load
    python cli.py train
    python cli.py score
    python cli.py run
    python cli.py test
    python cli.py prepare-demo-data
    python cli.py develop [segment]
    python cli.py retrain [segment]
    python cli.py tune-weights [segment] [model_version] [apply]
    python cli.py evaluate-outcomes [segment] [model_version]
    python cli.py compare [segment] [challenger_version]
    python cli.py promote [segment] [model_version]
    python cli.py score-live [segment]
    python cli.py cleanup
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.config_loader import get_feature_list, load_config
from engine.lifecycle import LifecycleManager
from engine.models import AnomalyModels
from engine.oracle_io import OracleConnector
from engine.pipeline import EWSPipeline
from engine.scorer import AnomalyScorer


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


def cmd_setup(*_):
    pipe = EWSPipeline()
    pipe.setup()


def cmd_load(*_):
    from generate_data import generate_outcome_labels, generate_scoring_data, generate_training_data

    pipe = EWSPipeline()
    print("Generating synthetic data...")
    train_df = generate_training_data()
    scoring_df, _ = generate_scoring_data()
    outcomes_df = generate_outcome_labels(train_df)
    pipe.load_data(train_df, scoring_df, outcomes_df)


def cmd_train(*_):
    pipe = EWSPipeline()
    pipe.train()


def cmd_score(*_):
    pipe = EWSPipeline()
    results = pipe.score()
    _print_summary(results)


def cmd_run(*_):
    pipe = EWSPipeline()
    results = pipe.run()
    _print_summary(results)


def cmd_test(*_):
    """Oracle-free full test with synthetic data."""
    from generate_data import generate_scoring_data, generate_training_data

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

    ev = results.merge(labels, on="customer_id")
    al = ev[ev["alert_band"].isin(["KIRMIZI", "TURUNCU", "SARI"])]

    print("\n=== VALIDATION ===")
    for anomaly_type in ["A_UNIVARIATE", "B_MULTIVARIATE", "C_SUBTLE_DRIFT"]:
        total = (ev["anomaly_type"] == anomaly_type).sum()
        caught = (al["anomaly_type"] == anomaly_type).sum()
        pct = caught / total * 100 if total > 0 else 0
        print(f"  {anomaly_type}: {caught}/{total} (%{pct:.0f})")

    total_anomalies = ev["is_anomaly"].sum()
    total_caught = al["is_anomaly"].sum()
    print(f"  TOPLAM: {total_caught}/{total_anomalies} (%{total_caught / total_anomalies * 100:.0f})")


def cmd_prepare_demo_data(*_):
    from generate_data import generate_outcome_labels, generate_scoring_data, generate_training_data

    config = load_config()
    sources_cfg = config.get("sources", {})

    input_source_name = config.get("development", {}).get("source_name", "input_features")
    live_source_name = config.get("live_scoring", {}).get("source_name", input_source_name)
    train_path = Path(sources_cfg.get(input_source_name, {}).get("csv", {}).get("path", "data/lifecycle_input_features.csv"))
    score_path = Path(sources_cfg.get(live_source_name, {}).get("csv", {}).get("path", "data/lifecycle_input_features.csv"))
    outcomes_path = Path(sources_cfg.get("outcomes", {}).get("csv", {}).get("path", "data/lifecycle_outcomes.csv"))

    train_df = generate_training_data()
    scoring_df, _ = generate_scoring_data()
    outcomes_df = generate_outcome_labels(train_df)

    dev_backend = sources_cfg.get(input_source_name, {}).get("backend", "csv")
    score_backend = sources_cfg.get(live_source_name, {}).get("backend", "csv")
    outcome_backend = sources_cfg.get("outcomes", {}).get("backend", "csv")

    if dev_backend == score_backend == outcome_backend == "oracle":
        dev_table = sources_cfg.get(input_source_name, {}).get("oracle", {}).get("table", "input_features")
        live_table = sources_cfg.get(live_source_name, {}).get("oracle", {}).get("table", dev_table)
        combined_input = pd.concat(
            [
                train_df.drop(columns=["split_flag"], errors="ignore"),
                scoring_df.drop(columns=["split_flag"], errors="ignore"),
            ],
            ignore_index=True,
        )
        with OracleConnector(config) as ora:
            ora.setup_tables(drop_existing=True)
            if dev_table == live_table:
                ora.replace_rows(dev_table, combined_input)
            else:
                ora.replace_rows(dev_table, train_df)
                ora.replace_rows(live_table, scoring_df)
            ora.replace_rows("outcomes", outcomes_df)
        print(ora._qualified_table_name(dev_table))
        if live_table != dev_table:
            print(ora._qualified_table_name(live_table))
        print(ora._qualified_table_name("outcomes"))
        return

    train_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    combined_input = pd.concat(
        [
            train_df.drop(columns=["split_flag"], errors="ignore"),
            scoring_df.drop(columns=["split_flag"], errors="ignore"),
        ],
        ignore_index=True,
    )
    if train_path == score_path:
        combined_input.to_csv(train_path, index=False)
    else:
        train_df.to_csv(train_path, index=False)
        scoring_df.to_csv(score_path, index=False)
    outcomes_df.to_csv(outcomes_path, index=False)

    print(train_path)
    print(score_path)
    print(outcomes_path)


def cmd_develop(*args):
    manager = LifecycleManager()
    result = manager.develop(segment=args[0] if args else None)
    print(result["model_version"])


def cmd_retrain(*args):
    manager = LifecycleManager()
    result = manager.retrain(segment=args[0] if args else None)
    print(result["model_version"])


def cmd_tune_weights(*args):
    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 and args[1] != "-" else None
    apply = None
    if len(args) > 2:
        apply = args[2].strip().lower() in {"1", "true", "yes", "apply"}
    result = manager.tune_weights(segment=segment, model_version=model_version, apply=apply)
    print(result["weight_version"])


def cmd_evaluate_outcomes(*args):
    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 else None
    result = manager.evaluate_outcomes(segment=segment, model_version=model_version)
    print(result["evaluation_path"])


def cmd_compare(*args):
    manager = LifecycleManager()
    segment = args[0] if args else None
    challenger_version = args[1] if len(args) > 1 else None
    result = manager.compare(segment=segment, challenger_version=challenger_version)
    print(result["recommendation"]["winner"])


def cmd_promote(*args):
    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 else None
    result = manager.promote(segment=segment, model_version=model_version)
    print(result["promoted_model"])


def cmd_score_live(*args):
    manager = LifecycleManager()
    result = manager.score_live(segment=args[0] if args else None)
    print(result["snapshot_date"])


def cmd_cleanup(*_):
    manager = LifecycleManager()
    result = manager.cleanup()
    print(result)


def _print_summary(results):
    print("\n=== SCORING SUMMARY ===")
    print(f"Total: {len(results)}")
    for band in ["KIRMIZI", "TURUNCU", "SARI", "NORMAL"]:
        cnt = (results["alert_band"] == band).sum()
        print(f"  {band}: {cnt}")

    print("\nTop 5:")
    for _, row in results.head(5).iterrows():
        print(
            f"  {row['customer_id']} | {row['anomaly_score']} | "
            f"{row['alert_band']} | flags: {row['uni_flag_count']}"
        )
        for _, detail in row["detay"].items():
            ico = "UP" if detail["degisim_pct"] > 0 else "DN"
            print(
                f"    {detail['label']}: {detail['beklenen']}->{detail['gerceklesen']}"
                f" ({ico}%{abs(detail['degisim_pct']):.0f})"
            )


COMMANDS = {
    "setup": cmd_setup,
    "load": cmd_load,
    "train": cmd_train,
    "score": cmd_score,
    "run": cmd_run,
    "test": cmd_test,
    "prepare-demo-data": cmd_prepare_demo_data,
    "develop": cmd_develop,
    "retrain": cmd_retrain,
    "tune-weights": cmd_tune_weights,
    "evaluate-outcomes": cmd_evaluate_outcomes,
    "compare": cmd_compare,
    "promote": cmd_promote,
    "score-live": cmd_score_live,
    "cleanup": cmd_cleanup,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    config = load_config()
    logs_dir = config.get("registry", {}).get("logs_dir", "logs")
    setup_logging(log_dir=logs_dir)

    cmd = sys.argv[1]
    args = sys.argv[2:]
    logging.info("CLI command: %s %s", cmd, " ".join(args))
    COMMANDS[cmd](*args)
