"""
EWS Anomaly Detection CLI.

Usage:
    python cli.py setup
    python cli.py train
    python cli.py score
    python cli.py run
    python cli.py develop [segment]
    python cli.py retrain [segment]
    python cli.py tune-weights [segment] [model_version] [apply]
    python cli.py evaluate-outcomes [segment] [model_version]
    python cli.py compare [segment] [challenger_version]
    python cli.py promote [segment] [model_version]
    python cli.py score-live [segment] [snapshot_date]
    python cli.py score-live [segment] [start_date] [end_date]
    python cli.py run-batch [segment]
    python cli.py load-multivar-oracle [input_path] [replace|append] [delete-local]
    python cli.py run-multivar-anomaly [oracle|input_path|-] [scoring_month|-] [max_train_rows] [max_score_rows|-]
    python cli.py compare-preprocessing [segment]
    python cli.py compare-feature-selection [segment]
    python cli.py compare-sampling [segment]
    python cli.py prepare-ticari-orta-faz1-demo [segment]
    python cli.py run-ticari-orta-faz1-demo [segment]
    python cli.py reset-runtime
    python cli.py cleanup
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from engine.config_loader import load_config, resolve_project_path
from engine.multivar_anomaly import load_multivar_csv_to_oracle, run_multivar_anomaly


def setup_logging(log_dir="logs", level="INFO", enable_file=True):
    resolved_log_dir = resolve_project_path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if enable_file:
        handlers.append(logging.FileHandler(resolved_log_dir / f"ews_{ts}.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_setup(*_):
    from engine.pipeline import EWSPipeline

    pipe = EWSPipeline()
    pipe.setup()

def cmd_train(*_):
    from engine.pipeline import EWSPipeline

    pipe = EWSPipeline()
    pipe.train()


def cmd_score(*_):
    from engine.pipeline import EWSPipeline

    pipe = EWSPipeline()
    results = pipe.score()
    _print_summary(results)


def cmd_run(*_):
    from engine.pipeline import EWSPipeline

    pipe = EWSPipeline()
    results = pipe.run()
    _print_summary(results)


def cmd_develop(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.develop(segment=args[0] if args else None)
    print(result["model_version"])


def cmd_retrain(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.retrain(segment=args[0] if args else None)
    print(result["model_version"])


def cmd_tune_weights(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 and args[1] != "-" else None
    apply = None
    if len(args) > 2:
        apply = args[2].strip().lower() in {"1", "true", "yes", "apply"}
    result = manager.tune_weights(segment=segment, model_version=model_version, apply=apply)
    print(result["weight_version"])


def cmd_evaluate_outcomes(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 else None
    result = manager.evaluate_outcomes(segment=segment, model_version=model_version)
    print(result["evaluation_path"])


def cmd_compare(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    segment = args[0] if args else None
    challenger_version = args[1] if len(args) > 1 else None
    result = manager.compare(segment=segment, challenger_version=challenger_version)
    print(result["recommendation"]["winner"])


def cmd_promote(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    segment = args[0] if args else None
    model_version = args[1] if len(args) > 1 else None
    result = manager.promote(segment=segment, model_version=model_version)
    print(result["promoted_model"])


def cmd_score_live(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    segment = None
    snapshot_date = None
    start_date = None
    end_date = None

    if len(args) == 1:
        if _looks_like_date(args[0]):
            snapshot_date = args[0]
        else:
            segment = args[0]
    elif len(args) == 2:
        if _looks_like_date(args[0]) and _looks_like_date(args[1]):
            start_date, end_date = args[0], args[1]
        else:
            segment, snapshot_date = args[0], args[1]
    elif len(args) >= 3:
        segment, start_date, end_date = args[0], args[1], args[2]

    result = manager.score_live(
        segment=segment,
        snapshot_date=snapshot_date,
        start_date=start_date,
        end_date=end_date,
    )
    print(result.get("snapshot_date") or result.get("status"))


def cmd_run_batch(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.run_batch(segment=args[0] if args else None)
    print(result)


def cmd_run_multivar_anomaly(*args):
    input_arg = args[0] if args else "oracle"
    source = "oracle" if input_arg in {"oracle", "-"} else "csv"
    input_path = None if source == "oracle" else input_arg
    scoring_month = args[1] if len(args) > 1 and args[1] != "-" else None
    max_train_rows = int(args[2]) if len(args) > 2 and args[2] != "-" else 150_000
    max_score_rows = int(args[3]) if len(args) > 3 and args[3] != "-" else None
    result = run_multivar_anomaly(
        input_path=input_path,
        source=source,
        scoring_month=scoring_month,
        max_train_rows=max_train_rows,
        max_score_rows=max_score_rows,
    )
    print(json_like_summary(result))


def cmd_load_multivar_oracle(*args):
    input_path = args[0] if args else "anomaly_multivar.csv"
    replace = not (len(args) > 1 and args[1].strip().lower() in {"append", "0", "false", "no"})
    delete_local = len(args) > 2 and args[2].strip().lower() in {"delete-local", "delete", "1", "true", "yes"}
    result = load_multivar_csv_to_oracle(
        input_path=input_path,
        replace=replace,
        delete_local=delete_local,
    )
    print(json_like_summary(result))


def cmd_compare_preprocessing(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.compare_preprocessing(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_compare_feature_selection(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.compare_feature_selection(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_compare_sampling(*args):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.compare_sampling(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_prepare_ticari_orta_faz1_demo(*args):
    from engine.ticari_orta_faz1_demo import prepare_ticari_orta_faz1_demo

    result = prepare_ticari_orta_faz1_demo(segment=args[0] if args else None)
    print(result)


def cmd_run_ticari_orta_faz1_demo(*args):
    from engine.ticari_orta_faz1_demo import run_ticari_orta_faz1_demo

    result = run_ticari_orta_faz1_demo(segment=args[0] if args else None)
    print(result)


def cmd_reset_runtime(*_):
    from engine.lifecycle import LifecycleManager

    manager = LifecycleManager()
    result = manager.reset_runtime()
    print(result)


def cmd_cleanup(*_):
    from engine.lifecycle import LifecycleManager

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
            f"{row['alert_band']} | flags: {row.get('uni_flag_count', 0)}"
        )
        detail_payload = row.get("detay")
        if not isinstance(detail_payload, dict):
            continue
        for _, detail in detail_payload.items():
            ae_ref = detail.get("ae_referansi")
            actual = detail.get("gerceklesen")
            contrib = detail.get("ensemble_katki_pct", detail.get("contribution_pct", 0))
            print(
                f"    {detail['label']}: actual={_fmt_number(actual)} | "
                f"ae_ref={_fmt_number(ae_ref)} | katkı=%{_fmt_pct(contrib)}"
            )


def _fmt_number(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value):
    if value is None:
        return "0"
    try:
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def json_like_summary(result):
    if "inserted_rows" in result:
        return "\n".join(
            [
                f"oracle_table: {result['oracle_table']}",
                f"inserted_rows: {result['inserted_rows']}",
                f"oracle_rows: {result['oracle_rows']}",
                f"deleted_local: {result['deleted_local']}",
            ]
        )
    lines = [
        f"scoring_month: {result['scoring_month']}",
        f"scored_rows: {result['scored_rows']}",
        f"train_rows: {result['train_rows']}",
        f"selected_feature_count: {result['selected_feature_count']}",
        f"alert_counts: {result['alert_counts']}",
        f"alert_type_counts: {result.get('alert_type_counts')}",
        f"review_queue_counts: {result.get('review_queue_counts')}",
        f"scores_path: {result['scores_path']}",
        f"top_path: {result['top_path']}",
    ]
    oracle_output = result.get("oracle_output") or {}
    if oracle_output:
        lines.extend(
            [
                f"oracle_results_table: {oracle_output['results_table']}",
                f"oracle_details_table: {oracle_output['details_table']}",
                f"oracle_inserted_results: {oracle_output['inserted_results']}",
                f"oracle_inserted_details: {oracle_output['inserted_details']}",
            ]
        )
    return "\n".join(lines)


def _looks_like_date(value: str) -> bool:
    text = str(value).strip()
    if len(text) != 10:
        return False
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return False
    return True


COMMANDS = {
    "setup": cmd_setup,
    "train": cmd_train,
    "score": cmd_score,
    "run": cmd_run,
    "develop": cmd_develop,
    "retrain": cmd_retrain,
    "tune-weights": cmd_tune_weights,
    "evaluate-outcomes": cmd_evaluate_outcomes,
    "compare": cmd_compare,
    "promote": cmd_promote,
    "score-live": cmd_score_live,
    "run-batch": cmd_run_batch,
    "load-multivar-oracle": cmd_load_multivar_oracle,
    "run-multivar-anomaly": cmd_run_multivar_anomaly,
    "compare-preprocessing": cmd_compare_preprocessing,
    "compare-feature-selection": cmd_compare_feature_selection,
    "compare-sampling": cmd_compare_sampling,
    "prepare-ticari-orta-faz1-demo": cmd_prepare_ticari_orta_faz1_demo,
    "run-ticari-orta-faz1-demo": cmd_run_ticari_orta_faz1_demo,
    "reset-runtime": cmd_reset_runtime,
    "cleanup": cmd_cleanup,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    config = load_config()
    logs_dir = config.get("registry", {}).get("logs_dir", "logs")
    setup_logging(log_dir=logs_dir, enable_file=cmd != "reset-runtime")

    args = sys.argv[2:]
    logging.info("CLI command: %s %s", cmd, " ".join(args))
    COMMANDS[cmd](*args)
