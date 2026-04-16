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

from engine.config_loader import load_config
from engine.lifecycle import LifecycleManager
from engine.pipeline import EWSPipeline
from engine.ticari_orta_faz1_demo import prepare_ticari_orta_faz1_demo, run_ticari_orta_faz1_demo


def setup_logging(log_dir="logs", level="INFO", enable_file=True):
    Path(log_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if enable_file:
        handlers.append(logging.FileHandler(f"{log_dir}/ews_{ts}.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_setup(*_):
    pipe = EWSPipeline()
    pipe.setup()

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
    print(result["snapshot_date"])


def cmd_run_batch(*args):
    manager = LifecycleManager()
    result = manager.run_batch(segment=args[0] if args else None)
    print(result)


def cmd_compare_preprocessing(*args):
    manager = LifecycleManager()
    result = manager.compare_preprocessing(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_compare_feature_selection(*args):
    manager = LifecycleManager()
    result = manager.compare_feature_selection(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_compare_sampling(*args):
    manager = LifecycleManager()
    result = manager.compare_sampling(segment=args[0] if args else None)
    print(result["comparison_path"])


def cmd_prepare_ticari_orta_faz1_demo(*args):
    result = prepare_ticari_orta_faz1_demo(segment=args[0] if args else None)
    print(result)


def cmd_run_ticari_orta_faz1_demo(*args):
    result = run_ticari_orta_faz1_demo(segment=args[0] if args else None)
    print(result)


def cmd_reset_runtime(*_):
    manager = LifecycleManager()
    result = manager.reset_runtime()
    print(result)


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
