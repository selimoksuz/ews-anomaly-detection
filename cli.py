"""Multivar anomaly detection CLI.

Usage:
    python cli.py load-multivar-oracle [input_path] [replace|append] [delete-local]
    python cli.py run-multivar-anomaly [oracle|input_path|-] [scoring_month|-] [max_train_rows|all|-] [max_score_rows|-]
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from engine.config_loader import load_config, resolve_project_path
from engine.multivar_anomaly import load_multivar_csv_to_oracle, run_multivar_anomaly


def setup_logging(log_dir="runtime/logs/cli", level="INFO", enable_file=True):
    resolved_log_dir = resolve_project_path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if enable_file:
        handlers.append(logging.FileHandler(resolved_log_dir / f"multivar_{ts}.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_run_multivar_anomaly(*args):
    input_arg = args[0] if args else "oracle"
    source = "oracle" if input_arg in {"oracle", "-"} else "csv"
    input_path = None if source == "oracle" else input_arg
    scoring_month = args[1] if len(args) > 1 and args[1] != "-" else None
    max_train_rows = parse_optional_limit(args[2]) if len(args) > 2 else None
    max_score_rows = int(args[3]) if len(args) > 3 and args[3] != "-" else None
    result = run_multivar_anomaly(
        input_path=input_path,
        source=source,
        scoring_month=scoring_month,
        max_train_rows=max_train_rows,
        max_score_rows=max_score_rows,
    )
    print(json_like_summary(result))


def parse_optional_limit(value):
    normalized = str(value).strip().lower()
    if normalized in {"", "-", "all", "none", "null", "0"}:
        return None
    return int(normalized)


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
        f"calibration_rows: {result.get('calibration_rows')}",
        f"selected_feature_count: {result['selected_feature_count']}",
        f"alert_counts: {result['alert_counts']}",
        f"alert_type_counts: {result.get('alert_type_counts')}",
        f"review_queue_counts: {result.get('review_queue_counts')}",
        f"scores_path: {result['scores_path']}",
        f"top_path: {result['top_path']}",
    ]
    peer_diag = result.get("reason_peer_representativeness_diagnostics") or result.get("peer_representativeness_diagnostics") or {}
    if peer_diag:
        lines.extend(
            [
                f"peer_corporate_assessment: {peer_diag.get('corporate_assessment')}",
                f"peer_meaningfulness_test: {(peer_diag.get('meaningfulness_test') or {}).get('result')}",
                f"peer_quality_pct: {peer_diag.get('quality_pct')}",
            ]
        )
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


COMMANDS = {
    "load-multivar-oracle": cmd_load_multivar_oracle,
    "run-multivar-anomaly": cmd_run_multivar_anomaly,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    config = load_config()
    logs_dir = config.get("logging", {}).get("directory", "runtime/logs/cli")
    setup_logging(log_dir=logs_dir, enable_file=True)

    args = sys.argv[2:]
    logging.info("CLI command: %s %s", cmd, " ".join(args))
    COMMANDS[cmd](*args)
