"""Persist LLM anomaly decisions to Oracle."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from engine.config_loader import load_config, load_secrets
from engine.multivar_anomaly import ID_COLUMN, TIME_COLUMN, parse_dates
from engine.oracle_io import OracleConnector


DEFAULT_LLM_RESULTS_TABLE_KEY = "llm_results"
DEFAULT_LLM_REASONS_TABLE_KEY = "llm_reason_details"
DEFAULT_LLM_FEATURES_TABLE_KEY = "llm_feature_details"
DEFAULT_LLM_OUTPUT_WRITE_MODE = "replace"

logger = logging.getLogger(__name__)


def resolve_llm_output_write_mode(config: dict[str, Any], explicit_mode: str | None = None) -> str:
    mode = explicit_mode
    if mode is None:
        mode = (
            ((config.get("llm") or {}).get("outputs") or {})
            .get("oracle", {})
            .get("write_mode")
        )
    mode = str(mode or DEFAULT_LLM_OUTPUT_WRITE_MODE).strip().lower()
    aliases = {
        "delete_insert": "replace",
        "overwrite": "replace",
        "insert": "append",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"replace", "append"}:
        raise ValueError("llm.outputs.oracle.write_mode must be 'replace' or 'append'.")
    return mode

LLM_RESULT_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "is_anomaly",
    "anomaly_type",
    "risk_level",
    "anomaly_score",
    "llm_confidence",
    "seasonality_assessment",
    "trend_assessment",
    "peer_assessment",
    "caveat",
    "ml_anomaly_score",
    "ml_ensemble_score",
    "ml_is_anomaly",
    "ml_alert_band",
    "ml_if_score",
    "ml_residual_score",
    "ml_autoencoder_score",
    "reason_summary",
    "reason_1",
    "reason_1_weight",
    "reason_2",
    "reason_2_weight",
    "reason_3",
    "reason_3_weight",
    "recommended_action",
    "llm_model",
    "evidence_source",
    "raw_response",
]

LLM_REASON_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "reason_rank",
    "feature_name",
    "evidence_text",
    "interpretation",
]

LLM_FEATURE_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "feature_rank",
    "feature_name",
    "feature_label",
    "feature_category",
    "formula",
    "source_columns",
    "risk_direction",
    "current_value",
    "previous_value",
    "change_pct",
    "history_period_count",
    "history_median",
    "history_p25",
    "history_p75",
    "history_robust_scale",
    "history_z",
    "rolling_3m_median",
    "rolling_6m_median",
    "rolling_12m_median",
    "trend_slope_6m",
    "trend_slope_12m",
    "trend_break_flag",
    "trend_note",
    "month_of_year",
    "same_month_last_year_value",
    "yoy_change_pct",
    "same_month_customer_median",
    "same_month_customer_z",
    "seasonal_peer_median",
    "seasonal_peer_z",
    "peer_median",
    "peer_z",
    "peer_support",
    "peer_definition_level",
    "peer_quality",
    "data_missing_flag",
    "snapshot_series_json",
    "feature_json",
]

LLM_RESULT_COLUMN_DDLS = {
    "RUN_ID": "RUN_ID VARCHAR2(64)",
    TIME_COLUMN.upper(): f"{TIME_COLUMN.upper()} DATE",
    ID_COLUMN.upper(): f"{ID_COLUMN.upper()} VARCHAR2(128)",
    "IS_ANOMALY": "IS_ANOMALY NUMBER(1)",
    "ANOMALY_TYPE": "ANOMALY_TYPE VARCHAR2(64)",
    "RISK_LEVEL": "RISK_LEVEL VARCHAR2(32)",
    "ANOMALY_SCORE": "ANOMALY_SCORE NUMBER(6,4)",
    "LLM_CONFIDENCE": "LLM_CONFIDENCE NUMBER(6,4)",
    "SEASONALITY_ASSESSMENT": "SEASONALITY_ASSESSMENT VARCHAR2(2000)",
    "TREND_ASSESSMENT": "TREND_ASSESSMENT VARCHAR2(2000)",
    "PEER_ASSESSMENT": "PEER_ASSESSMENT VARCHAR2(2000)",
    "CAVEAT": "CAVEAT VARCHAR2(2000)",
    "ML_ANOMALY_SCORE": "ML_ANOMALY_SCORE NUMBER(6,2)",
    "ML_ENSEMBLE_SCORE": "ML_ENSEMBLE_SCORE NUMBER(6,2)",
    "ML_IS_ANOMALY": "ML_IS_ANOMALY NUMBER(1)",
    "ML_ALERT_BAND": "ML_ALERT_BAND VARCHAR2(32)",
    "ML_IF_SCORE": "ML_IF_SCORE NUMBER(6,2)",
    "ML_RESIDUAL_SCORE": "ML_RESIDUAL_SCORE NUMBER(6,2)",
    "ML_AUTOENCODER_SCORE": "ML_AUTOENCODER_SCORE NUMBER(6,2)",
    "REASON_SUMMARY": "REASON_SUMMARY VARCHAR2(4000)",
    "REASON_1": "REASON_1 VARCHAR2(2000)",
    "REASON_1_WEIGHT": "REASON_1_WEIGHT NUMBER(6,4)",
    "REASON_2": "REASON_2 VARCHAR2(2000)",
    "REASON_2_WEIGHT": "REASON_2_WEIGHT NUMBER(6,4)",
    "REASON_3": "REASON_3 VARCHAR2(2000)",
    "REASON_3_WEIGHT": "REASON_3_WEIGHT NUMBER(6,4)",
    "RECOMMENDED_ACTION": "RECOMMENDED_ACTION VARCHAR2(128)",
    "LLM_MODEL": "LLM_MODEL VARCHAR2(128)",
    "EVIDENCE_SOURCE": "EVIDENCE_SOURCE VARCHAR2(64)",
    "RAW_RESPONSE": "RAW_RESPONSE CLOB",
    "CREATED_AT": "CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP",
}

LLM_REASON_COLUMN_DDLS = {
    "RUN_ID": "RUN_ID VARCHAR2(64)",
    TIME_COLUMN.upper(): f"{TIME_COLUMN.upper()} DATE",
    ID_COLUMN.upper(): f"{ID_COLUMN.upper()} VARCHAR2(128)",
    "REASON_RANK": "REASON_RANK NUMBER(4)",
    "FEATURE_NAME": "FEATURE_NAME VARCHAR2(128)",
    "EVIDENCE_TEXT": "EVIDENCE_TEXT VARCHAR2(2000)",
    "INTERPRETATION": "INTERPRETATION VARCHAR2(2000)",
    "CREATED_AT": "CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP",
}

LLM_FEATURE_COLUMN_DDLS = {
    "RUN_ID": "RUN_ID VARCHAR2(64)",
    TIME_COLUMN.upper(): f"{TIME_COLUMN.upper()} DATE",
    ID_COLUMN.upper(): f"{ID_COLUMN.upper()} VARCHAR2(128)",
    "FEATURE_RANK": "FEATURE_RANK NUMBER(4)",
    "FEATURE_NAME": "FEATURE_NAME VARCHAR2(128)",
    "FEATURE_LABEL": "FEATURE_LABEL VARCHAR2(256)",
    "FEATURE_CATEGORY": "FEATURE_CATEGORY VARCHAR2(64)",
    "FORMULA": "FORMULA VARCHAR2(1000)",
    "SOURCE_COLUMNS": "SOURCE_COLUMNS VARCHAR2(1000)",
    "RISK_DIRECTION": "RISK_DIRECTION VARCHAR2(32)",
    "CURRENT_VALUE": "CURRENT_VALUE NUMBER(24,8)",
    "PREVIOUS_VALUE": "PREVIOUS_VALUE NUMBER(24,8)",
    "CHANGE_PCT": "CHANGE_PCT NUMBER(18,6)",
    "HISTORY_PERIOD_COUNT": "HISTORY_PERIOD_COUNT NUMBER(10)",
    "HISTORY_MEDIAN": "HISTORY_MEDIAN NUMBER(24,8)",
    "HISTORY_P25": "HISTORY_P25 NUMBER(24,8)",
    "HISTORY_P75": "HISTORY_P75 NUMBER(24,8)",
    "HISTORY_ROBUST_SCALE": "HISTORY_ROBUST_SCALE NUMBER(24,8)",
    "HISTORY_Z": "HISTORY_Z NUMBER(18,6)",
    "ROLLING_3M_MEDIAN": "ROLLING_3M_MEDIAN NUMBER(24,8)",
    "ROLLING_6M_MEDIAN": "ROLLING_6M_MEDIAN NUMBER(24,8)",
    "ROLLING_12M_MEDIAN": "ROLLING_12M_MEDIAN NUMBER(24,8)",
    "TREND_SLOPE_6M": "TREND_SLOPE_6M NUMBER(24,8)",
    "TREND_SLOPE_12M": "TREND_SLOPE_12M NUMBER(24,8)",
    "TREND_BREAK_FLAG": "TREND_BREAK_FLAG NUMBER(1)",
    "TREND_NOTE": "TREND_NOTE VARCHAR2(1000)",
    "MONTH_OF_YEAR": "MONTH_OF_YEAR NUMBER(2)",
    "SAME_MONTH_LAST_YEAR_VALUE": "SAME_MONTH_LAST_YEAR_VALUE NUMBER(24,8)",
    "YOY_CHANGE_PCT": "YOY_CHANGE_PCT NUMBER(18,6)",
    "SAME_MONTH_CUSTOMER_MEDIAN": "SAME_MONTH_CUSTOMER_MEDIAN NUMBER(24,8)",
    "SAME_MONTH_CUSTOMER_Z": "SAME_MONTH_CUSTOMER_Z NUMBER(18,6)",
    "SEASONAL_PEER_MEDIAN": "SEASONAL_PEER_MEDIAN NUMBER(24,8)",
    "SEASONAL_PEER_Z": "SEASONAL_PEER_Z NUMBER(18,6)",
    "PEER_MEDIAN": "PEER_MEDIAN NUMBER(24,8)",
    "PEER_Z": "PEER_Z NUMBER(18,6)",
    "PEER_SUPPORT": "PEER_SUPPORT NUMBER(10)",
    "PEER_DEFINITION_LEVEL": "PEER_DEFINITION_LEVEL VARCHAR2(64)",
    "PEER_QUALITY": "PEER_QUALITY VARCHAR2(32)",
    "DATA_MISSING_FLAG": "DATA_MISSING_FLAG NUMBER(1)",
    "SNAPSHOT_SERIES_JSON": "SNAPSHOT_SERIES_JSON CLOB",
    "FEATURE_JSON": "FEATURE_JSON CLOB",
    "CREATED_AT": "CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP",
}


def write_llm_outputs_to_oracle(
    decisions: list[dict[str, Any]],
    *,
    llm_model: str,
    evidence_source: str = "oracle_input",
    results_table_key: str = DEFAULT_LLM_RESULTS_TABLE_KEY,
    reasons_table_key: str = DEFAULT_LLM_REASONS_TABLE_KEY,
    features_table_key: str = DEFAULT_LLM_FEATURES_TABLE_KEY,
    batch_size: int = 1000,
    write_mode: str | None = None,
) -> dict[str, Any]:
    if not decisions:
        logger.info("No LLM decisions to persist to Oracle.")
        return {
            "backend": "oracle",
            "run_id": None,
            "inserted_results": 0,
            "inserted_reasons": 0,
            "inserted_features": 0,
            "deleted_results": 0,
            "deleted_reasons": 0,
            "deleted_features": 0,
        }

    scoring_month = first_scoring_month(decisions)
    config = load_config()
    write_mode = resolve_llm_output_write_mode(config, write_mode)
    run_id = f"llm_{scoring_month.strftime('%Y%m%d')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result_frame = prepare_llm_result_frame(
        decisions,
        run_id=run_id,
        llm_model=llm_model,
        evidence_source=evidence_source,
    )
    reason_frame = prepare_llm_reason_frame(decisions, run_id=run_id)
    feature_frame = prepare_llm_feature_frame(decisions, run_id=run_id)
    logger.info(
        "Prepared LLM Oracle output frames: run_id=%s result_rows=%s reason_rows=%s feature_rows=%s",
        run_id,
        len(result_frame),
        len(reason_frame),
        len(feature_frame),
    )
    logger.info(
        "AUDIT STRUCTURED OUTPUT CONTRACT | results_columns=%s reasons_columns=%s features_columns=%s",
        ",".join(LLM_RESULT_COLUMNS),
        ",".join(LLM_REASON_COLUMNS),
        ",".join(LLM_FEATURE_COLUMNS),
    )

    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        logger.info(
            "Ensuring LLM Oracle output tables: results=%s reasons=%s features=%s",
            ora._qualified_table_name(results_table_key),
            ora._qualified_table_name(reasons_table_key),
            ora._qualified_table_name(features_table_key),
        )
        ensure_llm_output_tables(
            ora,
            results_table_key=results_table_key,
            reasons_table_key=reasons_table_key,
            features_table_key=features_table_key,
        )
        if write_mode == "replace":
            logger.info("Deleting previous LLM output rows for scoring_month=%s write_mode=replace", scoring_month.date())
            deleted = delete_llm_output_month(
                ora,
                scoring_month=scoring_month,
                results_table_key=results_table_key,
                reasons_table_key=reasons_table_key,
                features_table_key=features_table_key,
            )
            logger.info(
                "Deleted old LLM rows: results=%s reasons=%s features=%s",
                deleted["results"],
                deleted["reasons"],
                deleted["features"],
            )
        else:
            deleted = {"results": 0, "reasons": 0, "features": 0}
            logger.info(
                "Skipping delete of previous LLM output rows: scoring_month=%s write_mode=append",
                scoring_month.date(),
            )
        logger.info("Inserting LLM result rows: rows=%s", len(result_frame))
        inserted_results = insert_frame(
            ora,
            results_table_key,
            result_frame,
            batch_size=batch_size,
        )
        logger.info("Inserting LLM reason rows: rows=%s", len(reason_frame))
        inserted_reasons = insert_frame(
            ora,
            reasons_table_key,
            reason_frame,
            batch_size=batch_size,
        )
        logger.info("Inserting LLM feature detail rows: rows=%s", len(feature_frame))
        inserted_features = insert_frame(
            ora,
            features_table_key,
            feature_frame,
            batch_size=batch_size,
        )
        result_counts = count_llm_output_rows(
            ora,
            results_table_key,
            scoring_month=scoring_month,
            run_id=run_id,
        )
        reason_counts = count_llm_output_rows(
            ora,
            reasons_table_key,
            scoring_month=scoring_month,
            run_id=run_id,
        )
        feature_counts = count_llm_output_rows(
            ora,
            features_table_key,
            scoring_month=scoring_month,
            run_id=run_id,
        )
        logger.info(
            "Inserted LLM Oracle rows: results=%s reasons=%s features=%s",
            inserted_results,
            inserted_reasons,
            inserted_features,
        )
        logger.info(
            "AUDIT OUTPUT TABLE | table_key=%s table=%s inserted=%s total_rows_after=%s scoring_month_rows_after=%s run_rows_after=%s",
            results_table_key,
            ora._qualified_table_name(results_table_key),
            inserted_results,
            result_counts["total_rows"],
            result_counts["scoring_month_rows"],
            result_counts["run_rows"],
        )
        logger.info(
            "AUDIT OUTPUT TABLE | table_key=%s table=%s inserted=%s total_rows_after=%s scoring_month_rows_after=%s run_rows_after=%s",
            reasons_table_key,
            ora._qualified_table_name(reasons_table_key),
            inserted_reasons,
            reason_counts["total_rows"],
            reason_counts["scoring_month_rows"],
            reason_counts["run_rows"],
        )
        logger.info(
            "AUDIT OUTPUT TABLE | table_key=%s table=%s inserted=%s total_rows_after=%s scoring_month_rows_after=%s run_rows_after=%s",
            features_table_key,
            ora._qualified_table_name(features_table_key),
            inserted_features,
            feature_counts["total_rows"],
            feature_counts["scoring_month_rows"],
            feature_counts["run_rows"],
        )
        return {
            "backend": "oracle",
            "run_id": run_id,
            "write_mode": write_mode,
            "results_table_key": results_table_key,
            "reasons_table_key": reasons_table_key,
            "features_table_key": features_table_key,
            "results_table": ora._qualified_table_name(results_table_key),
            "reasons_table": ora._qualified_table_name(reasons_table_key),
            "features_table": ora._qualified_table_name(features_table_key),
            "deleted_results": int(deleted["results"]),
            "deleted_reasons": int(deleted["reasons"]),
            "deleted_features": int(deleted["features"]),
            "inserted_results": int(inserted_results),
            "inserted_reasons": int(inserted_reasons),
            "inserted_features": int(inserted_features),
            "result_table_total_rows_after": int(result_counts["total_rows"]),
            "result_table_scoring_month_rows_after": int(result_counts["scoring_month_rows"]),
            "result_table_run_rows_after": int(result_counts["run_rows"]),
            "reason_table_total_rows_after": int(reason_counts["total_rows"]),
            "reason_table_scoring_month_rows_after": int(reason_counts["scoring_month_rows"]),
            "reason_table_run_rows_after": int(reason_counts["run_rows"]),
            "feature_table_total_rows_after": int(feature_counts["total_rows"]),
            "feature_table_scoring_month_rows_after": int(feature_counts["scoring_month_rows"]),
            "feature_table_run_rows_after": int(feature_counts["run_rows"]),
        }


def audit_llm_output_tables(
    scoring_month,
    *,
    results_table_key: str = DEFAULT_LLM_RESULTS_TABLE_KEY,
    reasons_table_key: str = DEFAULT_LLM_REASONS_TABLE_KEY,
    features_table_key: str = DEFAULT_LLM_FEATURES_TABLE_KEY,
) -> dict[str, Any]:
    scoring_month = pd.Timestamp(scoring_month).normalize()
    config = load_config()
    secrets = load_secrets()
    audit: dict[str, Any] = {}
    with OracleConnector(config, secrets) as ora:
        for label, table_key in (
            ("results", results_table_key),
            ("reasons", reasons_table_key),
            ("features", features_table_key),
        ):
            table_name = ora._qualified_table_name(table_key)
            exists = ora._table_exists(table_key)
            counts = (
                count_llm_output_rows(ora, table_key, scoring_month=scoring_month, run_id=None)
                if exists
                else {"total_rows": 0, "scoring_month_rows": 0, "run_rows": 0}
            )
            audit[label] = {
                "table_key": table_key,
                "table": table_name,
                "exists": bool(exists),
                **counts,
            }
            logger.info(
                "AUDIT OUTPUT TABLE | label=%s table_key=%s table=%s exists=%s total_rows=%s scoring_month=%s scoring_month_rows=%s",
                label,
                table_key,
                table_name,
                exists,
                counts["total_rows"],
                scoring_month.date(),
                counts["scoring_month_rows"],
            )
    return audit


def ensure_llm_output_tables_in_oracle(
    *,
    scoring_month=None,
    results_table_key: str = DEFAULT_LLM_RESULTS_TABLE_KEY,
    reasons_table_key: str = DEFAULT_LLM_REASONS_TABLE_KEY,
    features_table_key: str = DEFAULT_LLM_FEATURES_TABLE_KEY,
) -> dict[str, Any]:
    month = pd.Timestamp(scoring_month).normalize() if scoring_month else None
    config = load_config()
    secrets = load_secrets()
    audit: dict[str, Any] = {}
    with OracleConnector(config, secrets) as ora:
        ensure_llm_output_tables(
            ora,
            results_table_key=results_table_key,
            reasons_table_key=reasons_table_key,
            features_table_key=features_table_key,
        )
        for label, table_key in (
            ("results", results_table_key),
            ("reasons", reasons_table_key),
            ("features", features_table_key),
        ):
            table_name = ora._qualified_table_name(table_key)
            exists = ora._table_exists(table_key)
            total_rows = count_table_rows(ora, table_key) if exists else 0
            scoring_month_rows = (
                count_llm_output_rows(ora, table_key, scoring_month=month, run_id=None)["scoring_month_rows"]
                if exists and month is not None
                else None
            )
            audit[label] = {
                "table_key": table_key,
                "table": table_name,
                "exists": bool(exists),
                "total_rows": int(total_rows),
                "scoring_month": month.strftime("%Y-%m-%d") if month is not None else None,
                "scoring_month_rows": scoring_month_rows,
            }
            logger.info(
                "AUDIT OUTPUT TABLE ENSURE | label=%s table_key=%s table=%s exists=%s total_rows=%s scoring_month=%s scoring_month_rows=%s",
                label,
                table_key,
                table_name,
                exists,
                total_rows,
                month.date() if month is not None else None,
                scoring_month_rows,
            )
    return audit


def count_table_rows(ora: OracleConnector, table_key: str) -> int:
    table_name = ora._qualified_table_name(table_key)
    frame = ora._read_query(f"SELECT COUNT(*) AS ROW_COUNT FROM {table_name}")
    return int(frame.iloc[0]["row_count"])


def count_llm_output_rows(
    ora: OracleConnector,
    table_key: str,
    *,
    scoring_month: pd.Timestamp,
    run_id: str | None,
) -> dict[str, int]:
    table_name = ora._qualified_table_name(table_key)
    params = {"scoring_month": pd.Timestamp(scoring_month).to_pydatetime()}
    total_frame = ora._read_query(f"SELECT COUNT(*) AS ROW_COUNT FROM {table_name}")
    month_frame = ora._read_query(
        f"""
        SELECT COUNT(*) AS ROW_COUNT
        FROM {table_name}
        WHERE TRUNC({TIME_COLUMN.upper()}) = TRUNC(:scoring_month)
        """,
        params,
    )
    run_rows = 0
    if run_id:
        run_params = dict(params)
        run_params["run_id"] = run_id
        run_frame = ora._read_query(
            f"""
            SELECT COUNT(*) AS ROW_COUNT
            FROM {table_name}
            WHERE TRUNC({TIME_COLUMN.upper()}) = TRUNC(:scoring_month)
              AND RUN_ID = :run_id
            """,
            run_params,
        )
        run_rows = int(run_frame.iloc[0]["row_count"])
    return {
        "total_rows": int(total_frame.iloc[0]["row_count"]),
        "scoring_month_rows": int(month_frame.iloc[0]["row_count"]),
        "run_rows": int(run_rows),
    }


def ensure_llm_output_tables(
    ora: OracleConnector,
    *,
    results_table_key: str,
    reasons_table_key: str,
    features_table_key: str,
) -> None:
    ddls = {
        results_table_key: llm_results_table_ddl(ora, results_table_key),
        reasons_table_key: llm_reasons_table_ddl(ora, reasons_table_key),
        features_table_key: llm_features_table_ddl(ora, features_table_key),
    }
    with ora.connection.cursor() as cursor:
        for table_key, ddl in ddls.items():
            if ora._table_exists(table_key):
                continue
            cursor.execute(ddl)
            ora.logger.info("Created %s", ora._qualified_table_name(table_key))
    ora.connection.commit()
    ensure_missing_columns(ora, results_table_key, LLM_RESULT_COLUMN_DDLS)
    ensure_missing_columns(ora, reasons_table_key, LLM_REASON_COLUMN_DDLS)
    ensure_missing_columns(ora, features_table_key, LLM_FEATURE_COLUMN_DDLS)


def ensure_missing_columns(ora: OracleConnector, table_key: str, column_ddls: dict[str, str]) -> None:
    existing = ora._table_columns(table_key)
    missing = [(column, ddl) for column, ddl in column_ddls.items() if column not in existing]
    if not missing:
        return
    with ora.connection.cursor() as cursor:
        for column, ddl in missing:
            cursor.execute(f"ALTER TABLE {ora._qualified_table_name(table_key)} ADD ({ddl})")
            ora.logger.info("Added missing LLM output column: table=%s column=%s", ora._qualified_table_name(table_key), column)
    ora.connection.commit()


def delete_llm_output_month(
    ora: OracleConnector,
    *,
    scoring_month: pd.Timestamp,
    results_table_key: str,
    reasons_table_key: str,
    features_table_key: str,
) -> dict[str, int]:
    deleted = {"results": 0, "reasons": 0, "features": 0}
    params = {"scoring_month": pd.Timestamp(scoring_month).to_pydatetime()}
    with ora.connection.cursor() as cursor:
        for label, table_key in (
            ("features", features_table_key),
            ("reasons", reasons_table_key),
            ("results", results_table_key),
        ):
            cursor.execute(
                f"""
                DELETE FROM {ora._qualified_table_name(table_key)}
                WHERE TRUNC({TIME_COLUMN.upper()}) = TRUNC(:scoring_month)
                """,
                params,
            )
            deleted[label] = int(cursor.rowcount or 0)
    ora.connection.commit()
    return deleted


def insert_frame(
    ora: OracleConnector,
    table_key: str,
    frame: pd.DataFrame,
    *,
    batch_size: int,
) -> int:
    if frame.empty:
        return 0
    columns = list(frame.columns)
    sql = f"""
        INSERT INTO {ora._qualified_table_name(table_key)} (
            {", ".join(column.upper() for column in columns)}
        ) VALUES (
            {", ".join(f":{index}" for index in range(1, len(columns) + 1))}
        )
    """
    rows = [
        ora._coerce_scalar_sequence(row)
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    return ora._executemany(sql, rows, batch_size=batch_size)


def prepare_llm_feature_frame(decisions: list[dict[str, Any]], *, run_id: str) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        scoring_date = parse_single_date(decision.get("cohort_dt"))
        mono_id = text_or_none(decision.get("mono_id"), 128)
        features = decision.get("evidence_features") or []
        if not isinstance(features, list):
            continue
        for rank, feature in enumerate(features, start=1):
            if not isinstance(feature, dict):
                continue
            dictionary = feature.get("dictionary") or {}
            history = feature.get("history") or {}
            trend = feature.get("trend") or {}
            seasonality = feature.get("seasonality") or {}
            peer = feature.get("peer") or {}
            data_quality = feature.get("data_quality") or {}
            current_value = number_or_none(feature.get("current_value"))
            history_median = number_or_none(history.get("median"))
            history_robust_scale = number_or_none(history.get("robust_scale"))
            rows.append(
                {
                    "run_id": text_or_none(run_id, 64),
                    TIME_COLUMN: scoring_date,
                    ID_COLUMN: mono_id,
                    "feature_rank": rank,
                    "feature_name": text_or_none(feature.get("name"), 128),
                    "feature_label": text_or_none(dictionary.get("label"), 256),
                    "feature_category": text_or_none(dictionary.get("category"), 64),
                    "formula": text_or_none(dictionary.get("formula"), 1000),
                    "source_columns": text_or_none(json.dumps(dictionary.get("source_columns"), ensure_ascii=False), 1000),
                    "risk_direction": text_or_none(dictionary.get("risk_direction"), 32),
                    "current_value": current_value,
                    "previous_value": number_or_none(feature.get("previous_value")),
                    "change_pct": number_or_none(feature.get("change_pct")),
                    "history_period_count": number_or_none(history.get("period_count")),
                    "history_median": history_median,
                    "history_p25": number_or_none(history.get("p25")),
                    "history_p75": number_or_none(history.get("p75")),
                    "history_robust_scale": history_robust_scale,
                    "history_z": robust_z_value(current_value, history_median, history_robust_scale),
                    "rolling_3m_median": number_or_none(history.get("rolling_3m_median")),
                    "rolling_6m_median": number_or_none(history.get("rolling_6m_median")),
                    "rolling_12m_median": number_or_none(history.get("rolling_12m_median")),
                    "trend_slope_6m": number_or_none(trend.get("slope_6m")),
                    "trend_slope_12m": number_or_none(trend.get("slope_12m")),
                    "trend_break_flag": bool_to_number_or_none(trend.get("trend_break_flag")),
                    "trend_note": text_or_none(trend.get("trend_note"), 1000),
                    "month_of_year": number_or_none(seasonality.get("month_of_year")),
                    "same_month_last_year_value": number_or_none(seasonality.get("same_month_last_year_value")),
                    "yoy_change_pct": number_or_none(seasonality.get("yoy_change_pct")),
                    "same_month_customer_median": number_or_none(seasonality.get("same_month_customer_median")),
                    "same_month_customer_z": number_or_none(seasonality.get("same_month_customer_z")),
                    "seasonal_peer_median": number_or_none(seasonality.get("seasonal_peer_median")),
                    "seasonal_peer_z": number_or_none(seasonality.get("seasonal_peer_z")),
                    "peer_median": number_or_none(peer.get("peer_median")),
                    "peer_z": number_or_none(peer.get("peer_z")),
                    "peer_support": number_or_none(peer.get("peer_support")),
                    "peer_definition_level": text_or_none(peer.get("peer_definition_level"), 64),
                    "peer_quality": text_or_none(peer.get("peer_quality"), 32),
                    "data_missing_flag": bool_to_number_or_none(data_quality.get("missing_flag")),
                    "snapshot_series_json": json.dumps(feature.get("snapshot_series") or {}, ensure_ascii=False),
                    "feature_json": json.dumps(feature, ensure_ascii=False),
                }
            )
    if not rows:
        return pd.DataFrame(columns=LLM_FEATURE_COLUMNS)
    return pd.DataFrame(rows, columns=LLM_FEATURE_COLUMNS)


def llm_confidence_value(decision: dict[str, Any]) -> Any:
    """Backward-compatible confidence column.

    The current contract uses anomaly_score as the decision score. If an older
    table has LLM_CONFIDENCE, keep it populated with the same normalized score
    unless the model explicitly returned a separate value.
    """

    value = decision.get("llm_confidence")
    if value is None:
        value = decision.get("confidence")
    if value is None:
        value = decision.get("anomaly_score")
    return value


def seasonality_assessment(decision: dict[str, Any]) -> str:
    explicit = text_or_empty(decision.get("seasonality_assessment") or decision.get("seasonality_assesment"))
    if explicit:
        return explicit
    feature = top_feature_by_metric(
        decision,
        lambda item: max_abs_number(
            ((item.get("seasonality") or {}).get("same_month_customer_z")),
            ((item.get("seasonality") or {}).get("seasonal_peer_z")),
            safe_divide_abs(((item.get("seasonality") or {}).get("yoy_change_pct")), 100.0),
        ),
    )
    if feature is None:
        return "Sezon: yeterli sezon metrigi yok; karar sezon kanitina dayanmiyor."
    seasonality = feature.get("seasonality") or {}
    return compact_assessment(
        "Sezon",
        feature,
        [
            ("month", seasonality.get("month_of_year")),
            ("same_month_last_year", seasonality.get("same_month_last_year_value")),
            ("yoy_change_pct", seasonality.get("yoy_change_pct")),
            ("same_month_z", seasonality.get("same_month_customer_z")),
            ("seasonal_peer_median", seasonality.get("seasonal_peer_median")),
            ("seasonal_peer_z", seasonality.get("seasonal_peer_z")),
            ("note", seasonality.get("seasonality_note")),
        ],
    )


def trend_assessment(decision: dict[str, Any]) -> str:
    explicit = text_or_empty(decision.get("trend_assessment") or decision.get("trend_assesment"))
    if explicit:
        return explicit
    feature = top_feature_by_metric(
        decision,
        lambda item: max_abs_number(
            ((item.get("trend") or {}).get("slope_6m")),
            ((item.get("trend") or {}).get("slope_12m")),
            10.0 if bool((item.get("trend") or {}).get("trend_break_flag")) else None,
        ),
    )
    if feature is None:
        return "Trend: yeterli trend metrigi yok; karar trend kanitina dayanmiyor."
    trend = feature.get("trend") or {}
    return compact_assessment(
        "Trend",
        feature,
        [
            ("slope_6m", trend.get("slope_6m")),
            ("slope_12m", trend.get("slope_12m")),
            ("trend_break", trend.get("trend_break_flag")),
            ("note", trend.get("trend_note")),
        ],
    )


def peer_assessment(decision: dict[str, Any]) -> str:
    explicit = text_or_empty(decision.get("peer_assessment") or decision.get("peer_assesment"))
    if explicit:
        return explicit
    feature = top_feature_by_metric(
        decision,
        lambda item: max_abs_number(((item.get("peer") or {}).get("peer_z"))),
    )
    if feature is None:
        return "Peer: yeterli peer metrigi yok; karar peer kanitina dayanmiyor."
    peer = feature.get("peer") or {}
    return compact_assessment(
        "Peer",
        feature,
        [
            ("peer_median", peer.get("peer_median")),
            ("peer_z", peer.get("peer_z")),
            ("peer_support", peer.get("peer_support")),
            ("peer_quality", peer.get("peer_quality")),
            ("peer_level", peer.get("peer_definition_level")),
        ],
    )


def caveat_assessment(decision: dict[str, Any]) -> str:
    explicit = text_or_empty(decision.get("caveat"))
    if explicit:
        return explicit
    data_quality = decision.get("evidence_data_quality") or {}
    parts = []
    add_assessment_text(parts, "note", data_quality.get("caveat"))
    add_assessment_number(parts, "coverage_ratio", data_quality.get("coverage_ratio"))
    add_assessment_number(parts, "missing_feature_count", data_quality.get("missing_feature_count"))
    add_assessment_number(parts, "customer_history_periods", data_quality.get("customer_history_periods"))
    features = evidence_features(decision)
    weak_peer_count = sum(
        1
        for feature in features
        if text_or_empty((feature.get("peer") or {}).get("peer_quality")).upper() == "ZAYIF"
    )
    if weak_peer_count:
        parts.append(f"weak_peer_features={weak_peer_count}")
    if not parts:
        return "Caveat: belirgin veri kisiti yok; yorum mevcut evidence kapsami icin gecerlidir."
    return "Caveat: " + "; ".join(parts)


def evidence_features(decision: dict[str, Any]) -> list[dict[str, Any]]:
    features = decision.get("evidence_features") or []
    if not isinstance(features, list):
        return []
    return [feature for feature in features if isinstance(feature, dict)]


def top_feature_by_metric(decision: dict[str, Any], scorer) -> dict[str, Any] | None:
    best_feature = None
    best_score = None
    for feature in evidence_features(decision):
        score = number_or_none(scorer(feature))
        if score is None:
            continue
        score = abs(score)
        if best_score is None or score > best_score:
            best_score = score
            best_feature = feature
    return best_feature


def compact_assessment(prefix: str, feature: dict[str, Any], fields: list[tuple[str, Any]]) -> str:
    parts = [f"feature={feature_display_name(feature)}"]
    for key, value in fields:
        if isinstance(value, bool):
            parts.append(f"{key}={int(value)}")
        elif isinstance(value, str):
            add_assessment_text(parts, key, value)
        else:
            add_assessment_number(parts, key, value)
    return f"{prefix}: " + "; ".join(parts)


def feature_display_name(feature: dict[str, Any]) -> str:
    dictionary = feature.get("dictionary") or {}
    return text_or_empty(dictionary.get("label")) or text_or_empty(feature.get("name")) or "degisken"


def add_assessment_number(parts: list[str], key: str, value: Any) -> None:
    number = number_or_none(value)
    if number is not None:
        parts.append(f"{key}={format_assessment_number(number)}")


def add_assessment_text(parts: list[str], key: str, value: Any) -> None:
    text = text_or_empty(value)
    if text:
        parts.append(f"{key}={text[:500]}")


def format_assessment_number(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.6g}"


def max_abs_number(*values: Any) -> float | None:
    numbers = [abs(number) for number in (number_or_none(value) for value in values) if number is not None]
    if not numbers:
        return None
    return max(numbers)


def safe_divide_abs(value: Any, denominator: float) -> float | None:
    number = number_or_none(value)
    if number is None or denominator == 0:
        return None
    return abs(number / denominator)


def prepare_llm_result_frame(
    decisions: list[dict[str, Any]],
    *,
    run_id: str,
    llm_model: str,
    evidence_source: str,
) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        ml_ensemble_score = decision.get("ml_ensemble_score")
        if ml_ensemble_score is None:
            ml_ensemble_score = decision.get("ml_anomaly_score")
        rows.append(
            {
                "run_id": text_or_none(run_id, 64),
                TIME_COLUMN: parse_single_date(decision.get("cohort_dt")),
                ID_COLUMN: text_or_none(decision.get("mono_id"), 128),
                "is_anomaly": 1 if bool(decision.get("is_anomaly")) else 0,
                "anomaly_type": text_or_none(decision.get("anomaly_type"), 64),
                "risk_level": text_or_none(decision.get("risk_level"), 32),
                "anomaly_score": number_or_none(decision.get("anomaly_score")),
                "llm_confidence": number_or_none(llm_confidence_value(decision)),
                "seasonality_assessment": text_or_none(seasonality_assessment(decision), 2000),
                "trend_assessment": text_or_none(trend_assessment(decision), 2000),
                "peer_assessment": text_or_none(peer_assessment(decision), 2000),
                "caveat": text_or_none(caveat_assessment(decision), 2000),
                "ml_anomaly_score": number_or_none(decision.get("ml_anomaly_score")),
                "ml_ensemble_score": number_or_none(ml_ensemble_score),
                "ml_is_anomaly": bool_to_number_or_none(decision.get("ml_is_anomaly")),
                "ml_alert_band": text_or_none(decision.get("ml_alert_band"), 32),
                "ml_if_score": number_or_none(decision.get("ml_if_score")),
                "ml_residual_score": number_or_none(decision.get("ml_residual_score")),
                "ml_autoencoder_score": number_or_none(decision.get("ml_autoencoder_score")),
                "reason_summary": text_or_none(decision.get("reason_summary"), 4000),
                "reason_1": text_or_none(decision.get("reason_1"), 2000),
                "reason_1_weight": number_or_none(decision.get("reason_1_weight")),
                "reason_2": text_or_none(decision.get("reason_2"), 2000),
                "reason_2_weight": number_or_none(decision.get("reason_2_weight")),
                "reason_3": text_or_none(decision.get("reason_3"), 2000),
                "reason_3_weight": number_or_none(decision.get("reason_3_weight")),
                "recommended_action": text_or_none(decision.get("recommended_action"), 128),
                "llm_model": text_or_none(llm_model, 128),
                "evidence_source": text_or_none(evidence_source, 64),
                "raw_response": json.dumps(result_raw_payload(decision), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=LLM_RESULT_COLUMNS)


def prepare_llm_reason_frame(decisions: list[dict[str, Any]], *, run_id: str) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        reasons = decision.get("main_reasons") or []
        if not isinstance(reasons, list):
            reasons = []
        for rank, reason in enumerate(reasons, start=1):
            if not isinstance(reason, dict):
                continue
            rows.append(
                {
                    "run_id": text_or_none(run_id, 64),
                    TIME_COLUMN: parse_single_date(decision.get("cohort_dt")),
                    ID_COLUMN: text_or_none(decision.get("mono_id"), 128),
                    "reason_rank": rank,
                    "feature_name": text_or_none(reason.get("feature"), 128),
                    "evidence_text": text_or_none(reason.get("evidence"), 2000),
                    "interpretation": text_or_none(reason.get("interpretation"), 2000),
                }
            )
    return pd.DataFrame(rows, columns=LLM_REASON_COLUMNS)


def first_scoring_month(decisions: list[dict[str, Any]]) -> pd.Timestamp:
    for decision in decisions:
        value = decision.get("cohort_dt")
        if value:
            return pd.Timestamp(value).normalize()
    raise ValueError("No cohort_dt found in LLM decisions.")


def parse_single_date(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    parsed = parse_dates(pd.Series([value])).iloc[0]
    return None if pd.isna(parsed) else parsed


def number_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_to_number_or_none(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "evet", "anomaly", "anomali"}:
            return 1
        if normalized in {"0", "false", "no", "n", "hayir", "normal"}:
            return 0
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return 1 if bool(value) else 0


def robust_z_value(current: float | None, median: float | None, scale: float | None) -> float | None:
    if current is None or median is None or scale is None or scale <= 0:
        return None
    return float((current - median) / scale)


def result_raw_payload(decision: dict[str, Any]) -> dict[str, Any]:
    payload = dict(decision)
    payload.pop("evidence_features", None)
    payload.pop("evidence_data_quality", None)
    payload.pop("evidence_peer_definition", None)
    payload.pop("_reason_numeric_evidence", None)
    return payload


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def text_or_none(value, limit: int) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def llm_results_table_ddl(ora: OracleConnector, table_key: str) -> str:
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            RUN_ID VARCHAR2(64) NOT NULL,
            {TIME_COLUMN.upper()} DATE NOT NULL,
            {ID_COLUMN.upper()} VARCHAR2(128) NOT NULL,
            IS_ANOMALY NUMBER(1) NOT NULL,
            ANOMALY_TYPE VARCHAR2(64),
            RISK_LEVEL VARCHAR2(32),
            ANOMALY_SCORE NUMBER(6,4),
            LLM_CONFIDENCE NUMBER(6,4),
            SEASONALITY_ASSESSMENT VARCHAR2(2000),
            TREND_ASSESSMENT VARCHAR2(2000),
            PEER_ASSESSMENT VARCHAR2(2000),
            CAVEAT VARCHAR2(2000),
            ML_ANOMALY_SCORE NUMBER(6,2),
            ML_ENSEMBLE_SCORE NUMBER(6,2),
            ML_IS_ANOMALY NUMBER(1),
            ML_ALERT_BAND VARCHAR2(32),
            ML_IF_SCORE NUMBER(6,2),
            ML_RESIDUAL_SCORE NUMBER(6,2),
            ML_AUTOENCODER_SCORE NUMBER(6,2),
            REASON_SUMMARY VARCHAR2(4000),
            REASON_1 VARCHAR2(2000),
            REASON_1_WEIGHT NUMBER(6,4),
            REASON_2 VARCHAR2(2000),
            REASON_2_WEIGHT NUMBER(6,4),
            REASON_3 VARCHAR2(2000),
            REASON_3_WEIGHT NUMBER(6,4),
            RECOMMENDED_ACTION VARCHAR2(128),
            LLM_MODEL VARCHAR2(128),
            EVIDENCE_SOURCE VARCHAR2(64),
            RAW_RESPONSE CLOB,
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY (RUN_ID, {TIME_COLUMN.upper()}, {ID_COLUMN.upper()})
        )
    """


def llm_reasons_table_ddl(ora: OracleConnector, table_key: str) -> str:
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            RUN_ID VARCHAR2(64) NOT NULL,
            {TIME_COLUMN.upper()} DATE NOT NULL,
            {ID_COLUMN.upper()} VARCHAR2(128) NOT NULL,
            REASON_RANK NUMBER(4) NOT NULL,
            FEATURE_NAME VARCHAR2(128),
            EVIDENCE_TEXT VARCHAR2(2000),
            INTERPRETATION VARCHAR2(2000),
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY (RUN_ID, {TIME_COLUMN.upper()}, {ID_COLUMN.upper()}, REASON_RANK)
        )
    """


def llm_features_table_ddl(ora: OracleConnector, table_key: str) -> str:
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            RUN_ID VARCHAR2(64) NOT NULL,
            {TIME_COLUMN.upper()} DATE NOT NULL,
            {ID_COLUMN.upper()} VARCHAR2(128) NOT NULL,
            FEATURE_RANK NUMBER(4) NOT NULL,
            FEATURE_NAME VARCHAR2(128),
            FEATURE_LABEL VARCHAR2(256),
            FEATURE_CATEGORY VARCHAR2(64),
            FORMULA VARCHAR2(1000),
            SOURCE_COLUMNS VARCHAR2(1000),
            RISK_DIRECTION VARCHAR2(32),
            CURRENT_VALUE NUMBER(24,8),
            PREVIOUS_VALUE NUMBER(24,8),
            CHANGE_PCT NUMBER(18,6),
            HISTORY_PERIOD_COUNT NUMBER(10),
            HISTORY_MEDIAN NUMBER(24,8),
            HISTORY_P25 NUMBER(24,8),
            HISTORY_P75 NUMBER(24,8),
            HISTORY_ROBUST_SCALE NUMBER(24,8),
            HISTORY_Z NUMBER(18,6),
            ROLLING_3M_MEDIAN NUMBER(24,8),
            ROLLING_6M_MEDIAN NUMBER(24,8),
            ROLLING_12M_MEDIAN NUMBER(24,8),
            TREND_SLOPE_6M NUMBER(24,8),
            TREND_SLOPE_12M NUMBER(24,8),
            TREND_BREAK_FLAG NUMBER(1),
            TREND_NOTE VARCHAR2(1000),
            MONTH_OF_YEAR NUMBER(2),
            SAME_MONTH_LAST_YEAR_VALUE NUMBER(24,8),
            YOY_CHANGE_PCT NUMBER(18,6),
            SAME_MONTH_CUSTOMER_MEDIAN NUMBER(24,8),
            SAME_MONTH_CUSTOMER_Z NUMBER(18,6),
            SEASONAL_PEER_MEDIAN NUMBER(24,8),
            SEASONAL_PEER_Z NUMBER(18,6),
            PEER_MEDIAN NUMBER(24,8),
            PEER_Z NUMBER(18,6),
            PEER_SUPPORT NUMBER(10),
            PEER_DEFINITION_LEVEL VARCHAR2(64),
            PEER_QUALITY VARCHAR2(32),
            DATA_MISSING_FLAG NUMBER(1),
            SNAPSHOT_SERIES_JSON CLOB,
            FEATURE_JSON CLOB,
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY (RUN_ID, {TIME_COLUMN.upper()}, {ID_COLUMN.upper()}, FEATURE_RANK)
        )
    """
