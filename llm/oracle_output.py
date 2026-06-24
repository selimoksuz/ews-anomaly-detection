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

logger = logging.getLogger(__name__)

LLM_RESULT_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "is_anomaly",
    "anomaly_type",
    "risk_level",
    "llm_confidence",
    "seasonality_assessment",
    "trend_assessment",
    "peer_assessment",
    "caveat",
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


def write_llm_outputs_to_oracle(
    decisions: list[dict[str, Any]],
    *,
    llm_model: str,
    evidence_source: str = "oracle_input",
    results_table_key: str = DEFAULT_LLM_RESULTS_TABLE_KEY,
    reasons_table_key: str = DEFAULT_LLM_REASONS_TABLE_KEY,
    batch_size: int = 1000,
) -> dict[str, Any]:
    if not decisions:
        logger.info("No LLM decisions to persist to Oracle.")
        return {
            "backend": "oracle",
            "run_id": None,
            "inserted_results": 0,
            "inserted_reasons": 0,
            "deleted_results": 0,
            "deleted_reasons": 0,
        }

    scoring_month = first_scoring_month(decisions)
    run_id = f"llm_{scoring_month.strftime('%Y%m%d')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result_frame = prepare_llm_result_frame(
        decisions,
        run_id=run_id,
        llm_model=llm_model,
        evidence_source=evidence_source,
    )
    reason_frame = prepare_llm_reason_frame(decisions, run_id=run_id)
    logger.info(
        "Prepared LLM Oracle output frames: run_id=%s result_rows=%s reason_rows=%s",
        run_id,
        len(result_frame),
        len(reason_frame),
    )

    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        logger.info(
            "Ensuring LLM Oracle output tables: results=%s reasons=%s",
            ora._qualified_table_name(results_table_key),
            ora._qualified_table_name(reasons_table_key),
        )
        ensure_llm_output_tables(
            ora,
            results_table_key=results_table_key,
            reasons_table_key=reasons_table_key,
        )
        logger.info("Deleting previous LLM output rows for scoring_month=%s", scoring_month.date())
        deleted = delete_llm_output_month(
            ora,
            scoring_month=scoring_month,
            results_table_key=results_table_key,
            reasons_table_key=reasons_table_key,
        )
        logger.info("Deleted old LLM rows: results=%s reasons=%s", deleted["results"], deleted["reasons"])
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
        logger.info(
            "Inserted LLM Oracle rows: results=%s reasons=%s",
            inserted_results,
            inserted_reasons,
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
        return {
            "backend": "oracle",
            "run_id": run_id,
            "results_table_key": results_table_key,
            "reasons_table_key": reasons_table_key,
            "results_table": ora._qualified_table_name(results_table_key),
            "reasons_table": ora._qualified_table_name(reasons_table_key),
            "deleted_results": int(deleted["results"]),
            "deleted_reasons": int(deleted["reasons"]),
            "inserted_results": int(inserted_results),
            "inserted_reasons": int(inserted_reasons),
            "result_table_total_rows_after": int(result_counts["total_rows"]),
            "result_table_scoring_month_rows_after": int(result_counts["scoring_month_rows"]),
            "result_table_run_rows_after": int(result_counts["run_rows"]),
            "reason_table_total_rows_after": int(reason_counts["total_rows"]),
            "reason_table_scoring_month_rows_after": int(reason_counts["scoring_month_rows"]),
            "reason_table_run_rows_after": int(reason_counts["run_rows"]),
        }


def audit_llm_output_tables(
    scoring_month,
    *,
    results_table_key: str = DEFAULT_LLM_RESULTS_TABLE_KEY,
    reasons_table_key: str = DEFAULT_LLM_REASONS_TABLE_KEY,
) -> dict[str, Any]:
    scoring_month = pd.Timestamp(scoring_month).normalize()
    config = load_config()
    secrets = load_secrets()
    audit: dict[str, Any] = {}
    with OracleConnector(config, secrets) as ora:
        for label, table_key in (("results", results_table_key), ("reasons", reasons_table_key)):
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
) -> None:
    ddls = {
        results_table_key: llm_results_table_ddl(ora, results_table_key),
        reasons_table_key: llm_reasons_table_ddl(ora, reasons_table_key),
    }
    with ora.connection.cursor() as cursor:
        for table_key, ddl in ddls.items():
            if ora._table_exists(table_key):
                continue
            cursor.execute(ddl)
            ora.logger.info("Created %s", ora._qualified_table_name(table_key))
    ora.connection.commit()


def delete_llm_output_month(
    ora: OracleConnector,
    *,
    scoring_month: pd.Timestamp,
    results_table_key: str,
    reasons_table_key: str,
) -> dict[str, int]:
    deleted = {"results": 0, "reasons": 0}
    params = {"scoring_month": pd.Timestamp(scoring_month).to_pydatetime()}
    with ora.connection.cursor() as cursor:
        for label, table_key in (("reasons", reasons_table_key), ("results", results_table_key)):
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


def prepare_llm_result_frame(
    decisions: list[dict[str, Any]],
    *,
    run_id: str,
    llm_model: str,
    evidence_source: str,
) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        rows.append(
            {
                "run_id": text_or_none(run_id, 64),
                TIME_COLUMN: parse_single_date(decision.get("cohort_dt")),
                ID_COLUMN: text_or_none(decision.get("mono_id"), 128),
                "is_anomaly": 1 if bool(decision.get("is_anomaly")) else 0,
                "anomaly_type": text_or_none(decision.get("anomaly_type"), 64),
                "risk_level": text_or_none(decision.get("risk_level"), 32),
                "llm_confidence": number_or_none(decision.get("confidence")),
                "seasonality_assessment": text_or_none(decision.get("seasonality_assessment"), 2000),
                "trend_assessment": text_or_none(decision.get("trend_assessment"), 2000),
                "peer_assessment": text_or_none(decision.get("peer_assessment"), 2000),
                "caveat": text_or_none(decision.get("caveat"), 2000),
                "recommended_action": text_or_none(decision.get("recommended_action"), 128),
                "llm_model": text_or_none(llm_model, 128),
                "evidence_source": text_or_none(evidence_source, 64),
                "raw_response": json.dumps(decision, ensure_ascii=False),
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
            LLM_CONFIDENCE NUMBER(6,4),
            SEASONALITY_ASSESSMENT VARCHAR2(2000),
            TREND_ASSESSMENT VARCHAR2(2000),
            PEER_ASSESSMENT VARCHAR2(2000),
            CAVEAT VARCHAR2(2000),
            RECOMMENDED_ACTION VARCHAR2(128),
            LLM_MODEL VARCHAR2(128),
            EVIDENCE_SOURCE VARCHAR2(64),
            RAW_RESPONSE CLOB,
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY ({TIME_COLUMN.upper()}, {ID_COLUMN.upper()})
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
            CONSTRAINT {pk_name} PRIMARY KEY ({TIME_COLUMN.upper()}, {ID_COLUMN.upper()}, REASON_RANK)
        )
    """
