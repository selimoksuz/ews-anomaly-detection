"""Build evidence packages for LLM-based anomaly decisions.

This module does not create a model score. It prepares the transformed inputs
that make an LLM decision auditable: variable dictionary, customer history,
seasonality, peer comparison, trend, and data-quality signals.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.config_loader import load_config, load_secrets
from engine.multivar_anomaly import (
    CONTEXT_COLUMNS,
    FEATURE_LABELS,
    FORBIDDEN_DERIVED_FEATURES,
    ID_COLUMN,
    INCREASE_IS_RISK_TOKENS,
    PEER_MIN_SUPPORT,
    RAW_MODEL_EXCLUDE_COLUMNS,
    TIME_COLUMN,
    build_feature_frame,
    build_peer_artifacts,
    build_peer_context,
    load_windows_oracle,
    normalize_columns,
    parse_dates,
    peer_hierarchy_for_feature,
    profile_months_oracle,
    sample_oracle_frame,
    select_model_features,
)
from engine.oracle_io import OracleConnector


logger = logging.getLogger(__name__)

RAW_COLUMN_LABELS = {
    "cohort_dt": "Kaydin ait oldugu ay sonu / skorlanan donem",
    "mono_id": "Musteri tekil anonim numarasi",
    "musteri_segment": "Musteri segment kodu",
    "bilanco_flg": "Bilanco veri tipi bayragi",
    "cst_sector": "Musteri faaliyet sektoru metni",
    "cst_nace_code": "Musteri faaliyet NACE kodu",
    "cst_nace_code_id": "Musteri faaliyet NACE kod id",
    "bank_total_risk": "Bankadaki toplam kredi riski",
    "financial_term_l1y": "Son 12 ay/yillik finansal donem tarihi",
    "fs_net_sales_cumulative_l1y": "Son 12 ay net satis",
    "fs_trade_receivables_l1y": "Son 12 ay ticari alacak",
    "fs_notes_receivable_l1y": "Son 12 ay senetli alacak",
    "supheli_ticari_alacaklar_l1y": "Son 12 ay supheli ticari alacak",
    "equity_l1y": "Son 12 ay ozkaynak",
    "fs_net_profit_cumulative_l1y": "Son 12 ay net kar",
    "financial_term_q": "Ara donem/ceyrek finansal donem tarihi",
    "annualization_q": "Ara donem finansali yilliklandirma katsayisi",
    "fs_net_sales_cumulative_q": "Ara donem net satis",
    "fs_ebitda_cumulative_q": "Ara donem FAVOK/EBITDA",
    "fs_net_profit_cumulative_q": "Ara donem net kar",
    "fs_trade_receivables_q": "Ara donem ticari alacak",
    "fs_notes_receivable_q": "Ara donem senetli alacak",
    "supheli_alacaklar_q": "Ara donem supheli alacak",
    "fs_equity_q": "Ara donem ozkaynak",
    "memzuc_total_risk": "Tum bankacilik sistemi toplam riski",
    "memzuc_total_limit": "Tum bankacilik sistemi toplam limiti",
    "memzuc_st_mt_cash_risk": "Kisa/orta vadeli nakdi risk",
    "irb_rating_pd": "Rating notuna karsilik gelen PD",
    "irb_model_pd": "Istatistiksel model bazli PD",
    "rating_group": "Derecelendirme grubu",
    "toplam_varlik_ttr": "Musterinin toplam varlik degeri",
    "ref_donem_id": "Varlik verisinin referans donemi",
    "gunceltkn_dgr": "Guncel bireysel KKB borcu",
    "gunceltbe_dgr": "Guncel ticari KKB borcu",
    "kkbguncelsorgu_no": "En son KKB sorgu numarasi",
    "yukleme_zmn": "Kaydin yuklenme zamani",
    "data_time": "Oracle input tablosu teknik veri zamani",
    "created_at": "Oracle input tablosu teknik olusturma zamani",
}

FEATURE_FORMULAS = {
    "memzuc_limit_utilization": "memzuc_total_risk / memzuc_total_limit",
    "memzuc_st_mt_cash_share": "memzuc_st_mt_cash_risk / memzuc_total_risk",
    "bank_risk_to_assets": "bank_total_risk / toplam_varlik_ttr",
    "memzuc_risk_to_assets": "memzuc_total_risk / toplam_varlik_ttr",
    "l1y_equity_to_assets": "equity_l1y / toplam_varlik_ttr",
    "l1y_debt_to_sales": "bank_total_risk / fs_net_sales_cumulative_l1y",
    "memzuc_debt_to_l1y_sales": "memzuc_total_risk / fs_net_sales_cumulative_l1y",
    "memzuc_to_bank_risk_ratio": "memzuc_total_risk / bank_total_risk",
    "bank_to_memzuc_risk_ratio": "bank_total_risk / memzuc_total_risk",
    "l1y_trade_receivables_to_assets": "fs_trade_receivables_l1y / toplam_varlik_ttr",
    "l1y_notes_receivable_to_assets": "fs_notes_receivable_l1y / toplam_varlik_ttr",
    "pd_ratio": "irb_rating_pd / irb_model_pd",
    "internal_tkn_to_assets": "gunceltkn_dgr / toplam_varlik_ttr",
    "internal_tbe_to_assets": "gunceltbe_dgr / toplam_varlik_ttr",
    "internal_tkn_to_sales": "gunceltkn_dgr / fs_net_sales_cumulative_l1y",
    "internal_tbe_to_sales": "gunceltbe_dgr / fs_net_sales_cumulative_l1y",
    "internal_tkn_tbe_ratio": "gunceltkn_dgr / gunceltbe_dgr",
}

DECREASE_IS_RISK_HINTS = (
    "equity",
    "ozkaynak",
    "profit",
    "kar",
    "ebitda",
    "margin",
)

TECHNICAL_COLUMNS = {
    "kkbguncelsorgu_no",
    "yukleme_zmn",
    "data_time",
    "created_at",
}

FORBIDDEN_PD_RATING_COMPARISON_SIGNALS = {
    "pd_ratio",
    "pd_to_rating_group",
}


@dataclass(frozen=True)
class EvidenceConfig:
    scoring_month: str | None = None
    max_customers: int | None = None
    top_features: int = 12
    min_history_periods: int = 3


def load_input_frame(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig", decimal=",", low_memory=False)
    return normalize_columns(frame)


def build_evidence_packages(frame: pd.DataFrame, config: EvidenceConfig | None = None) -> list[dict[str, Any]]:
    config = config or EvidenceConfig()
    frame = normalize_columns(frame).copy()
    logger.info(
        "Building LLM evidence packages: rows=%s columns=%s requested_scoring_month=%s max_customers=%s top_features=%s",
        len(frame),
        len(frame.columns),
        config.scoring_month or "latest",
        config.max_customers,
        config.top_features,
    )
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"Missing required id column: {ID_COLUMN}")
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"Missing required time column: {TIME_COLUMN}")

    frame[TIME_COLUMN] = parse_dates(frame[TIME_COLUMN])
    frame = frame.dropna(subset=[TIME_COLUMN]).sort_values([ID_COLUMN, TIME_COLUMN]).reset_index(drop=True)
    scoring_month = resolve_scoring_month(frame, config.scoring_month)
    train_df = frame[frame[TIME_COLUMN] < scoring_month].copy()
    score_df = frame[frame[TIME_COLUMN] == scoring_month].copy()
    if train_df.empty:
        raise ValueError(f"No prior rows available before scoring month {scoring_month.date()}.")
    if score_df.empty:
        raise ValueError(f"No scoring rows found for {scoring_month.date()}.")
    logger.info(
        "Resolved scoring month: %s train_rows=%s score_rows=%s",
        scoring_month.date(),
        len(train_df),
        len(score_df),
    )

    numeric_source_columns = infer_numeric_source_columns(frame)
    logger.info("Inferred numeric source columns: %s", len(numeric_source_columns))
    train_features = build_feature_frame(train_df, numeric_source_columns)
    score_features = build_feature_frame(score_df, numeric_source_columns)
    selected_features = select_model_features(train_features, score_features)
    selected_features = [feature for feature in selected_features if feature not in FORBIDDEN_DERIVED_FEATURES]
    if not selected_features:
        raise ValueError("No usable transformed features could be selected.")
    logger.info("Selected transformed features: %s", len(selected_features))
    logger.debug("Selected transformed feature names: %s", ", ".join(selected_features))
    log_transformed_feature_audit(selected_features, numeric_source_columns)
    log_evidence_contract(selected_features)

    logger.info("Building peer artifacts.")
    logger.info(
        "Peer scope: peer reference is computed on full scoring cohort before max_customers filtering: score_rows=%s selected_features=%s grouping=cohort_dt+segment+sector+monthly_size",
        len(score_df),
        len(selected_features),
    )
    peer_artifacts = build_peer_artifacts(score_df, score_features, selected_features)
    train_context = build_peer_context(train_df, train_features)
    score_context = build_peer_context(score_df, score_features)
    logger.info("Peer/context artifacts are ready.")

    packages: list[dict[str, Any]] = []
    for row_position, row_index in enumerate(score_df.index):
        if config.max_customers is not None and len(packages) >= config.max_customers:
            break
        row = score_df.loc[row_index]
        customer_id = str(row[ID_COLUMN])
        customer_history = train_df[train_df[ID_COLUMN].astype(str) == customer_id].copy()
        customer_feature_history = train_features.loc[customer_history.index] if len(customer_history) else pd.DataFrame()
        feature_evidence = []
        for feature in selected_features:
            item = build_feature_evidence(
                feature=feature,
                row_position=row_position,
                row_index=row_index,
                score_features=score_features,
                train_features=train_features,
                customer_history=customer_history,
                customer_feature_history=customer_feature_history,
                peer_artifacts=peer_artifacts,
                train_context=train_context,
                score_context_row=score_context.iloc[row_position],
                scoring_month=scoring_month,
            )
            feature_evidence.append(item)

        feature_evidence = rank_feature_evidence(feature_evidence)[: config.top_features]
        coverage_ratio = coverage_for_features(score_features.loc[row_index], selected_features)
        packages.append(
            {
                "mono_id": customer_id,
                "cohort_dt": scoring_month.strftime("%Y-%m-%d"),
                "context": context_payload(row),
                "decision_contract": {
                    "target_or_label_available": False,
                    "llm_should_decide_is_anomaly": True,
                    "future_periods_included": False,
                    "scoring_month_only": True,
                },
                "peer_definition": {
                    "base": "same cohort month plus segment/sector/size fallback hierarchy",
                    "min_support": PEER_MIN_SUPPORT,
                    "note": "IRB PD, model PD and rating_group are direct PD/rating signals; only PD/rating cross-ratios such as pd_ratio are excluded.",
                },
                "data_quality": data_quality_payload(
                    row=row,
                    feature_row=score_features.loc[row_index],
                    selected_features=selected_features,
                    coverage_ratio=coverage_ratio,
                    min_history_periods=config.min_history_periods,
                    customer_history_periods=len(customer_history),
                ),
                "features": feature_evidence,
            }
        )
        if len(packages) == 1 or len(packages) % 100 == 0:
            logger.info("Built evidence packages: %s/%s", len(packages), min(len(score_df), config.max_customers or len(score_df)))
    logger.info("Completed LLM evidence package build: packages=%s", len(packages))
    log_step_done(
        "03",
        f"evidence_packages={len(packages)} transformed_features={len(selected_features)} scoring_month={scoring_month.date()} peer_variables=6 history_variables=10 trend_variables=4 seasonality_variables=8",
    )
    return packages


def build_evidence_from_result_rows(frame: pd.DataFrame, *, max_customers: int | None = None) -> list[dict[str, Any]]:
    """Create LLM evidence from existing runtime result rows.

    This path is intentionally score-blind: anomaly_score, alert_band,
    if_score, residual_score, confidence, and review_queue are not sent to the
    LLM. It is useful when only runtime outputs are available locally. For full
    history/trend/season evidence, use raw input rows with build_evidence_packages.
    """

    frame = normalize_columns(frame).copy()
    if max_customers is not None:
        frame = frame.head(max_customers).copy()
    packages: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        details = parse_reason_details(row.get("reason_details"))
        features = [
            feature_evidence_from_detail(detail)
            for detail in details
            if not is_forbidden_pd_rating_comparison(detail.get("feature") or detail.get("feature_name"))
        ]
        packages.append(
            {
                "mono_id": clean_context_value(row.get(ID_COLUMN)),
                "cohort_dt": clean_context_value(row.get(TIME_COLUMN)),
                "context": context_payload(row),
                "decision_contract": {
                    "target_or_label_available": False,
                    "llm_should_decide_is_anomaly": True,
                    "future_periods_included": False,
                    "score_fields_excluded": True,
                    "source_note": "Built from runtime reason_details because raw input rows were not available locally.",
                },
                "peer_definition": {
                    "base": "same cohort month plus segment/sector/size fallback hierarchy",
                    "min_support": PEER_MIN_SUPPORT,
                    "note": "IRB PD, model PD and rating_group are direct PD/rating signals; only PD/rating cross-ratios such as pd_ratio are excluded.",
                },
                "data_quality": {
                    "coverage_ratio": clean_number(row.get("coverage_ratio")),
                    "missing_feature_count": int(clean_number(row.get("missing_feature_count")) or 0),
                    "history_periods_available_in_payload": False,
                    "seasonality_available_in_payload": False,
                    "caveat": "Bu evidence mevcut runtime reason_details uzerinden uretildi; tam sezon/trend icin ham aylik input gerekir.",
                },
                "features": features,
            }
        )
    return packages


def build_evidence_packages_from_oracle(
    *,
    scoring_month: str | None = None,
    max_customers: int | None = None,
    max_train_rows: int | None = 300_000,
    top_features: int = 12,
    table_key: str = "multivar_input",
    chunk_size: int = 250_000,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Build full LLM evidence directly from the configured Oracle input table."""

    log_step("01", "Oracle kaynak tablo ve ay profili okunuyor")
    logger.info(
        "Profiling Oracle input months: table_key=%s requested_scoring_month=%s",
        table_key,
        scoring_month or "latest",
    )
    month_profile = profile_months_oracle(table_key=table_key)
    selected_month = resolve_scoring_month_from_profile(month_profile, scoring_month)
    prior_rows = int(sum(count for month, count in month_profile["month_counts"].items() if month < selected_month))
    if prior_rows <= 0:
        raise ValueError(f"No prior rows available before scoring month {selected_month.date()}.")
    logger.info(
        "Oracle month profile resolved: selected_month=%s total_rows=%s prior_rows=%s month_count=%s",
        selected_month.date(),
        month_profile["total_rows"],
        prior_rows,
        len(month_profile["month_counts"]),
    )
    log_step_done(
        "01",
        f"source_table={configured_oracle_table_name(table_key)} selected_month={selected_month.date()} total_rows={month_profile['total_rows']} prior_rows={prior_rows}",
    )

    log_step("02", "Ham tablo kolonlari ve veri sozlugu denetleniyor")
    logger.info("Sampling Oracle input frame for feature inference: limit=%s", 100_000)
    sample_frame = sample_oracle_frame(table_key=table_key, limit=100_000)
    numeric_source_columns = infer_numeric_source_columns(sample_frame)
    keep_columns = [
        column
        for column in dict.fromkeys([ID_COLUMN, TIME_COLUMN, *CONTEXT_COLUMNS, *numeric_source_columns])
        if column in sample_frame.columns
    ]
    logger.info(
        "Oracle sample loaded: rows=%s columns=%s numeric_source_columns=%s keep_columns=%s",
        len(sample_frame),
        len(sample_frame.columns),
        len(numeric_source_columns),
        len(keep_columns),
    )
    log_raw_table_audit(
        table_key=table_key,
        sample_frame=sample_frame,
        numeric_source_columns=numeric_source_columns,
        keep_columns=keep_columns,
    )
    log_step_done(
        "02",
        f"raw_columns={len(sample_frame.columns)} used_input_columns={len(keep_columns)} numeric_source_columns={len(numeric_source_columns)} forbidden_pd_rating_comparisons={','.join(sorted(FORBIDDEN_PD_RATING_COMPARISON_SIGNALS))}",
    )
    logger.info(
        "Loading Oracle reference/score windows: reference_rows_limit=%s chunk_size=%s note='reference rows are used for feature/trend/seasonality statistics; they are not sent to the LLM'",
        max_train_rows,
        chunk_size,
    )
    train_df, score_df, prior_df = load_windows_oracle(
        table_key=table_key,
        selected_month=selected_month,
        prior_rows=prior_rows,
        max_train_rows=max_train_rows,
        max_score_rows=None,
        chunk_size=chunk_size,
        random_state=random_state,
        keep_columns=keep_columns,
    )
    logger.info(
        "Oracle windows loaded: train_rows=%s prior_rows=%s score_rows=%s",
        len(train_df),
        len(prior_df),
        len(score_df),
    )
    selected_customer_ids = selected_scoring_customer_ids(score_df, max_customers=max_customers)
    selected_history_df = load_selected_customer_history_oracle(
        table_key=table_key,
        customer_ids=selected_customer_ids,
        selected_month=selected_month,
        keep_columns=keep_columns,
    )
    logger.info(
        "Selected customer full history loaded: selected_customer_count=%s selected_history_rows=%s note='this full history is used for the actual LLM payload customers before max_customers filtering'",
        len(selected_customer_ids),
        len(selected_history_df),
    )
    combined = pd.concat([train_df, prior_df, selected_history_df, score_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=[ID_COLUMN, TIME_COLUMN], keep="last").reset_index(drop=True)
    logger.info("Combined Oracle evidence frame: rows=%s columns=%s", len(combined), len(combined.columns))
    logger.info(
        "LLM payload scope: scoring_rows_available=%s llm_customer_payloads_requested=%s llm_customer_payloads_to_build=%s reference_rows_not_sent_to_llm=%s",
        len(score_df),
        max_customers or "ALL",
        len(selected_customer_ids) if max_customers is not None else len(score_df),
        len(train_df),
    )
    log_step("03", "Musteri bazli history ve aylik peer gruplariyla LLM evidence uretiliyor")
    return build_evidence_packages(
        combined,
        EvidenceConfig(
            scoring_month=selected_month.strftime("%Y-%m-%d"),
            max_customers=max_customers,
            top_features=top_features,
        ),
    )


def feature_evidence_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    feature = str(detail.get("feature") or detail.get("feature_name") or "")
    current = clean_number(detail.get("actual"))
    previous = clean_number(detail.get("customer_previous_reference"))
    return {
        "name": feature,
        "dictionary": feature_dictionary(feature),
        "current_value": current,
        "previous_value": previous,
        "change_pct": pct_change(current, previous),
        "history": {
            "period_count": None,
            "median": clean_number(detail.get("train_reference")),
            "note": "Runtime output only has previous/train reference; full rolling history is not available in this file.",
        },
        "trend": {
            "previous_comment": detail.get("previous_comment"),
            "financial_term_detail": detail.get("financial_term_detail"),
            "trend_break_flag": None,
            "trend_note": "Full trend metrics require raw monthly input rows.",
        },
        "seasonality": {
            "available": False,
            "seasonality_note": "Full seasonality metrics require raw monthly input rows.",
        },
        "peer": {
            "peer_definition_level": detail.get("peer_level"),
            "peer_hierarchy": [level for level, _ in peer_hierarchy_for_feature(feature)],
            "peer_median": clean_number(detail.get("peer_reference")),
            "peer_z": clean_number(detail.get("peer_z")),
            "peer_support": int(clean_number(detail.get("peer_support")) or 0),
            "peer_quality": detail.get("peer_quality"),
            "peer_representativeness_score": clean_number(detail.get("peer_representativeness_score")),
        },
        "data_quality": {
            "missing_flag": bool(detail.get("is_missing_reason")),
            "reference_used": detail.get("reference_used"),
        },
        "direction_comment": detail.get("direction_comment"),
    }


def parse_reason_details(value: Any) -> list[dict[str, Any]]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def build_feature_evidence(
    *,
    feature: str,
    row_position: int,
    row_index: int,
    score_features: pd.DataFrame,
    train_features: pd.DataFrame,
    customer_history: pd.DataFrame,
    customer_feature_history: pd.DataFrame,
    peer_artifacts,
    train_context: pd.DataFrame,
    score_context_row: pd.Series,
    scoring_month: pd.Timestamp,
) -> dict[str, Any]:
    current = clean_number(score_features.loc[row_index, feature])
    history_series = (
        pd.to_numeric(customer_feature_history.get(feature), errors="coerce")
        if not customer_feature_history.empty and feature in customer_feature_history
        else pd.Series(dtype=float)
    )
    history_dates = (
        parse_dates(customer_history[TIME_COLUMN])
        if not customer_history.empty and TIME_COLUMN in customer_history
        else pd.Series(dtype="datetime64[ns]")
    )
    previous = clean_number(history_series.dropna().iloc[-1]) if history_series.dropna().size else None
    peer_reference = clean_number(peer_artifacts.median.loc[row_index, feature])
    peer_z = clean_number(peer_artifacts.zscore.loc[row_index, feature])
    peer_support = clean_number(peer_artifacts.support.loc[row_index, feature])
    peer_level = str(peer_artifacts.level.loc[row_index, feature])
    peer_quality = str(peer_artifacts.quality.loc[row_index, feature])

    history_payload = history_metrics(history_series)
    trend_payload = trend_metrics(history_dates, history_series, current)
    seasonality_payload = seasonality_metrics(
        feature=feature,
        current=current,
        history_dates=history_dates,
        history_series=history_series,
        train_features=train_features,
        train_context=train_context,
        score_context_row=score_context_row,
        scoring_month=scoring_month,
    )

    return {
        "name": feature,
        "dictionary": feature_dictionary(feature),
        "current_value": current,
        "previous_value": previous,
        "change_pct": pct_change(current, previous),
        "history": history_payload,
        "trend": trend_payload,
        "seasonality": seasonality_payload,
        "peer": {
            "peer_definition_level": peer_level,
            "peer_hierarchy": [level for level, _ in peer_hierarchy_for_feature(feature)],
            "peer_median": peer_reference,
            "peer_z": peer_z,
            "peer_support": int(peer_support or 0),
            "peer_quality": peer_quality,
        },
        "data_quality": {
            "missing_flag": current is None,
            "history_periods": int(history_series.notna().sum()),
        },
    }


def selected_scoring_customer_ids(score_df: pd.DataFrame, *, max_customers: int | None) -> list[str]:
    ids = score_df[ID_COLUMN].astype(str).drop_duplicates()
    if max_customers is not None:
        ids = ids.head(max_customers)
    return ids.tolist()


def load_selected_customer_history_oracle(
    *,
    table_key: str,
    customer_ids: list[str],
    selected_month: pd.Timestamp,
    keep_columns: list[str],
    batch_size: int = 900,
) -> pd.DataFrame:
    if not customer_ids:
        return pd.DataFrame(columns=keep_columns)
    parts = []
    config = load_config()
    secrets = load_secrets()
    selected_columns = ", ".join(column.upper() for column in keep_columns)
    with OracleConnector(config, secrets) as ora:
        table_name = ora._qualified_table_name(table_key)
        for offset in range(0, len(customer_ids), batch_size):
            batch = customer_ids[offset : offset + batch_size]
            binds = {f"id_{index}": value for index, value in enumerate(batch)}
            binds["selected_month"] = pd.Timestamp(selected_month).to_pydatetime()
            placeholders = ", ".join(f":id_{index}" for index in range(len(batch)))
            sql = f"""
                SELECT {selected_columns}
                FROM {table_name}
                WHERE {ID_COLUMN.upper()} IN ({placeholders})
                  AND TRUNC({TIME_COLUMN.upper()}) < TRUNC(:selected_month)
            """
            parts.append(normalize_columns(ora._read_query(sql, binds)))
    if not parts:
        return pd.DataFrame(columns=keep_columns)
    history = pd.concat(parts, ignore_index=True)
    if TIME_COLUMN in history.columns:
        history[TIME_COLUMN] = parse_dates(history[TIME_COLUMN])
    return history


def resolve_scoring_month(frame: pd.DataFrame, scoring_month: str | None) -> pd.Timestamp:
    if scoring_month:
        return pd.Timestamp(scoring_month).normalize()
    return pd.Timestamp(frame[TIME_COLUMN].max()).normalize()


def resolve_scoring_month_from_profile(month_profile: dict[str, Any], scoring_month: str | None) -> pd.Timestamp:
    if scoring_month:
        return pd.Timestamp(scoring_month).normalize()
    months = list(month_profile.get("month_counts", {}).keys())
    if not months:
        raise ValueError("No months found in Oracle source profile.")
    return pd.Timestamp(max(months)).normalize()


def infer_numeric_source_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if (
            column in {ID_COLUMN, TIME_COLUMN}
            or (column in CONTEXT_COLUMNS and column != "rating_group")
            or column in TECHNICAL_COLUMNS
        ):
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if float(series.notna().mean()) >= 0.05:
            columns.append(column)
    return columns


def feature_dictionary(feature: str) -> dict[str, Any]:
    return {
        "label": FEATURE_LABELS.get(feature, feature),
        "category": variable_category(feature),
        "formula": FEATURE_FORMULAS.get(feature),
        "risk_direction": risk_direction(feature),
        "interpretation_note": interpretation_note(feature),
    }


def is_forbidden_pd_rating_comparison(feature: Any) -> bool:
    if feature is None:
        return False
    return str(feature).strip().lower() in FORBIDDEN_PD_RATING_COMPARISON_SIGNALS


def log_step(step_no: str, title: str) -> None:
    logger.info("========== STEP %s START | %s ==========", step_no, title)


def log_step_done(step_no: str, detail: str) -> None:
    logger.info("========== STEP %s DONE | %s ==========", step_no, detail)


def raw_column_dictionary(column: str) -> dict[str, Any]:
    return {
        "label": RAW_COLUMN_LABELS.get(column) or FEATURE_LABELS.get(column),
        "category": variable_category(column),
        "role": raw_column_role(column),
    }


def raw_column_role(column: str) -> str:
    if column == ID_COLUMN:
        return "id"
    if column == TIME_COLUMN:
        return "time"
    if variable_category(column) == "pd_rating":
        return "direct_pd_rating_signal"
    if column in CONTEXT_COLUMNS:
        return "context_or_peer"
    if column in TECHNICAL_COLUMNS:
        return "technical_excluded"
    return "numeric_source_candidate"


def log_raw_table_audit(
    *,
    table_key: str,
    sample_frame: pd.DataFrame,
    numeric_source_columns: list[str],
    keep_columns: list[str],
) -> None:
    raw_columns = list(sample_frame.columns)
    used_columns = set(keep_columns)
    numeric_columns = set(numeric_source_columns)
    missing_all = [column for column in raw_columns if not raw_column_dictionary(column)["label"]]
    missing_used = [column for column in raw_columns if column in used_columns and not raw_column_dictionary(column)["label"]]
    logger.info(
        "AUDIT RAW SOURCE | table_key=%s table=%s raw_table_columns=%s used_input_columns=%s numeric_source_columns=%s dictionary_coverage_all=%s/%s dictionary_coverage_used=%s/%s",
        table_key,
        configured_oracle_table_name(table_key),
        len(raw_columns),
        len(used_columns),
        len(numeric_columns),
        len(raw_columns) - len(missing_all),
        len(raw_columns),
        len(used_columns) - len(missing_used),
        len(used_columns),
    )
    for column in raw_columns:
        dictionary = raw_column_dictionary(column)
        logger.info(
            "AUDIT RAW VARIABLE | name=%s category=%s used=%s numeric_source=%s role=%s description=%s",
            column,
            dictionary["category"],
            column in used_columns,
            column in numeric_columns,
            dictionary["role"],
            dictionary["label"] or "MISSING_DICTIONARY",
        )
    if missing_used:
        logger.warning("AUDIT RAW DICTIONARY MISSING FOR USED VARIABLES | columns=%s", ", ".join(missing_used))
    if missing_all:
        logger.warning("AUDIT RAW DICTIONARY MISSING FOR TABLE VARIABLES | columns=%s", ", ".join(missing_all))


def log_transformed_feature_audit(selected_features: list[str], numeric_source_columns: list[str]) -> None:
    missing = [feature for feature in selected_features if not feature_dictionary(feature).get("label")]
    logger.info(
        "AUDIT TRANSFORMED FEATURES | selected_count=%s dictionary_coverage=%s/%s",
        len(selected_features),
        len(selected_features) - len(missing),
        len(selected_features),
    )
    for feature in selected_features:
        dictionary = feature_dictionary(feature)
        logger.info(
            "AUDIT TRANSFORMED FEATURE | name=%s category=%s description=%s formula=%s source_raw_columns=%s risk_direction=%s note=%s",
            feature,
            dictionary["category"],
            dictionary["label"] or "MISSING_DICTIONARY",
            dictionary["formula"] or "raw_or_direct_transformed_feature",
            ",".join(feature_source_columns(feature, numeric_source_columns)) or "UNKNOWN",
            dictionary["risk_direction"],
            dictionary["interpretation_note"],
        )
    if missing:
        logger.warning("AUDIT TRANSFORMED DICTIONARY MISSING | features=%s", ", ".join(missing))


def log_evidence_contract(selected_features: list[str]) -> None:
    logger.info(
        "AUDIT PIPELINE CONTRACT | raw_input=Oracle raw monthly rows generated_features=%s peer_grouping=%s excluded_signals=%s",
        len(selected_features),
        "cohort_dt + musteri_segment + sector + monthly_size fallback; direct PD/rating signals allowed; PD/rating cross-ratios forbidden",
        ",".join(sorted(FORBIDDEN_PD_RATING_COMPARISON_SIGNALS)),
    )
    logger.info(
        "AUDIT PEER VARIABLES | variables=peer_definition_level,peer_hierarchy,peer_median,peer_z,peer_support,peer_quality"
    )
    logger.info(
        "AUDIT HISTORY VARIABLES | variables=period_count,median,p25,p75,robust_scale,rolling_3m_median,rolling_6m_median,rolling_12m_median,previous_value,change_pct"
    )
    logger.info(
        "AUDIT TREND VARIABLES | variables=slope_6m,slope_12m,trend_break_flag,trend_note"
    )
    logger.info(
        "AUDIT SEASONALITY VARIABLES | variables=month_of_year,same_month_last_year_value,yoy_change_pct,same_month_customer_median,same_month_customer_z,seasonal_peer_median,seasonal_peer_z,seasonality_note"
    )
    logger.info(
        "AUDIT LLM INPUT CONTRACT | one JSON evidence package per customer-period; includes context,data_quality,feature dictionary,current/history/peer/trend/seasonality; excludes model score,target,PD/rating cross-ratios"
    )


def feature_source_columns(feature: str, numeric_source_columns: list[str]) -> list[str]:
    formula = FEATURE_FORMULAS.get(feature)
    if not formula:
        return [feature] if feature in numeric_source_columns else []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    known_columns = set(numeric_source_columns) | set(RAW_COLUMN_LABELS) | set(FEATURE_LABELS)
    return [token for token in tokens if token in known_columns]


def variable_category(name: str) -> str:
    lower = str(name).lower()
    if lower in {ID_COLUMN, TIME_COLUMN}:
        return "identity_time"
    if lower in TECHNICAL_COLUMNS:
        return "technical"
    if lower in {"musteri_segment", "cst_sector", "cst_nace_code", "cst_nace_code_id", "bilanco_flg", "ref_donem_id"}:
        return "context"
    if "memzuc" in lower:
        return "memzuc"
    if (
        lower.startswith("fs_")
        or lower.startswith("l1y_")
        or lower.startswith("q_")
        or "supheli" in lower
        or "alacak" in lower
        or "receivable" in lower
        or lower
        in {
            "equity_l1y",
            "toplam_varlik_ttr",
            "financial_term_l1y",
            "financial_term_q",
            "annualization_q",
        }
    ):
        return "financial"
    if lower.startswith("irb_") or "pd" in lower or "rating" in lower:
        return "pd_rating"
    if "gunceltkn" in lower or "gunceltbe" in lower or "tkn" in lower or "tbe" in lower:
        return "internal_kkb"
    if "kkb" in lower:
        return "kkb"
    if lower.startswith("bank_"):
        return "bank_risk"
    if "external" in lower:
        return "external"
    return "other"


def configured_oracle_table_name(table_key: str) -> str:
    try:
        config = load_config()
        secrets = load_secrets()
        ora = OracleConnector(config, secrets)
        return ora._qualified_table_name(table_key)
    except Exception as exc:  # pragma: no cover - only affects diagnostic log text
        logger.debug("Could not resolve configured Oracle table name for %s: %s", table_key, exc)
        return table_key


def risk_direction(feature: str) -> str:
    lower = feature.lower()
    if lower == "rating_group":
        return "HIGHER_IS_RISKY"
    if any(token in lower for token in DECREASE_IS_RISK_HINTS):
        return "LOWER_IS_RISKY"
    if any(token in lower for token in INCREASE_IS_RISK_TOKENS):
        return "HIGHER_IS_RISKY"
    return "UNKNOWN"


def interpretation_note(feature: str) -> str:
    lower = feature.lower()
    if lower.startswith("pd_") or "pd" in lower:
        return "PD sinyali rating_group ile cift kanit gibi sayilmamali."
    if lower.startswith("l1y_") or lower.endswith("_l1y"):
        return "L1Y finansal term kontrolu ile okunmali."
    if "memzuc" in lower:
        return "Bankamiz disi toplam sistem riskiyle birlikte yorumlanmali."
    if "internal" in lower or "tkn" in lower or "tbe" in lower:
        return "KKB/internal sorgu degeri; stale veya missing olmasi ayrica veri kalitesi sinyalidir."
    return "Musteri gecmisi, peer ve sezon etkisiyle birlikte yorumlanmali."


def history_metrics(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if values.empty:
        return empty_history_payload()
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = mad * 1.4826 if mad > 1e-9 else float(values.std() or 0.0)
    scale = max(scale, 1e-6)
    return {
        "period_count": int(values.size),
        "median": clean_number(median),
        "p25": clean_number(values.quantile(0.25)),
        "p75": clean_number(values.quantile(0.75)),
        "robust_scale": clean_number(scale),
        "rolling_3m_median": rolling_median(values, 3),
        "rolling_6m_median": rolling_median(values, 6),
        "rolling_12m_median": rolling_median(values, 12),
    }


def empty_history_payload() -> dict[str, Any]:
    return {
        "period_count": 0,
        "median": None,
        "p25": None,
        "p75": None,
        "robust_scale": None,
        "rolling_3m_median": None,
        "rolling_6m_median": None,
        "rolling_12m_median": None,
    }


def trend_metrics(dates: pd.Series, values: pd.Series, current: float | None) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce")
    history = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "value": values}).dropna()
    result = {
        "slope_6m": None,
        "slope_12m": None,
        "trend_break_flag": False,
        "trend_note": "Yeterli tarihsel veri yok.",
    }
    if current is None or len(history) < 3:
        return result
    current_row = pd.DataFrame({"date": [pd.Timestamp.max.normalize()], "value": [current]})
    combined = pd.concat([history.tail(12), current_row], ignore_index=True)
    result["slope_6m"] = slope_for_tail(combined, 6)
    result["slope_12m"] = slope_for_tail(combined, 12)
    hist_values = history["value"].astype(float)
    median = hist_values.median()
    mad = (hist_values - median).abs().median()
    scale = max(float(mad * 1.4826 if mad > 1e-9 else hist_values.std() or 1.0), 1e-6)
    z_value = abs(float(current) - float(median)) / scale
    result["trend_break_flag"] = bool(z_value >= 3.0)
    result["trend_note"] = "Cari deger tarihsel robust banda gore kirilim gosteriyor." if z_value >= 3.0 else "Belirgin trend kirilimi yok."
    return result


def seasonality_metrics(
    *,
    feature: str,
    current: float | None,
    history_dates: pd.Series,
    history_series: pd.Series,
    train_features: pd.DataFrame,
    train_context: pd.DataFrame,
    score_context_row: pd.Series,
    scoring_month: pd.Timestamp,
) -> dict[str, Any]:
    month_of_year = int(scoring_month.month)
    same_month_mask = pd.to_datetime(history_dates, errors="coerce").dt.month == month_of_year
    same_month_values = pd.to_numeric(history_series[same_month_mask], errors="coerce").dropna()
    same_month_last_year_value = previous_same_month_value(history_dates, history_series, scoring_month)
    same_month_median = clean_number(same_month_values.median()) if len(same_month_values) else None
    same_month_z = robust_z(current, same_month_values) if current is not None and len(same_month_values) >= 2 else None
    seasonal_peer_median = historical_seasonal_peer_median(
        feature=feature,
        train_features=train_features,
        train_context=train_context,
        score_context_row=score_context_row,
        month_of_year=month_of_year,
    )
    return {
        "month_of_year": month_of_year,
        "same_month_last_year_value": same_month_last_year_value,
        "yoy_change_pct": pct_change(current, same_month_last_year_value),
        "same_month_customer_median": same_month_median,
        "same_month_customer_z": same_month_z,
        "seasonal_peer_median": seasonal_peer_median,
        "seasonal_peer_z": robust_z_against_reference(current, seasonal_peer_median, train_features.get(feature)),
        "seasonality_note": "Sezon etkisi ayni ay gecmisi ve ayni ay peer referansiyla okunmali.",
    }


def previous_same_month_value(dates: pd.Series, values: pd.Series, scoring_month: pd.Timestamp) -> float | None:
    frame = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty:
        return None
    same_month = frame[frame["date"].dt.month == scoring_month.month].sort_values("date")
    if same_month.empty:
        return None
    return clean_number(same_month.iloc[-1]["value"])


def historical_seasonal_peer_median(
    *,
    feature: str,
    train_features: pd.DataFrame,
    train_context: pd.DataFrame,
    score_context_row: pd.Series,
    month_of_year: int,
) -> float | None:
    if feature not in train_features.columns or train_context.empty:
        return None
    months = pd.to_datetime(train_context[TIME_COLUMN], errors="coerce").dt.month
    mask = months.eq(month_of_year)
    for _, keys in peer_hierarchy_for_feature(feature):
        usable_keys = [key for key in keys if key != TIME_COLUMN and key in train_context.columns and key in score_context_row.index]
        local_mask = mask.copy()
        for key in usable_keys:
            local_mask &= train_context[key].astype(str).eq(str(score_context_row[key]))
        values = pd.to_numeric(train_features.loc[local_mask, feature], errors="coerce").dropna()
        if len(values) >= PEER_MIN_SUPPORT:
            return clean_number(values.median())
    values = pd.to_numeric(train_features.loc[mask, feature], errors="coerce").dropna()
    return clean_number(values.median()) if len(values) else None


def rank_feature_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=evidence_strength, reverse=True)


def evidence_strength(item: dict[str, Any]) -> float:
    current_missing = 1.0 if item["data_quality"]["missing_flag"] else 0.0
    peer_z = abs(float(item["peer"].get("peer_z") or 0.0))
    hist_z = abs(float(history_z_from_item(item) or 0.0))
    season_z = abs(float(item["seasonality"].get("same_month_customer_z") or 0.0))
    trend_flag = 1.0 if item["trend"].get("trend_break_flag") else 0.0
    return max(peer_z, hist_z, season_z) + 1.5 * trend_flag + 0.5 * current_missing


def history_z_from_item(item: dict[str, Any]) -> float | None:
    current = item.get("current_value")
    median = item.get("history", {}).get("median")
    scale = item.get("history", {}).get("robust_scale")
    if current is None or median is None or not scale:
        return None
    return clean_number((float(current) - float(median)) / float(scale))


def data_quality_payload(
    *,
    row: pd.Series,
    feature_row: pd.Series,
    selected_features: list[str],
    coverage_ratio: float,
    min_history_periods: int,
    customer_history_periods: int,
) -> dict[str, Any]:
    l1y_term = clean_context_value(row.get("financial_term_l1y"))
    q_term = clean_context_value(row.get("financial_term_q"))
    return {
        "coverage_ratio": clean_number(coverage_ratio),
        "missing_feature_count": int(feature_row[selected_features].isna().sum()),
        "customer_history_periods": int(customer_history_periods),
        "insufficient_history_flag": bool(customer_history_periods < min_history_periods),
        "financial_term_l1y": l1y_term,
        "financial_term_q": q_term,
    }


def context_payload(row: pd.Series) -> dict[str, Any]:
    payload = {}
    for column in CONTEXT_COLUMNS:
        if column in row.index:
            payload[column] = clean_context_value(row.get(column))
    return payload


def coverage_for_features(row: pd.Series, features: list[str]) -> float:
    if not features:
        return 0.0
    return float(pd.to_numeric(row[features], errors="coerce").notna().mean())


def rolling_median(values: pd.Series, window: int) -> float | None:
    if values.empty:
        return None
    return clean_number(values.tail(window).median())


def slope_for_tail(frame: pd.DataFrame, window: int) -> float | None:
    tail = frame.tail(window).dropna(subset=["value"])
    if len(tail) < 3:
        return None
    x = np.arange(len(tail), dtype=float)
    y = tail["value"].astype(float).to_numpy()
    return clean_number(np.polyfit(x, y, 1)[0])


def robust_z(current: float | None, values: pd.Series) -> float | None:
    if current is None:
        return None
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(values) < 2:
        return None
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = max(mad * 1.4826 if mad > 1e-9 else float(values.std() or 1.0), 1e-6)
    return clean_number((float(current) - median) / scale)


def robust_z_against_reference(current: float | None, reference: float | None, population: pd.Series | None) -> float | None:
    if current is None or reference is None or population is None:
        return None
    values = pd.to_numeric(population, errors="coerce").dropna().astype(float)
    if len(values) < 2:
        return None
    mad = float((values - values.median()).abs().median())
    scale = max(mad * 1.4826 if mad > 1e-9 else float(values.std() or 1.0), 1e-6)
    return clean_number((float(current) - float(reference)) / scale)


def pct_change(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None or abs(float(reference)) < 1e-9:
        return None
    return clean_number((float(current) - float(reference)) / abs(float(reference)) * 100.0)


def clean_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def clean_context_value(value: Any) -> str | int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float, np.number)):
        return clean_number(value)
    text = str(value).strip()
    return text or None


def write_jsonl(items: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LLM anomaly evidence JSONL.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--from-results", action="store_true")
    parser.add_argument("--scoring-month")
    parser.add_argument("--max-customers", type=int)
    parser.add_argument("--top-features", type=int, default=12)
    args = parser.parse_args(argv)
    frame = load_input_frame(args.input_path)
    if args.from_results:
        packages = build_evidence_from_result_rows(frame, max_customers=args.max_customers)
    else:
        packages = build_evidence_packages(
            frame,
            EvidenceConfig(
                scoring_month=args.scoring_month,
                max_customers=args.max_customers,
                top_features=args.top_features,
            ),
        )
    output_path = write_jsonl(packages, args.output_path)
    print(f"wrote {len(packages)} evidence packages to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
