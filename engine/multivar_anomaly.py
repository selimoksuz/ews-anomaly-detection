"""Multivariate anomaly scoring for the anomaly_multivar dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from engine.config_loader import load_config, load_secrets, resolve_project_path
from engine.oracle_io import OracleConnector


logger = logging.getLogger(__name__)

ID_COLUMN = "mono_id"
TIME_COLUMN = "cohort_dt"
DEFAULT_MULTIVAR_TABLE_KEY = "multivar_input"
DEFAULT_MULTIVAR_RESULTS_TABLE_KEY = "multivar_results"
DEFAULT_MULTIVAR_DETAILS_TABLE_KEY = "multivar_details"

EXCLUDED_FEATURE_COLUMNS = {
    "financial_term_l1y",
    "bilanco_flg",
    "financial_term_q",
    "annualization_q",
    "ref_donem_id",
    "kkbguncelsorgu_no",
    "yukleme_zmn",
}

DESCRIPTOR_COLUMNS = {
    "musteri_segment",
    "cst_sector",
    "cst_nace_code",
    "cst_nace_code_id",
}

CONTEXT_COLUMNS = [
    "musteri_segment",
    "rating_group",
    "cst_sector",
    "cst_nace_code",
    "cst_nace_code_id",
    "financial_term_l1y",
    "financial_term_q",
    "annualization_q",
    "ref_donem_id",
    "yukleme_zmn",
]

OPERATIONAL_BAND_POLICY = {
    "yellow_quantile": 0.95,
    "orange_quantile": 0.98,
    "red_quantile": 0.99,
    "yellow_floor": 90.0,
    "orange_floor": 95.0,
    "red_floor": 97.5,
}

DERIVED_INPUT_COLUMNS = {
    "bank_total_risk",
    "toplam_varlik_ttr",
    "memzuc_total_risk",
    "memzuc_total_limit",
    "memzuc_st_mt_cash_risk",
    "fs_net_sales_cumulative_l1y",
    "fs_trade_receivables_l1y",
    "fs_notes_receivable_l1y",
    "fs_net_profit_cumulative_l1y",
    "equity_l1y",
    "fs_net_sales_cumulative_q",
    "fs_trade_receivables_q",
    "fs_notes_receivable_q",
    "fs_net_profit_cumulative_q",
    "fs_ebitda_cumulative_q",
    "fs_equity_q",
    "irb_rating_pd",
    "irb_model_pd",
    "rating_group",
    "gunceltkn_dgr",
    "gunceltbe_dgr",
}

RAW_MODEL_EXCLUDE_COLUMNS = DERIVED_INPUT_COLUMNS | {
    "supheli_ticari_alacaklar_l1y",
    "supheli_alacaklar_q",
    "irb_rating_pd",
    "irb_model_pd",
    "rating_group",
    "gunceltkn_dgr",
    "gunceltbe_dgr",
}

DERIVED_FEATURE_PREFIXES = (
    "l1y_",
    "q_",
    "memzuc_",
    "bank_",
    "pd_",
    "rating_",
    "kkb_",
    "internal_",
)

MULTIVAR_COLUMN_TYPES = {
    TIME_COLUMN: "DATE",
    ID_COLUMN: "VARCHAR2(128)",
    "musteri_segment": "NUMBER",
    "bilanco_flg": "NUMBER",
    "cst_sector": "VARCHAR2(500)",
    "cst_nace_code": "VARCHAR2(500)",
    "cst_nace_code_id": "NUMBER",
    "financial_term_l1y": "DATE",
    "financial_term_q": "DATE",
    "rating_group": "NUMBER",
    "ref_donem_id": "NUMBER",
    "kkbguncelsorgu_no": "NUMBER",
    "yukleme_zmn": "TIMESTAMP",
}

MULTIVAR_BASE_COLUMNS = [
    "cohort_dt",
    "mono_id",
    "musteri_segment",
    "bilanco_flg",
    "cst_sector",
    "cst_nace_code",
    "cst_nace_code_id",
    "bank_total_risk",
    "financial_term_l1y",
    "fs_net_sales_cumulative_l1y",
    "fs_trade_receivables_l1y",
    "fs_notes_receivable_l1y",
    "supheli_ticari_alacaklar_l1y",
    "equity_l1y",
    "fs_net_profit_cumulative_l1y",
    "financial_term_q",
    "annualization_q",
    "fs_net_sales_cumulative_q",
    "fs_ebitda_cumulative_q",
    "fs_net_profit_cumulative_q",
    "fs_trade_receivables_q",
    "fs_notes_receivable_q",
    "supheli_alacaklar_q",
    "fs_equity_q",
    "memzuc_total_risk",
    "memzuc_total_limit",
    "memzuc_st_mt_cash_risk",
    "irb_rating_pd",
    "irb_model_pd",
    "rating_group",
    "toplam_varlik_ttr",
    "ref_donem_id",
    "gunceltkn_dgr",
    "gunceltbe_dgr",
    "kkbguncelsorgu_no",
    "yukleme_zmn",
]

MULTIVAR_RESULT_REASON_COLUMNS = ["reason_1", "reason_2", "reason_3"]

MULTIVAR_RESULT_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "musteri_segment",
    "rating_group",
    "cst_sector",
    "cst_nace_code",
    "cst_nace_code_id",
    "financial_term_l1y",
    "financial_term_q",
    "annualization_q",
    "ref_donem_id",
    "yukleme_zmn",
    "anomaly_score",
    "alert_band",
    "alert_type",
    "review_queue",
    "if_score",
    "residual_score",
    "confidence",
    "coverage_ratio",
    "data_gap_score",
    "missing_feature_count",
    "rank_in_run",
    *MULTIVAR_RESULT_REASON_COLUMNS,
    "source_table_key",
    "model_feature_count",
    "peer_feature_count",
]

MULTIVAR_DETAIL_COLUMNS = [
    "run_id",
    TIME_COLUMN,
    ID_COLUMN,
    "feature_rank",
    "feature_name",
    "feature_label",
    "is_missing_reason",
    "actual_value",
    "customer_previous_reference",
    "peer_reference",
    "peer_z",
    "peer_support",
    "train_reference",
    "reference_used",
    "contribution_pct",
    "raw_contribution_pct",
    "peer_contribution_pct",
    "missing_contribution_pct",
    "direction_comment",
    "previous_comment",
    "financial_term_detail",
    "reason_text",
]

FEATURE_LABELS = {
    "bank_total_risk": "Banka toplam risk",
    "fs_net_sales_cumulative_l1y": "Net satis L1Y",
    "fs_trade_receivables_l1y": "Ticari alacak L1Y",
    "fs_notes_receivable_l1y": "Senetli alacak L1Y",
    "supheli_ticari_alacaklar_l1y": "Supheli ticari alacak L1Y",
    "equity_l1y": "Ozkaynak L1Y",
    "fs_net_profit_cumulative_l1y": "Net kar L1Y",
    "fs_net_sales_cumulative_q": "Net satis ara donem",
    "fs_ebitda_cumulative_q": "EBITDA ara donem",
    "fs_net_profit_cumulative_q": "Net kar ara donem",
    "fs_trade_receivables_q": "Ticari alacak ara donem",
    "fs_notes_receivable_q": "Senetli alacak ara donem",
    "supheli_alacaklar_q": "Supheli alacak ara donem",
    "fs_equity_q": "Ozkaynak ara donem",
    "memzuc_total_risk": "Memzuc toplam risk",
    "memzuc_total_limit": "Memzuc toplam limit",
    "memzuc_st_mt_cash_risk": "Memzuc KV/OV nakdi risk",
    "irb_rating_pd": "IRB rating PD",
    "irb_model_pd": "IRB model PD",
    "rating_group": "Rating grup",
    "toplam_varlik_ttr": "Toplam varlik",
    "gunceltkn_dgr": "Guncel TKN degeri",
    "gunceltbe_dgr": "Guncel TBE degeri",
    "memzuc_limit_utilization": "Memzuc limit kullanim orani",
    "memzuc_st_mt_cash_share": "Memzuc KV/OV nakdi risk payi",
    "bank_risk_to_assets": "Banka risk / varlik",
    "memzuc_risk_to_assets": "Memzuc risk / varlik",
    "l1y_trade_receivables_to_sales": "L1Y ticari alacak / satis",
    "l1y_notes_receivable_to_sales": "L1Y senetli alacak / satis",
    "q_trade_receivables_to_sales": "Ara donem ticari alacak / satis",
    "q_notes_receivable_to_sales": "Ara donem senetli alacak / satis",
    "l1y_profit_margin": "L1Y kar marji",
    "q_profit_margin": "Ara donem kar marji",
    "l1y_equity_to_assets": "L1Y ozkaynak / varlik",
    "q_equity_to_assets": "Ara donem ozkaynak / varlik",
    "q_to_l1y_sales_ratio": "Ara donem satis / L1Y satis",
    "q_to_l1y_profit_ratio": "Ara donem kar / L1Y kar",
    "q_ebitda_margin": "Ara donem EBITDA marji",
    "l1y_debt_to_sales": "Banka risk / L1Y satis",
    "q_debt_to_sales": "Banka risk / ara donem satis",
    "memzuc_debt_to_l1y_sales": "Memzuc risk / L1Y satis",
    "memzuc_debt_to_q_sales": "Memzuc risk / ara donem satis",
    "memzuc_to_bank_risk_ratio": "Memzuc risk / banka risk",
    "bank_to_memzuc_risk_ratio": "Banka risk / memzuc risk",
    "l1y_trade_receivables_to_assets": "L1Y ticari alacak / varlik",
    "l1y_notes_receivable_to_assets": "L1Y senetli alacak / varlik",
    "q_trade_receivables_to_assets": "Ara donem ticari alacak / varlik",
    "q_notes_receivable_to_assets": "Ara donem senetli alacak / varlik",
    "l1y_suspicious_receivables_to_sales": "L1Y supheli alacak / satis",
    "q_suspicious_receivables_to_sales": "Ara donem supheli alacak / satis",
    "l1y_suspicious_to_trade_receivables": "L1Y supheli alacak / ticari alacak",
    "q_suspicious_to_trade_receivables": "Ara donem supheli alacak / ticari alacak",
    "q_to_l1y_equity_ratio": "Ara donem ozkaynak / L1Y ozkaynak",
    "q_to_l1y_trade_receivables_ratio": "Ara donem ticari alacak / L1Y ticari alacak",
    "q_to_l1y_notes_receivable_ratio": "Ara donem senetli alacak / L1Y senetli alacak",
    "q_to_l1y_suspicious_receivables_ratio": "Ara donem supheli alacak / L1Y supheli alacak",
    "pd_ratio": "IRB rating PD / model PD",
    "pd_to_rating_group": "PD / rating grup",
    "internal_tkn_to_assets": "TKN / varlik",
    "internal_tbe_to_assets": "TBE / varlik",
    "internal_tkn_to_sales": "TKN / L1Y satis",
    "internal_tbe_to_sales": "TBE / L1Y satis",
    "internal_tkn_tbe_ratio": "TKN / TBE",
}

INCREASE_IS_RISK_TOKENS = (
    "risk",
    "pd",
    "utilization",
    "supheli",
    "alacak",
    "receivable",
    "debt",
)
DECREASE_IS_RISK_TOKENS = (
    "profit",
    "kar",
    "equity",
    "ozkaynak",
    "ebitda",
    "margin",
)

PEER_MIN_SUPPORT = 50
PEER_Z_CLIP = 10.0
PEER_FEATURE_SUFFIX = "__peer_z"


@dataclass
class MultivarRunArtifacts:
    output_dir: Path
    scores_path: Path
    top_path: Path
    summary_path: Path
    feature_profile_path: Path


@dataclass
class RobustPreprocessor:
    features: list[str]
    continuous_features: set[str]
    fill_values: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    center: pd.Series
    scale: pd.Series

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.features].copy()
        values = values.fillna(self.fill_values)
        if self.continuous_features:
            continuous = list(self.continuous_features)
            values[continuous] = values[continuous].clip(
                lower=self.lower_bounds[continuous],
                upper=self.upper_bounds[continuous],
                axis=1,
            )
            values[continuous] = signed_log1p(values[continuous])
        values = (values - self.center) / self.scale
        return values.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)


@dataclass
class PeerArtifacts:
    model_features: pd.DataFrame
    median: pd.DataFrame
    support: pd.DataFrame
    zscore: pd.DataFrame
    peer_key: pd.Series


def run_multivar_anomaly(
    input_path: str | Path | None = None,
    *,
    source: str = "auto",
    table_key: str = DEFAULT_MULTIVAR_TABLE_KEY,
    results_table_key: str = DEFAULT_MULTIVAR_RESULTS_TABLE_KEY,
    details_table_key: str = DEFAULT_MULTIVAR_DETAILS_TABLE_KEY,
    output_dir: str | Path | None = None,
    scoring_month: str | None = None,
    max_train_rows: int = 150_000,
    max_score_rows: int | None = None,
    chunk_size: int = 250_000,
    random_state: int = 42,
    top_n_reasons: int = 3,
    n_estimators: int = 150,
    persist_oracle_outputs: bool | None = None,
) -> dict:
    """Train on prior months and score one monthly cohort from CSV or Oracle."""

    source = str(source).strip().lower()
    if source == "auto":
        source = "csv" if input_path is not None else "oracle"
    if source not in {"oracle", "csv"}:
        raise ValueError("source must be 'auto', 'oracle', or 'csv'.")

    input_path = resolve_project_path(input_path) if input_path is not None else None
    if source == "csv":
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        month_profile = profile_months(input_path, chunk_size=chunk_size)
        sample_frame = pd.read_csv(input_path, nrows=100_000, encoding="utf-8-sig", decimal=",", low_memory=False)
        sample_frame = normalize_columns(sample_frame)
    else:
        month_profile = profile_months_oracle(table_key=table_key)
        sample_frame = sample_oracle_frame(table_key=table_key, limit=100_000)

    selected_month = _resolve_scoring_month(scoring_month, month_profile)
    prior_rows = int(sum(count for month, count in month_profile["month_counts"].items() if month < selected_month))
    if prior_rows <= 0:
        raise ValueError(f"No prior rows available before scoring month {selected_month.date()}.")

    numeric_source_columns = infer_numeric_source_columns(sample_frame)

    if source == "csv":
        train_df, score_df, prior_df = load_windows(
            input_path,
            selected_month=selected_month,
            prior_rows=prior_rows,
            max_train_rows=max_train_rows,
            max_score_rows=max_score_rows,
            chunk_size=chunk_size,
            random_state=random_state,
            keep_columns=_keep_columns(sample_frame.columns, numeric_source_columns),
        )
    else:
        train_df, score_df, prior_df = load_windows_oracle(
            table_key=table_key,
            selected_month=selected_month,
            prior_rows=prior_rows,
            max_train_rows=max_train_rows,
            max_score_rows=max_score_rows,
            chunk_size=chunk_size,
            random_state=random_state,
            keep_columns=_keep_columns(sample_frame.columns, numeric_source_columns),
        )

    train_features = build_feature_frame(train_df, numeric_source_columns)
    score_features = build_feature_frame(score_df, numeric_source_columns)
    prior_features = build_feature_frame(prior_df, numeric_source_columns) if not prior_df.empty else pd.DataFrame()

    selected_features = select_model_features(train_features, score_features)
    if not selected_features:
        raise ValueError("No usable numeric features remained after coverage/variance filtering.")

    train_peer = build_peer_artifacts(train_df, train_features, selected_features)
    score_peer = build_peer_artifacts(score_df, score_features, selected_features)
    train_model = train_peer.model_features.copy()
    score_model = score_peer.model_features.copy()
    missing_features = detect_missing_features(train_features, score_features, selected_features)
    model_features = list(train_model.columns)

    preprocessor = fit_preprocessor(train_model, model_features)
    x_train = preprocessor.transform(train_model)
    x_score = preprocessor.transform(score_model)

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(x_train)

    train_if_raw = -iso.decision_function(x_train)
    score_if_raw = -iso.decision_function(x_score)
    train_residual_raw = row_top_mean(np.abs(x_train), top_k=3)
    score_residual_raw = row_top_mean(np.abs(x_score), top_k=3)

    if_score = empirical_percentile(train_if_raw, score_if_raw)
    residual_score = empirical_percentile(train_residual_raw, score_residual_raw)
    anomaly_score = np.clip(0.55 * if_score + 0.45 * residual_score, 0, 100)
    band_thresholds = operational_band_thresholds(anomaly_score)

    score_feature_values = score_features[selected_features].copy()
    train_reference = train_features[selected_features].median(axis=0, skipna=True)
    peer_reference = score_features[selected_features].median(axis=0, skipna=True)
    prior_reference = (
        prior_features.set_index(ID_COLUMN)[selected_features]
        if not prior_features.empty and ID_COLUMN in prior_features.columns
        else pd.DataFrame()
    )
    prior_context = (
        normalize_columns(prior_df).set_index(ID_COLUMN)
        if not prior_df.empty and ID_COLUMN in prior_df.columns
        else pd.DataFrame()
    )

    coverage_ratio = score_feature_values.notna().mean(axis=1).fillna(0.0).to_numpy(dtype=float)
    data_gap_score = np.clip((1.0 - coverage_ratio) * 100.0, 0, 100)
    missing_feature_count = score_feature_values.isna().sum(axis=1).to_numpy(dtype=int)
    agreement = 1.0 - (np.abs(if_score - residual_score) / 100.0)
    confidence = np.clip((0.65 * coverage_ratio + 0.35 * agreement) * 100.0, 0, 100)

    feature_severity = pd.DataFrame(np.abs(x_score), columns=model_features, index=score_features.index)
    results = build_results(
        score_df=score_df,
        score_features=score_features,
        feature_severity=feature_severity,
        selected_features=selected_features,
        missing_features=missing_features,
        train_reference=train_reference,
        peer_reference=peer_reference,
        peer_artifacts=score_peer,
        prior_reference=prior_reference,
        prior_context=prior_context,
        anomaly_score=anomaly_score,
        if_score=if_score,
        residual_score=residual_score,
        confidence=confidence,
        coverage_ratio=coverage_ratio,
        data_gap_score=data_gap_score,
        missing_feature_count=missing_feature_count,
        band_thresholds=band_thresholds,
        top_n_reasons=top_n_reasons,
    )

    artifacts = write_outputs(
        results=results,
        output_dir=output_dir,
        scoring_month=selected_month,
        selected_features=selected_features,
        missing_features=missing_features,
        numeric_source_columns=numeric_source_columns,
        month_profile=month_profile,
        train_rows=len(train_df),
        prior_rows=prior_rows,
        input_path=input_path or Path(f"oracle:{table_key}"),
        band_thresholds=band_thresholds,
        model_feature_count=len(model_features),
        peer_feature_count=score_peer.model_features.shape[1],
    )
    if persist_oracle_outputs is None:
        persist_oracle_outputs = source == "oracle"
    oracle_output = None
    if persist_oracle_outputs:
        oracle_output = write_multivar_outputs_to_oracle(
            results=results,
            scoring_month=selected_month,
            source_table_key=table_key,
            results_table_key=results_table_key,
            details_table_key=details_table_key,
            model_feature_count=len(model_features),
            peer_feature_count=score_peer.model_features.shape[1],
        )

    summary = {
        "source": source,
        "input_path": str(input_path) if input_path is not None else None,
        "oracle_table_key": table_key if source == "oracle" else None,
        "oracle_output": oracle_output,
        "scoring_month": selected_month.strftime("%Y-%m-%d"),
        "scored_rows": int(len(results)),
        "train_rows": int(len(train_df)),
        "prior_rows_available": int(prior_rows),
        "selected_feature_count": int(len(selected_features)),
        "model_feature_count": int(len(model_features)),
        "peer_feature_count": int(score_peer.model_features.shape[1]),
        "missing_indicator_count": 0,
        "missing_tracked_feature_count": int(len(missing_features)),
        "alert_counts": results["alert_band"].value_counts().to_dict(),
        "alert_type_counts": results["alert_type"].value_counts().to_dict(),
        "review_queue_counts": results["review_queue"].value_counts().to_dict(),
        "operational_band_policy": OPERATIONAL_BAND_POLICY,
        "operational_band_thresholds": {
            key: round(float(value), 4)
            for key, value in band_thresholds.items()
        },
        "top_score": float(results["anomaly_score"].max()) if len(results) else None,
        "scores_path": str(artifacts.scores_path),
        "top_path": str(artifacts.top_path),
        "summary_path": str(artifacts.summary_path),
        "feature_profile_path": str(artifacts.feature_profile_path),
        "selected_features": selected_features,
        "excluded_feature_columns": sorted(EXCLUDED_FEATURE_COLUMNS),
    }
    with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def load_multivar_csv_to_oracle(
    input_path: str | Path = "anomaly_multivar.csv",
    *,
    table_key: str = DEFAULT_MULTIVAR_TABLE_KEY,
    replace: bool = True,
    delete_local: bool = False,
    chunk_size: int = 100_000,
    batch_size: int = 10_000,
) -> dict:
    """Create/replace the multivar Oracle table and stream CSV rows into it."""

    path = resolve_project_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    ensure_multivar_oracle_table(table_key=table_key, replace=replace)
    inserted = 0
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        encoding="utf-8-sig",
        decimal=",",
        low_memory=False,
    ):
        frame = prepare_multivar_oracle_frame(chunk)
        inserted += insert_multivar_oracle_rows(table_key, frame, batch_size=batch_size)
        logger.info("Loaded %s rows into Oracle table %s", inserted, table_key)

    oracle_rows = count_multivar_oracle_rows(table_key)
    if delete_local and oracle_rows >= inserted and inserted > 0:
        path.unlink()

    return {
        "input_path": str(path),
        "table_key": table_key,
        "oracle_table": qualified_multivar_table_name(table_key),
        "inserted_rows": int(inserted),
        "oracle_rows": int(oracle_rows),
        "deleted_local": bool(delete_local and not path.exists()),
    }


def ensure_multivar_oracle_table(*, table_key: str = DEFAULT_MULTIVAR_TABLE_KEY, replace: bool = False) -> None:
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        table_full = ora._qualified_table_name(table_key)
        exists = ora._table_exists(table_key)
        with ora.connection.cursor() as cursor:
            if exists and replace:
                cursor.execute(f"TRUNCATE TABLE {table_full}")
                ora.logger.info("Truncated %s", table_full)
            elif not exists:
                cursor.execute(multivar_table_ddl(ora, table_key))
                ora.logger.info("Created %s", table_full)
        ora.connection.commit()


def insert_multivar_oracle_rows(table_key: str, frame: pd.DataFrame, *, batch_size: int) -> int:
    if frame.empty:
        return 0
    config = load_config()
    secrets = load_secrets()
    columns = [column for column in MULTIVAR_BASE_COLUMNS if column in frame.columns]
    with OracleConnector(config, secrets) as ora:
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


def profile_months_oracle(*, table_key: str = DEFAULT_MULTIVAR_TABLE_KEY) -> dict:
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        sql = f"""
            SELECT TRUNC({TIME_COLUMN.upper()}) AS {TIME_COLUMN.upper()}, COUNT(*) AS ROW_COUNT
            FROM {ora._qualified_table_name(table_key)}
            GROUP BY TRUNC({TIME_COLUMN.upper()})
            ORDER BY TRUNC({TIME_COLUMN.upper()})
        """
        frame = ora._read_query(sql)
    counts = {
        pd.Timestamp(row[TIME_COLUMN]).normalize(): int(row["row_count"])
        for _, row in frame.iterrows()
    }
    if not counts:
        raise ValueError(f"No rows found in Oracle table key {table_key}.")
    return {
        "total_rows": int(sum(counts.values())),
        "month_counts": counts,
    }


def sample_oracle_frame(*, table_key: str = DEFAULT_MULTIVAR_TABLE_KEY, limit: int = 100_000) -> pd.DataFrame:
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        sql = f"SELECT * FROM {ora._qualified_table_name(table_key)} FETCH FIRST {int(limit)} ROWS ONLY"
        return normalize_columns(ora._read_query(sql))


def load_windows_oracle(
    *,
    table_key: str,
    selected_month: pd.Timestamp,
    prior_rows: int,
    max_train_rows: int,
    max_score_rows: int | None,
    chunk_size: int,
    random_state: int,
    keep_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    score_parts = []
    latest_prior_by_id = pd.DataFrame()
    train_frac = min(1.0, float(max_train_rows) / float(max(prior_rows, 1)))
    rng = np.random.default_rng(random_state)

    for chunk in iter_multivar_oracle_chunks(table_key=table_key, columns=keep_columns, chunk_size=chunk_size):
        chunk[TIME_COLUMN] = parse_dates(chunk[TIME_COLUMN])
        prior = chunk[chunk[TIME_COLUMN] < selected_month]
        scoring = chunk[chunk[TIME_COLUMN] == selected_month]

        if not prior.empty:
            if train_frac >= 1.0:
                sample = prior
            else:
                seed = int(rng.integers(0, 2**31 - 1))
                sample = prior.sample(frac=train_frac, random_state=seed)
            if not sample.empty:
                train_parts.append(sample)
            latest_prior_by_id = update_latest_by_id(latest_prior_by_id, prior)

        if not scoring.empty:
            score_parts.append(scoring)

    if not train_parts:
        raise ValueError("No training sample was collected from prior months.")
    if not score_parts:
        raise ValueError(f"No scoring rows found for {selected_month.date()}.")

    train_df = pd.concat(train_parts, ignore_index=True)
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=random_state).reset_index(drop=True)
    score_df = pd.concat(score_parts, ignore_index=True)
    if max_score_rows is not None and len(score_df) > max_score_rows:
        score_df = score_df.sample(n=max_score_rows, random_state=random_state).reset_index(drop=True)

    score_ids = set(score_df[ID_COLUMN].astype(str))
    prior_df = (
        latest_prior_by_id[latest_prior_by_id[ID_COLUMN].astype(str).isin(score_ids)].copy()
        if not latest_prior_by_id.empty
        else latest_prior_by_id
    )
    return train_df.reset_index(drop=True), score_df.reset_index(drop=True), prior_df.reset_index(drop=True)


def iter_multivar_oracle_chunks(
    *,
    table_key: str,
    columns: list[str],
    chunk_size: int,
):
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        select_columns = ", ".join(column.upper() for column in columns)
        sql = f"SELECT {select_columns} FROM {ora._qualified_table_name(table_key)}"
        with ora.connection.cursor() as cursor:
            cursor.execute(sql)
            output_columns = [description[0].lower() for description in cursor.description]
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=output_columns)


def count_multivar_oracle_rows(table_key: str = DEFAULT_MULTIVAR_TABLE_KEY) -> int:
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        frame = ora._read_query(f"SELECT COUNT(*) AS ROW_COUNT FROM {ora._qualified_table_name(table_key)}")
    return int(frame.iloc[0]["row_count"])


def qualified_multivar_table_name(table_key: str = DEFAULT_MULTIVAR_TABLE_KEY) -> str:
    config = load_config()
    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        return ora._qualified_table_name(table_key)


def multivar_table_ddl(ora: OracleConnector, table_key: str) -> str:
    columns = []
    for column in MULTIVAR_BASE_COLUMNS:
        ddl_type = MULTIVAR_COLUMN_TYPES.get(column, "NUMBER(24,8)")
        nullable = " NOT NULL" if column in {TIME_COLUMN, ID_COLUMN} else ""
        columns.append(f"{column.upper()} {ddl_type}{nullable}")
    columns.append("DATA_TIME TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL")
    columns.append("CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL")
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    columns.append(f"CONSTRAINT {pk_name} PRIMARY KEY ({TIME_COLUMN.upper()}, {ID_COLUMN.upper()})")
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            {", ".join(columns)}
        )
    """


def prepare_multivar_oracle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_columns(frame)
    result = pd.DataFrame(index=normalized.index)
    for column in MULTIVAR_BASE_COLUMNS:
        if column not in normalized.columns:
            result[column] = pd.NA
            continue
        if MULTIVAR_COLUMN_TYPES.get(column) == "DATE":
            result[column] = parse_dates(normalized[column])
        elif MULTIVAR_COLUMN_TYPES.get(column) == "TIMESTAMP":
            result[column] = pd.to_datetime(normalized[column], dayfirst=True, errors="coerce")
        elif MULTIVAR_COLUMN_TYPES.get(column, "").startswith("VARCHAR2"):
            result[column] = normalized[column].where(normalized[column].isna(), normalized[column].astype(str).str.strip())
        else:
            result[column] = coerce_numeric(normalized[column])
    return result


def profile_months(input_path: Path, *, chunk_size: int = 250_000) -> dict:
    counts: dict[pd.Timestamp, int] = {}
    total_rows = 0
    for chunk in pd.read_csv(
        input_path,
        chunksize=chunk_size,
        usecols=lambda name: str(name).strip().lower() == TIME_COLUMN,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk = normalize_columns(chunk)
        dates = parse_dates(chunk[TIME_COLUMN])
        total_rows += len(chunk)
        for month, count in dates.value_counts(dropna=True).items():
            month = pd.Timestamp(month).normalize()
            counts[month] = counts.get(month, 0) + int(count)
    if not counts:
        raise ValueError(f"No valid {TIME_COLUMN} values found in {input_path}.")
    return {
        "total_rows": int(total_rows),
        "month_counts": {month: count for month, count in sorted(counts.items())},
    }


def load_windows(
    input_path: Path,
    *,
    selected_month: pd.Timestamp,
    prior_rows: int,
    max_train_rows: int,
    max_score_rows: int | None,
    chunk_size: int,
    random_state: int,
    keep_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    score_parts = []
    latest_prior_by_id = pd.DataFrame()
    train_frac = min(1.0, float(max_train_rows) / float(max(prior_rows, 1)))
    rng = np.random.default_rng(random_state)

    for chunk in pd.read_csv(
        input_path,
        chunksize=chunk_size,
        usecols=lambda name: str(name).strip().lower() in keep_columns,
        encoding="utf-8-sig",
        decimal=",",
        low_memory=False,
    ):
        chunk = normalize_columns(chunk)
        chunk[TIME_COLUMN] = parse_dates(chunk[TIME_COLUMN])
        prior = chunk[chunk[TIME_COLUMN] < selected_month]
        scoring = chunk[chunk[TIME_COLUMN] == selected_month]

        if not prior.empty:
            if train_frac >= 1.0:
                sample = prior
            else:
                seed = int(rng.integers(0, 2**31 - 1))
                sample = prior.sample(frac=train_frac, random_state=seed)
            if not sample.empty:
                train_parts.append(sample)
            latest_prior_by_id = update_latest_by_id(latest_prior_by_id, prior)

        if not scoring.empty:
            score_parts.append(scoring)

    if not train_parts:
        raise ValueError("No training sample was collected from prior months.")
    if not score_parts:
        raise ValueError(f"No scoring rows found for {selected_month.date()}.")

    train_df = pd.concat(train_parts, ignore_index=True)
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=random_state).reset_index(drop=True)
    score_df = pd.concat(score_parts, ignore_index=True)
    if max_score_rows is not None and len(score_df) > max_score_rows:
        score_df = score_df.sample(n=max_score_rows, random_state=random_state).reset_index(drop=True)

    score_ids = set(score_df[ID_COLUMN].astype(str))
    if latest_prior_by_id.empty:
        prior_df = latest_prior_by_id
    else:
        prior_df = latest_prior_by_id[latest_prior_by_id[ID_COLUMN].astype(str).isin(score_ids)].copy()
    return train_df.reset_index(drop=True), score_df.reset_index(drop=True), prior_df.reset_index(drop=True)


def infer_numeric_source_columns(frame: pd.DataFrame, *, min_coverage: float = 0.20) -> list[str]:
    frame = normalize_columns(frame)
    reserved = {ID_COLUMN, TIME_COLUMN} | EXCLUDED_FEATURE_COLUMNS | DESCRIPTOR_COLUMNS
    selected = []
    for column in frame.columns:
        if column in reserved:
            continue
        values = coerce_numeric(frame[column])
        if float(values.notna().mean()) < min_coverage:
            continue
        if int(values.nunique(dropna=True)) <= 1:
            continue
        selected.append(column)

    for column in DERIVED_INPUT_COLUMNS:
        if column in frame.columns and column not in selected and column not in reserved:
            selected.append(column)
    return list(dict.fromkeys(selected))


def build_feature_frame(frame: pd.DataFrame, source_columns: Iterable[str]) -> pd.DataFrame:
    normalized = normalize_columns(frame)
    result = pd.DataFrame(index=normalized.index)
    result[ID_COLUMN] = normalized[ID_COLUMN].astype(str)
    result[TIME_COLUMN] = parse_dates(normalized[TIME_COLUMN])

    for column in source_columns:
        if column in normalized.columns:
            result[column] = coerce_numeric(normalized[column])

    result["memzuc_limit_utilization"] = safe_divide(result.get("memzuc_total_risk"), result.get("memzuc_total_limit"))
    result["memzuc_st_mt_cash_share"] = safe_divide(result.get("memzuc_st_mt_cash_risk"), result.get("memzuc_total_risk"))
    result["bank_risk_to_assets"] = safe_divide(result.get("bank_total_risk"), result.get("toplam_varlik_ttr"))
    result["memzuc_risk_to_assets"] = safe_divide(result.get("memzuc_total_risk"), result.get("toplam_varlik_ttr"))
    result["l1y_trade_receivables_to_sales"] = safe_divide(
        result.get("fs_trade_receivables_l1y"),
        result.get("fs_net_sales_cumulative_l1y"),
    )
    result["l1y_notes_receivable_to_sales"] = safe_divide(
        result.get("fs_notes_receivable_l1y"),
        result.get("fs_net_sales_cumulative_l1y"),
    )
    result["q_trade_receivables_to_sales"] = safe_divide(
        result.get("fs_trade_receivables_q"),
        result.get("fs_net_sales_cumulative_q"),
    )
    result["q_notes_receivable_to_sales"] = safe_divide(
        result.get("fs_notes_receivable_q"),
        result.get("fs_net_sales_cumulative_q"),
    )
    result["l1y_profit_margin"] = safe_divide(
        result.get("fs_net_profit_cumulative_l1y"),
        result.get("fs_net_sales_cumulative_l1y"),
    )
    result["q_profit_margin"] = safe_divide(
        result.get("fs_net_profit_cumulative_q"),
        result.get("fs_net_sales_cumulative_q"),
    )
    result["l1y_equity_to_assets"] = safe_divide(result.get("equity_l1y"), result.get("toplam_varlik_ttr"))
    result["q_equity_to_assets"] = safe_divide(result.get("fs_equity_q"), result.get("toplam_varlik_ttr"))
    result["q_to_l1y_sales_ratio"] = safe_divide(
        result.get("fs_net_sales_cumulative_q"),
        result.get("fs_net_sales_cumulative_l1y"),
    )
    result["q_to_l1y_profit_ratio"] = safe_divide(
        result.get("fs_net_profit_cumulative_q"),
        result.get("fs_net_profit_cumulative_l1y").abs()
        if "fs_net_profit_cumulative_l1y" in result
        else None,
    )
    result["q_ebitda_margin"] = safe_divide(
        result.get("fs_ebitda_cumulative_q"),
        result.get("fs_net_sales_cumulative_q"),
    )
    result["l1y_debt_to_sales"] = safe_divide(result.get("bank_total_risk"), result.get("fs_net_sales_cumulative_l1y"))
    result["q_debt_to_sales"] = safe_divide(result.get("bank_total_risk"), result.get("fs_net_sales_cumulative_q"))
    result["memzuc_debt_to_l1y_sales"] = safe_divide(result.get("memzuc_total_risk"), result.get("fs_net_sales_cumulative_l1y"))
    result["memzuc_debt_to_q_sales"] = safe_divide(result.get("memzuc_total_risk"), result.get("fs_net_sales_cumulative_q"))
    result["memzuc_to_bank_risk_ratio"] = safe_divide(result.get("memzuc_total_risk"), result.get("bank_total_risk"))
    result["bank_to_memzuc_risk_ratio"] = safe_divide(result.get("bank_total_risk"), result.get("memzuc_total_risk"))
    result["l1y_trade_receivables_to_assets"] = safe_divide(result.get("fs_trade_receivables_l1y"), result.get("toplam_varlik_ttr"))
    result["l1y_notes_receivable_to_assets"] = safe_divide(result.get("fs_notes_receivable_l1y"), result.get("toplam_varlik_ttr"))
    result["q_trade_receivables_to_assets"] = safe_divide(result.get("fs_trade_receivables_q"), result.get("toplam_varlik_ttr"))
    result["q_notes_receivable_to_assets"] = safe_divide(result.get("fs_notes_receivable_q"), result.get("toplam_varlik_ttr"))
    result["l1y_suspicious_receivables_to_sales"] = safe_divide(
        result.get("supheli_ticari_alacaklar_l1y"),
        result.get("fs_net_sales_cumulative_l1y"),
    )
    result["q_suspicious_receivables_to_sales"] = safe_divide(
        result.get("supheli_alacaklar_q"),
        result.get("fs_net_sales_cumulative_q"),
    )
    result["l1y_suspicious_to_trade_receivables"] = safe_divide(
        result.get("supheli_ticari_alacaklar_l1y"),
        result.get("fs_trade_receivables_l1y"),
    )
    result["q_suspicious_to_trade_receivables"] = safe_divide(
        result.get("supheli_alacaklar_q"),
        result.get("fs_trade_receivables_q"),
    )
    result["q_to_l1y_equity_ratio"] = safe_divide(result.get("fs_equity_q"), result.get("equity_l1y"))
    result["q_to_l1y_trade_receivables_ratio"] = safe_divide(
        result.get("fs_trade_receivables_q"),
        result.get("fs_trade_receivables_l1y"),
    )
    result["q_to_l1y_notes_receivable_ratio"] = safe_divide(
        result.get("fs_notes_receivable_q"),
        result.get("fs_notes_receivable_l1y"),
    )
    result["q_to_l1y_suspicious_receivables_ratio"] = safe_divide(
        result.get("supheli_alacaklar_q"),
        result.get("supheli_ticari_alacaklar_l1y"),
    )
    result["pd_ratio"] = safe_divide(result.get("irb_rating_pd"), result.get("irb_model_pd"))
    result["pd_to_rating_group"] = safe_divide(result.get("irb_rating_pd"), result.get("rating_group"))
    result["internal_tkn_to_assets"] = safe_divide(result.get("gunceltkn_dgr"), result.get("toplam_varlik_ttr"))
    result["internal_tbe_to_assets"] = safe_divide(result.get("gunceltbe_dgr"), result.get("toplam_varlik_ttr"))
    result["internal_tkn_to_sales"] = safe_divide(result.get("gunceltkn_dgr"), result.get("fs_net_sales_cumulative_l1y"))
    result["internal_tbe_to_sales"] = safe_divide(result.get("gunceltbe_dgr"), result.get("fs_net_sales_cumulative_l1y"))
    result["internal_tkn_tbe_ratio"] = safe_divide(result.get("gunceltkn_dgr"), result.get("gunceltbe_dgr"))
    return result.replace([np.inf, -np.inf], np.nan)


def build_peer_artifacts(
    raw_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    selected_features: list[str],
    *,
    min_support: int = PEER_MIN_SUPPORT,
) -> PeerArtifacts:
    context = build_peer_context(raw_frame, feature_frame)
    peer_model = pd.DataFrame(index=feature_frame.index)
    peer_median = pd.DataFrame(index=feature_frame.index)
    peer_support = pd.DataFrame(index=feature_frame.index)
    peer_zscore = pd.DataFrame(index=feature_frame.index)

    for feature in selected_features:
        display_median, transformed_median, support, scale = peer_reference_for_feature(
            context,
            pd.to_numeric(feature_frame[feature], errors="coerce"),
            min_support=min_support,
        )
        actual = pd.to_numeric(feature_frame[feature], errors="coerce")
        transformed_actual = peer_transform_values(actual)
        zscore = ((transformed_actual - transformed_median) / scale.replace(0.0, np.nan)).clip(-PEER_Z_CLIP, PEER_Z_CLIP)
        peer_median[feature] = display_median
        peer_support[feature] = support.fillna(0).astype(int)
        peer_zscore[feature] = zscore
        peer_model[f"{feature}{PEER_FEATURE_SUFFIX}"] = zscore

    return PeerArtifacts(
        model_features=peer_model,
        median=peer_median,
        support=peer_support,
        zscore=peer_zscore,
        peer_key=context["peer_key"],
    )


def build_peer_context(raw_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    raw = normalize_columns(raw_frame)
    context = pd.DataFrame(index=feature_frame.index)
    context[TIME_COLUMN] = parse_dates(raw[TIME_COLUMN])
    context["peer_segment"] = normalized_category(
        raw.get("musteri_segment"),
        default="SEG_NA",
        index=feature_frame.index,
    )
    context["peer_rating"] = normalized_category(
        raw.get("rating_group"),
        default="RATING_NA",
        index=feature_frame.index,
    )
    sector_source = raw.get("cst_nace_code_id")
    if sector_source is None:
        sector_source = raw.get("cst_nace_code")
    if sector_source is None:
        sector_source = raw.get("cst_sector")
    context["peer_sector"] = normalized_category(
        sector_source,
        default="SECTOR_NA",
        index=feature_frame.index,
    )
    context["peer_size"] = monthly_size_bucket(feature_frame)
    context["peer_key"] = (
        context["peer_segment"].astype(str)
        + "|"
        + context["peer_rating"].astype(str)
        + "|"
        + context["peer_sector"].astype(str)
        + "|"
        + context["peer_size"].astype(str)
    )
    return context


def peer_reference_for_feature(
    context: pd.DataFrame,
    values: pd.Series,
    *,
    min_support: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    transformed_values = peer_transform_values(values)
    result_median = pd.Series(np.nan, index=values.index, dtype=float)
    result_transformed_median = pd.Series(np.nan, index=values.index, dtype=float)
    result_support = pd.Series(0, index=values.index, dtype=float)
    result_scale = pd.Series(np.nan, index=values.index, dtype=float)
    hierarchy = [
        [TIME_COLUMN, "peer_segment", "peer_rating", "peer_sector", "peer_size"],
        [TIME_COLUMN, "peer_segment", "peer_rating", "peer_size"],
        [TIME_COLUMN, "peer_segment", "peer_sector"],
        [TIME_COLUMN, "peer_segment", "peer_size"],
        [TIME_COLUMN, "peer_segment"],
        [TIME_COLUMN],
    ]
    global_scale = robust_scale(transformed_values)
    for keys in hierarchy:
        median, transformed_median, support, scale = grouped_median_support_scale(
            context,
            values,
            transformed_values,
            keys,
            global_scale,
        )
        eligible = result_median.isna() & support.ge(min_support) & median.notna()
        result_median.loc[eligible] = median.loc[eligible]
        result_transformed_median.loc[eligible] = transformed_median.loc[eligible]
        result_support.loc[eligible] = support.loc[eligible]
        result_scale.loc[eligible] = scale.loc[eligible]

    fallback_median = values.median(skipna=True)
    fallback_transformed_median = transformed_values.median(skipna=True)
    result_median = result_median.fillna(fallback_median if pd.notna(fallback_median) else 0.0)
    result_transformed_median = result_transformed_median.fillna(
        fallback_transformed_median if pd.notna(fallback_transformed_median) else 0.0
    )
    result_support = result_support.fillna(int(values.notna().sum()))
    result_scale = result_scale.fillna(global_scale).replace(0.0, global_scale)
    return result_median, result_transformed_median, result_support, result_scale


def grouped_median_support_scale(
    context: pd.DataFrame,
    values: pd.Series,
    transformed_values: pd.Series,
    keys: list[str],
    global_scale: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    helper = context[keys].copy()
    helper["_value"] = values
    helper["_transformed_value"] = transformed_values
    grouped = helper.groupby(keys, dropna=False)["_value"]
    median = grouped.transform("median")
    transformed_grouped = helper.groupby(keys, dropna=False)["_transformed_value"]
    transformed_median = transformed_grouped.transform("median")
    support = grouped.transform("count").astype(float)
    abs_dev = (transformed_values - transformed_median).abs()
    helper["_abs_dev"] = abs_dev
    mad = helper.groupby(keys, dropna=False)["_abs_dev"].transform("median")
    scale = (mad * 1.4826).replace(0.0, np.nan).fillna(global_scale)
    return median, transformed_median, support, scale


def peer_transform_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    return np.sign(values) * np.log1p(np.abs(values))


def robust_scale(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if values.empty:
        return 1.0
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad > 1e-9:
        return max(mad * 1.4826, 1e-6)
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    if iqr > 1e-9:
        return max(iqr / 1.349, 1e-6)
    std = float(values.std())
    return max(std, 1.0)


def normalized_category(series: pd.Series | None, *, default: str, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    normalized = pd.Series(series, copy=False).astype("string").str.strip()
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    normalized.index = index
    return normalized.fillna(default).astype(str)


def monthly_size_bucket(feature_frame: pd.DataFrame) -> pd.Series:
    base = feature_frame.get("toplam_varlik_ttr")
    if base is None or pd.to_numeric(base, errors="coerce").notna().mean() < 0.5:
        bank_risk = feature_frame.get("bank_total_risk")
        memzuc_risk = feature_frame.get("memzuc_total_risk")
        if bank_risk is not None or memzuc_risk is not None:
            base = pd.Series(0.0, index=feature_frame.index)
            if bank_risk is not None:
                base = base + pd.to_numeric(bank_risk, errors="coerce").fillna(0.0)
            if memzuc_risk is not None:
                base = base + pd.to_numeric(memzuc_risk, errors="coerce").fillna(0.0)
    if base is None:
        return pd.Series("SIZE_NA", index=feature_frame.index)
    values = pd.to_numeric(base, errors="coerce")
    months = pd.to_datetime(feature_frame[TIME_COLUMN], errors="coerce")
    result = pd.Series("SIZE_NA", index=feature_frame.index, dtype=object)
    for _, idx in months.groupby(months, dropna=False).groups.items():
        bucket_values = values.loc[idx]
        valid = bucket_values.notna()
        if valid.sum() < 4:
            continue
        try:
            buckets = pd.qcut(
                np.log1p(bucket_values.loc[valid].clip(lower=0.0)),
                q=4,
                labels=["S1", "S2", "S3", "S4"],
                duplicates="drop",
            )
        except ValueError:
            continue
        result.loc[bucket_values.loc[valid].index] = buckets.astype(str)
    return result


def select_model_features(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    *,
    min_train_coverage: float = 0.50,
) -> list[str]:
    candidates = [
        column
        for column in train_features.columns
        if column not in {ID_COLUMN, TIME_COLUMN}
        and column not in RAW_MODEL_EXCLUDE_COLUMNS
        and column.startswith(DERIVED_FEATURE_PREFIXES)
    ]
    selected = []
    for column in candidates:
        train_series = pd.to_numeric(train_features[column], errors="coerce")
        score_series = pd.to_numeric(score_features[column], errors="coerce")
        if float(train_series.notna().mean()) < min_train_coverage:
            continue
        combined = pd.concat([train_series, score_series], ignore_index=True)
        if int(combined.nunique(dropna=True)) <= 1:
            continue
        selected.append(column)
    return selected


def add_missing_indicators(
    train_model: pd.DataFrame,
    score_model: pd.DataFrame,
    selected_features: list[str],
    *,
    train_source: pd.DataFrame,
    score_source: pd.DataFrame,
    min_missing_rate: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_model = train_model.copy()
    score_model = score_model.copy()
    missing_features = []
    for feature in selected_features:
        train_missing = float(train_source[feature].isna().mean())
        score_missing = float(score_source[feature].isna().mean())
        if max(train_missing, score_missing) < min_missing_rate:
            continue
        indicator = f"{feature}__missing_flag"
        train_model[indicator] = train_source[feature].isna().astype(float)
        score_model[indicator] = score_source[feature].isna().astype(float)
        missing_features.append(feature)
    return train_model, score_model, missing_features


def detect_missing_features(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    selected_features: list[str],
    *,
    min_missing_rate: float = 0.005,
) -> list[str]:
    missing_features = []
    for feature in selected_features:
        train_missing = float(train_features[feature].isna().mean())
        score_missing = float(score_features[feature].isna().mean())
        if max(train_missing, score_missing) >= min_missing_rate:
            missing_features.append(feature)
    return missing_features


def fit_preprocessor(frame: pd.DataFrame, features: list[str]) -> RobustPreprocessor:
    values = frame[features].copy()
    continuous_features = {feature for feature in features if not feature.endswith("__missing_flag")}
    fill_values = values.median(axis=0, skipna=True).fillna(0.0)
    filled = values.fillna(fill_values)
    lower_bounds = filled.quantile(0.01, axis=0).fillna(fill_values)
    upper_bounds = filled.quantile(0.99, axis=0).fillna(fill_values)
    if continuous_features:
        continuous = list(continuous_features)
        filled[continuous] = filled[continuous].clip(
            lower=lower_bounds[continuous],
            upper=upper_bounds[continuous],
            axis=1,
        )
        filled[continuous] = signed_log1p(filled[continuous])
    center = filled.median(axis=0, skipna=True).fillna(0.0)
    q75 = filled.quantile(0.75, axis=0)
    q25 = filled.quantile(0.25, axis=0)
    scale = (q75 - q25).replace(0.0, np.nan)
    fallback = filled.std(axis=0, skipna=True).replace(0.0, np.nan)
    scale = scale.fillna(fallback).fillna(1.0)
    return RobustPreprocessor(
        features=features,
        continuous_features=continuous_features,
        fill_values=fill_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        center=center,
        scale=scale,
    )


def build_results(
    *,
    score_df: pd.DataFrame,
    score_features: pd.DataFrame,
    feature_severity: pd.DataFrame,
    selected_features: list[str],
    missing_features: list[str],
    train_reference: pd.Series,
    peer_reference: pd.Series,
    peer_artifacts: PeerArtifacts,
    prior_reference: pd.DataFrame,
    prior_context: pd.DataFrame,
    anomaly_score: np.ndarray,
    if_score: np.ndarray,
    residual_score: np.ndarray,
    confidence: np.ndarray,
    coverage_ratio: np.ndarray,
    data_gap_score: np.ndarray,
    missing_feature_count: np.ndarray,
    band_thresholds: dict[str, float],
    top_n_reasons: int,
) -> pd.DataFrame:
    context_cols = [column for column in CONTEXT_COLUMNS if column in score_df.columns]
    result = score_df[[ID_COLUMN, TIME_COLUMN, *context_cols]].copy()
    result[TIME_COLUMN] = parse_dates(result[TIME_COLUMN]).dt.strftime("%Y-%m-%d")
    result["anomaly_score"] = np.round(anomaly_score, 1)
    result["alert_band"] = assign_operational_bands(anomaly_score, band_thresholds)
    result["if_score"] = np.round(if_score, 1)
    result["residual_score"] = np.round(residual_score, 1)
    result["confidence"] = np.round(confidence, 1)
    result["coverage_ratio"] = np.round(coverage_ratio, 4)
    result["data_gap_score"] = np.round(data_gap_score, 1)
    result["missing_feature_count"] = missing_feature_count

    reasons = []
    detail_payloads = []
    alert_types = []
    alert_band_values = result["alert_band"].tolist()
    for row_position, row_index in enumerate(score_features.index):
        severity = feature_severity.loc[row_index]
        total = float(severity.sum()) or 1.0
        row_reasons = []
        row_details = []
        family_severity = aggregate_family_severity(severity, selected_features)

        for base_feature, family_value in family_severity.items():
            contribution = float(family_value) / total * 100.0
            detail = build_reason_detail(
                row_id=str(score_features.iloc[row_position][ID_COLUMN]),
                base_feature=base_feature,
                is_missing_reason=False,
                actual=score_features.iloc[row_position][base_feature],
                prior_reference=lookup_prior_reference(
                    prior_reference,
                    str(score_features.iloc[row_position][ID_COLUMN]),
                    base_feature,
                ),
                peer_reference=peer_reference.get(base_feature, np.nan),
                peer_median=peer_artifacts.median.loc[row_index, base_feature],
                peer_support=peer_artifacts.support.loc[row_index, base_feature],
                peer_z=peer_artifacts.zscore.loc[row_index, base_feature],
                train_reference=train_reference.get(base_feature, np.nan),
                contribution_pct=contribution,
                component_contributions=component_contribution_pct(severity, base_feature, total),
                current_context=score_df.iloc[row_position],
                prior_context=lookup_prior_context(
                    prior_context,
                    str(score_features.iloc[row_position][ID_COLUMN]),
                ),
            )
            row_details.append(detail)
            row_reasons.append(format_reason(detail))
            if len(row_reasons) >= top_n_reasons:
                break
        reasons.append(row_reasons)
        detail_payloads.append(row_details)
        alert_types.append(
            classify_alert_type(
                row_details,
                coverage_ratio[row_position],
                data_gap_score[row_position],
                alert_band_values[row_position],
            )
        )

    result["alert_type"] = alert_types
    result["review_queue"] = [
        review_queue_for(band, alert_type, gap_score)
        for band, alert_type, gap_score in zip(
            result["alert_band"],
            result["alert_type"],
            result["data_gap_score"],
        )
    ]
    result["reason_details"] = [json.dumps(item, ensure_ascii=False) for item in detail_payloads]
    for idx in range(top_n_reasons):
        result[f"reason_{idx + 1}"] = [items[idx] if idx < len(items) else None for items in reasons]
    result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    result["rank_in_run"] = np.arange(1, len(result) + 1)
    return result


def write_outputs(
    *,
    results: pd.DataFrame,
    output_dir: str | Path | None,
    scoring_month: pd.Timestamp,
    selected_features: list[str],
    missing_features: list[str],
    numeric_source_columns: list[str],
    month_profile: dict,
    train_rows: int,
    prior_rows: int,
    input_path: Path,
    band_thresholds: dict[str, float],
    model_feature_count: int,
    peer_feature_count: int,
) -> MultivarRunArtifacts:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    month_label = scoring_month.strftime("%Y%m%d")
    base_dir = resolve_project_path(output_dir or Path("runtime") / "multivar_anomaly" / f"{month_label}_{stamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    scores_path = base_dir / f"anomaly_multivar_scores_{month_label}.csv"
    top_path = base_dir / f"anomaly_multivar_top_{month_label}.csv"
    summary_path = base_dir / f"anomaly_multivar_summary_{month_label}.json"
    feature_profile_path = base_dir / f"anomaly_multivar_feature_profile_{month_label}.json"

    results.to_csv(scores_path, index=False, encoding="utf-8-sig")
    results.head(500).to_csv(top_path, index=False, encoding="utf-8-sig")

    feature_profile = {
        "input_path": str(input_path),
        "scoring_month": scoring_month.strftime("%Y-%m-%d"),
        "total_rows": int(month_profile["total_rows"]),
        "train_rows": int(train_rows),
        "prior_rows_available": int(prior_rows),
        "numeric_source_columns": numeric_source_columns,
        "selected_features": selected_features,
        "selected_feature_labels": {
            feature: feature_label(feature)
            for feature in selected_features
        },
        "feature_policy": "ratio_only_no_product_sum_or_difference",
        "model_feature_count": int(model_feature_count),
        "peer_feature_count": int(peer_feature_count),
        "peer_min_support": int(PEER_MIN_SUPPORT),
        "missing_indicator_base_features": missing_features,
        "operational_band_policy": OPERATIONAL_BAND_POLICY,
        "operational_band_thresholds": {
            key: round(float(value), 4)
            for key, value in band_thresholds.items()
        },
        "excluded_feature_columns": sorted(EXCLUDED_FEATURE_COLUMNS),
        "descriptor_columns_not_modeled": sorted(DESCRIPTOR_COLUMNS),
        "month_counts": {
            month.strftime("%Y-%m-%d"): int(count)
            for month, count in month_profile["month_counts"].items()
        },
    }
    with open(feature_profile_path, "w", encoding="utf-8") as handle:
        json.dump(feature_profile, handle, ensure_ascii=False, indent=2)

    return MultivarRunArtifacts(
        output_dir=base_dir,
        scores_path=scores_path,
        top_path=top_path,
        summary_path=summary_path,
        feature_profile_path=feature_profile_path,
    )


def write_multivar_outputs_to_oracle(
    *,
    results: pd.DataFrame,
    scoring_month: pd.Timestamp,
    source_table_key: str,
    results_table_key: str = DEFAULT_MULTIVAR_RESULTS_TABLE_KEY,
    details_table_key: str = DEFAULT_MULTIVAR_DETAILS_TABLE_KEY,
    model_feature_count: int,
    peer_feature_count: int,
    batch_size: int = 1000,
) -> dict:
    """Persist final multivar scores and top reason details into Oracle."""

    config = load_config()
    oracle_output_cfg = (
        (config.get("multivar_anomaly", {}) or {})
        .get("outputs", {})
        .get("oracle", {})
    )
    results_table_key = oracle_output_cfg.get("results_table_key", results_table_key)
    details_table_key = oracle_output_cfg.get("details_table_key", details_table_key)

    run_id = f"multivar_{scoring_month.strftime('%Y%m%d')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result_frame = prepare_multivar_oracle_results(
        results,
        run_id=run_id,
        source_table_key=source_table_key,
        model_feature_count=model_feature_count,
        peer_feature_count=peer_feature_count,
    )
    detail_frame = prepare_multivar_oracle_details(results, run_id=run_id)

    secrets = load_secrets()
    with OracleConnector(config, secrets) as ora:
        ensure_multivar_output_tables(
            ora,
            results_table_key=results_table_key,
            details_table_key=details_table_key,
        )
        deleted = delete_multivar_output_month(
            ora,
            scoring_month=scoring_month,
            results_table_key=results_table_key,
            details_table_key=details_table_key,
        )
        inserted_results = insert_multivar_output_frame(
            ora,
            results_table_key,
            result_frame,
            batch_size=batch_size,
        )
        inserted_details = insert_multivar_output_frame(
            ora,
            details_table_key,
            detail_frame,
            batch_size=batch_size,
        )

        return {
            "backend": "oracle",
            "run_id": run_id,
            "results_table_key": results_table_key,
            "details_table_key": details_table_key,
            "results_table": ora._qualified_table_name(results_table_key),
            "details_table": ora._qualified_table_name(details_table_key),
            "deleted_results": int(deleted["results"]),
            "deleted_details": int(deleted["details"]),
            "inserted_results": int(inserted_results),
            "inserted_details": int(inserted_details),
        }


def ensure_multivar_output_tables(
    ora: OracleConnector,
    *,
    results_table_key: str,
    details_table_key: str,
) -> None:
    ddls = {
        results_table_key: multivar_results_table_ddl(ora, results_table_key),
        details_table_key: multivar_details_table_ddl(ora, details_table_key),
    }
    with ora.connection.cursor() as cursor:
        for table_key, ddl in ddls.items():
            if ora._table_exists(table_key):
                continue
            cursor.execute(ddl)
            ora.logger.info("Created %s", ora._qualified_table_name(table_key))
    ora.connection.commit()


def delete_multivar_output_month(
    ora: OracleConnector,
    *,
    scoring_month: pd.Timestamp,
    results_table_key: str,
    details_table_key: str,
) -> dict[str, int]:
    deleted = {"results": 0, "details": 0}
    params = {"scoring_month": pd.Timestamp(scoring_month).to_pydatetime()}
    with ora.connection.cursor() as cursor:
        for label, table_key in (("details", details_table_key), ("results", results_table_key)):
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


def insert_multivar_output_frame(
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


def prepare_multivar_oracle_results(
    results: pd.DataFrame,
    *,
    run_id: str,
    source_table_key: str,
    model_feature_count: int,
    peer_feature_count: int,
) -> pd.DataFrame:
    frame = normalize_columns(results)
    out = pd.DataFrame(index=frame.index)
    out["run_id"] = run_id
    out[TIME_COLUMN] = parse_dates(frame[TIME_COLUMN])
    out[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    out["musteri_segment"] = frame.get("musteri_segment")
    out["rating_group"] = to_numeric_or_none(frame.get("rating_group"))
    out["cst_sector"] = frame.get("cst_sector")
    out["cst_nace_code"] = frame.get("cst_nace_code")
    out["cst_nace_code_id"] = to_numeric_or_none(frame.get("cst_nace_code_id"))
    out["financial_term_l1y"] = parse_dates_or_none(frame.get("financial_term_l1y"))
    out["financial_term_q"] = parse_dates_or_none(frame.get("financial_term_q"))
    out["annualization_q"] = to_numeric_or_none(frame.get("annualization_q"))
    out["ref_donem_id"] = to_numeric_or_none(frame.get("ref_donem_id"))
    out["yukleme_zmn"] = parse_dates_or_none(frame.get("yukleme_zmn"))
    for column in (
        "anomaly_score",
        "if_score",
        "residual_score",
        "confidence",
        "coverage_ratio",
        "data_gap_score",
        "missing_feature_count",
        "rank_in_run",
    ):
        out[column] = to_numeric_or_none(frame.get(column))
    for column in ("alert_band", "alert_type", "review_queue"):
        out[column] = frame.get(column)
    for column in MULTIVAR_RESULT_REASON_COLUMNS:
        out[column] = frame[column] if column in frame.columns else None
    out["source_table_key"] = source_table_key
    out["model_feature_count"] = int(model_feature_count)
    out["peer_feature_count"] = int(peer_feature_count)

    for column, limit in (
        ("run_id", 64),
        (ID_COLUMN, 128),
        ("musteri_segment", 64),
        ("cst_sector", 500),
        ("cst_nace_code", 500),
        ("alert_band", 32),
        ("alert_type", 64),
        ("review_queue", 64),
        ("source_table_key", 64),
        ("reason_1", 1000),
        ("reason_2", 1000),
        ("reason_3", 1000),
    ):
        out[column] = out[column].map(lambda value, max_len=limit: text_or_none(value, max_len))
    return out[MULTIVAR_RESULT_COLUMNS]


def prepare_multivar_oracle_details(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    frame = normalize_columns(results)
    rows = []
    for _, row in frame.iterrows():
        details = parse_reason_details(row.get("reason_details"))
        scoring_date = parse_single_date(row.get(TIME_COLUMN))
        mono_id = text_or_none(row.get(ID_COLUMN), 128)
        for rank, detail in enumerate(details, start=1):
            components = detail.get("component_contributions") or {}
            rows.append(
                {
                    "run_id": text_or_none(run_id, 64),
                    TIME_COLUMN: scoring_date,
                    ID_COLUMN: mono_id,
                    "feature_rank": rank,
                    "feature_name": text_or_none(detail.get("feature"), 128),
                    "feature_label": text_or_none(detail.get("label"), 256),
                    "is_missing_reason": 1 if detail.get("is_missing_reason") else 0,
                    "actual_value": number_or_none(detail.get("actual")),
                    "customer_previous_reference": number_or_none(detail.get("customer_previous_reference")),
                    "peer_reference": number_or_none(detail.get("peer_reference")),
                    "peer_z": number_or_none(detail.get("peer_z")),
                    "peer_support": number_or_none(detail.get("peer_support")),
                    "train_reference": number_or_none(detail.get("train_reference")),
                    "reference_used": text_or_none(detail.get("reference_used"), 64),
                    "contribution_pct": number_or_none(detail.get("contribution_pct")),
                    "raw_contribution_pct": number_or_none(components.get("raw_pct")),
                    "peer_contribution_pct": number_or_none(components.get("peer_pct")),
                    "missing_contribution_pct": number_or_none(components.get("missing_pct")),
                    "direction_comment": text_or_none(detail.get("direction_comment"), 1000),
                    "previous_comment": text_or_none(detail.get("previous_comment"), 1000),
                    "financial_term_detail": text_or_none(detail.get("financial_term_detail"), 1000),
                    "reason_text": text_or_none(row.get(f"reason_{rank}"), 2000),
                }
            )
    if not rows:
        return pd.DataFrame(columns=MULTIVAR_DETAIL_COLUMNS)
    return pd.DataFrame(rows, columns=MULTIVAR_DETAIL_COLUMNS)


def parse_reason_details(value) -> list[dict]:
    if is_nullish(value):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    return []


def multivar_results_table_ddl(ora: OracleConnector, table_key: str) -> str:
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            RUN_ID VARCHAR2(64) NOT NULL,
            {TIME_COLUMN.upper()} DATE NOT NULL,
            {ID_COLUMN.upper()} VARCHAR2(128) NOT NULL,
            MUSTERI_SEGMENT VARCHAR2(64),
            RATING_GROUP NUMBER(10,4),
            CST_SECTOR VARCHAR2(500),
            CST_NACE_CODE VARCHAR2(500),
            CST_NACE_CODE_ID NUMBER(18,4),
            FINANCIAL_TERM_L1Y DATE,
            FINANCIAL_TERM_Q DATE,
            ANNUALIZATION_Q NUMBER(18,6),
            REF_DONEM_ID NUMBER(12),
            YUKLEME_ZMN TIMESTAMP,
            ANOMALY_SCORE NUMBER(6,2) NOT NULL,
            ALERT_BAND VARCHAR2(32) NOT NULL,
            ALERT_TYPE VARCHAR2(64),
            REVIEW_QUEUE VARCHAR2(64),
            IF_SCORE NUMBER(6,2),
            RESIDUAL_SCORE NUMBER(6,2),
            CONFIDENCE NUMBER(6,2),
            COVERAGE_RATIO NUMBER(10,6),
            DATA_GAP_SCORE NUMBER(6,2),
            MISSING_FEATURE_COUNT NUMBER(8),
            RANK_IN_RUN NUMBER(10),
            REASON_1 VARCHAR2(1000),
            REASON_2 VARCHAR2(1000),
            REASON_3 VARCHAR2(1000),
            SOURCE_TABLE_KEY VARCHAR2(64),
            MODEL_FEATURE_COUNT NUMBER(10),
            PEER_FEATURE_COUNT NUMBER(10),
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY ({TIME_COLUMN.upper()}, {ID_COLUMN.upper()})
        )
    """


def multivar_details_table_ddl(ora: OracleConnector, table_key: str) -> str:
    pk_name = f"PK_{ora._table_name(table_key)}"[:30]
    return f"""
        CREATE TABLE {ora._qualified_table_name(table_key)} (
            RUN_ID VARCHAR2(64) NOT NULL,
            {TIME_COLUMN.upper()} DATE NOT NULL,
            {ID_COLUMN.upper()} VARCHAR2(128) NOT NULL,
            FEATURE_RANK NUMBER(4) NOT NULL,
            FEATURE_NAME VARCHAR2(128) NOT NULL,
            FEATURE_LABEL VARCHAR2(256),
            IS_MISSING_REASON NUMBER(1),
            ACTUAL_VALUE NUMBER(24,8),
            CUSTOMER_PREVIOUS_REFERENCE NUMBER(24,8),
            PEER_REFERENCE NUMBER(24,8),
            PEER_Z NUMBER(18,6),
            PEER_SUPPORT NUMBER(10),
            TRAIN_REFERENCE NUMBER(24,8),
            REFERENCE_USED VARCHAR2(64),
            CONTRIBUTION_PCT NUMBER(10,4),
            RAW_CONTRIBUTION_PCT NUMBER(10,4),
            PEER_CONTRIBUTION_PCT NUMBER(10,4),
            MISSING_CONTRIBUTION_PCT NUMBER(10,4),
            DIRECTION_COMMENT VARCHAR2(1000),
            PREVIOUS_COMMENT VARCHAR2(1000),
            FINANCIAL_TERM_DETAIL VARCHAR2(1000),
            REASON_TEXT VARCHAR2(2000),
            CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT {pk_name} PRIMARY KEY ({TIME_COLUMN.upper()}, {ID_COLUMN.upper()}, FEATURE_RANK)
        )
    """


def to_numeric_or_none(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(values, errors="coerce")


def parse_dates_or_none(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")
    return parse_mixed_date_series(values)


def parse_single_date(value):
    if is_nullish(value):
        return None
    parsed = parse_mixed_date_series(pd.Series([value])).iloc[0]
    return None if pd.isna(parsed) else parsed


def number_or_none(value):
    if is_nullish(value):
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def text_or_none(value, max_len: int):
    if is_nullish(value):
        return None
    return str(value)[:max_len]


def is_nullish(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (np.ndarray, pd.Series, list)):
        return False
    return bool(result)


def parse_mixed_date_series(values) -> pd.Series:
    series = pd.Series(values, copy=False)
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], errors="coerce")
    other_mask = ~iso_mask
    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(text.loc[other_mask], errors="coerce", dayfirst=True)
    return parsed.dt.normalize()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def parse_dates(series: pd.Series) -> pd.Series:
    return parse_mixed_date_series(series)


def coerce_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    text = pd.Series(series, copy=False).astype("string").str.strip()
    text = text.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    comma_mask = text.str.contains(",", regex=False, na=False)
    normalized = text.copy()
    normalized.loc[comma_mask] = (
        normalized.loc[comma_mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(normalized, errors="coerce").astype(float)


def safe_divide(numerator, denominator) -> pd.Series:
    index = None
    for value in (numerator, denominator):
        if isinstance(value, pd.Series):
            index = value.index
            break
    if index is None:
        index = pd.RangeIndex(1)
    if numerator is None:
        num = pd.Series(np.nan, index=index, dtype=float)
    elif isinstance(numerator, pd.Series):
        num = pd.to_numeric(numerator, errors="coerce").astype(float)
    else:
        num = pd.Series(float(numerator), index=index, dtype=float)
    if denominator is None:
        den = pd.Series(np.nan, index=index, dtype=float)
    elif isinstance(denominator, pd.Series):
        den = pd.to_numeric(denominator, errors="coerce").astype(float)
    else:
        den = pd.Series(float(denominator), index=index, dtype=float)
    values = num / den.replace(0.0, np.nan)
    return values.replace([np.inf, -np.inf], np.nan).astype(float)


def signed_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    return np.sign(frame) * np.log1p(np.abs(frame))


def row_top_mean(values: np.ndarray, *, top_k: int = 3) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    k = min(top_k, values.shape[1])
    partitioned = np.partition(values, -k, axis=1)[:, -k:]
    return partitioned.mean(axis=1)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        return np.zeros(len(values), dtype=float)
    ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
    return ranks / ref.size * 100.0


def operational_band_thresholds(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "sari": OPERATIONAL_BAND_POLICY["yellow_floor"],
            "turuncu": OPERATIONAL_BAND_POLICY["orange_floor"],
            "kirmizi": OPERATIONAL_BAND_POLICY["red_floor"],
        }
    thresholds = {
        "sari": max(
            float(np.quantile(values, OPERATIONAL_BAND_POLICY["yellow_quantile"])),
            OPERATIONAL_BAND_POLICY["yellow_floor"],
        ),
        "turuncu": max(
            float(np.quantile(values, OPERATIONAL_BAND_POLICY["orange_quantile"])),
            OPERATIONAL_BAND_POLICY["orange_floor"],
        ),
        "kirmizi": max(
            float(np.quantile(values, OPERATIONAL_BAND_POLICY["red_quantile"])),
            OPERATIONAL_BAND_POLICY["red_floor"],
        ),
    }
    thresholds["turuncu"] = max(thresholds["turuncu"], thresholds["sari"])
    thresholds["kirmizi"] = max(thresholds["kirmizi"], thresholds["turuncu"])
    return thresholds


def assign_operational_bands(scores: np.ndarray, thresholds: dict[str, float]) -> list[str]:
    assigned = []
    for score in scores:
        if score >= thresholds["kirmizi"]:
            assigned.append("KIRMIZI")
        elif score >= thresholds["turuncu"]:
            assigned.append("TURUNCU")
        elif score >= thresholds["sari"]:
            assigned.append("SARI")
        else:
            assigned.append("NORMAL")
    return assigned


def aggregate_family_severity(severity: pd.Series, selected_features: list[str]) -> dict[str, float]:
    totals = {feature: 0.0 for feature in selected_features}
    selected = set(selected_features)
    for model_feature, value in severity.items():
        base_feature = base_feature_name(model_feature)
        if base_feature not in selected:
            continue
        totals[base_feature] += float(value)
    return {
        feature: value
        for feature, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    }


def component_contribution_pct(severity: pd.Series, base_feature: str, total: float) -> dict[str, float]:
    raw = float(severity.get(base_feature, 0.0))
    peer = float(severity.get(f"{base_feature}{PEER_FEATURE_SUFFIX}", 0.0))
    missing = float(severity.get(f"{base_feature}__missing_flag", 0.0))
    denominator = max(float(total), 1e-12)
    return {
        "raw_pct": round(raw / denominator * 100.0, 2),
        "peer_pct": round(peer / denominator * 100.0, 2),
        "missing_pct": round(missing / denominator * 100.0, 2),
    }


def base_feature_name(model_feature: str) -> str:
    name = str(model_feature)
    for suffix in (PEER_FEATURE_SUFFIX, "__missing_flag"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_reason_detail(
    *,
    row_id: str,
    base_feature: str,
    is_missing_reason: bool,
    actual,
    prior_reference,
    peer_reference,
    peer_median,
    peer_support,
    peer_z,
    train_reference,
    contribution_pct: float,
    component_contributions: dict[str, float],
    current_context: pd.Series,
    prior_context: pd.Series | None,
) -> dict:
    reference_label, reference_value = choose_reason_reference(
        feature=base_feature,
        actual=actual,
        prior_reference=prior_reference,
        peer_reference=peer_median,
        train_reference=train_reference,
        peer_z=peer_z,
        component_contributions=component_contributions,
    )
    direction_comment = direction_text(base_feature, actual, reference_label, reference_value)
    term_detail = financial_term_detail(base_feature, current_context, prior_context)
    previous_comment = previous_reference_comment(
        feature=base_feature,
        actual=actual,
        prior_reference=prior_reference,
        peer_reference=peer_median,
        term_detail=term_detail,
    )
    return {
        "mono_id": row_id,
        "feature": base_feature,
        "label": feature_label(base_feature),
        "is_missing_reason": bool(is_missing_reason),
        "actual": round_optional(actual, 6),
        "customer_previous_reference": round_optional(prior_reference, 6),
        "peer_reference": round_optional(peer_median if pd.notna(peer_median) else peer_reference, 6),
        "peer_z": round_optional(peer_z, 4),
        "peer_support": int(peer_support) if pd.notna(peer_support) else 0,
        "train_reference": round_optional(train_reference, 6),
        "reference_used": reference_label,
        "contribution_pct": round(float(contribution_pct), 2),
        "component_contributions": component_contributions,
        "direction_comment": direction_comment,
        "previous_comment": previous_comment,
        "financial_term_detail": term_detail,
    }


def format_reason(detail: dict) -> str:
    label = detail["label"]
    if detail["is_missing_reason"]:
        return (
            f"{label}: deger missing; katki %{detail['contribution_pct']:.1f}; "
            f"referans={detail.get('reference_used') or 'NA'}"
        )
    if detail.get("reference_used") == "peer medyan":
        parts = [
            f"{label}: peer'e gore sapma yuksek",
            f"gerceklesen={format_number(detail['actual'])}",
            f"peer={format_number(detail['peer_reference'])}",
            f"peer_z={format_number(detail.get('peer_z'))}",
            f"peer_support={detail.get('peer_support', 0)}",
            f"katki=%{detail['contribution_pct']:.1f}",
            detail["direction_comment"],
        ]
        if detail.get("previous_comment"):
            parts.append(detail["previous_comment"])
        if detail.get("financial_term_detail"):
            parts.append(detail["financial_term_detail"])
        return "; ".join(parts)
    return (
        f"{label}: gerceklesen={format_number(detail['actual'])}, "
        f"onceki={format_number(detail['customer_previous_reference'])}, "
        f"peer={format_number(detail['peer_reference'])}, "
        f"katki=%{detail['contribution_pct']:.1f}; {detail['direction_comment']}"
    )


def choose_reason_reference(
    *,
    feature: str,
    actual,
    prior_reference,
    peer_reference,
    train_reference,
    peer_z,
    component_contributions: dict[str, float],
) -> tuple[str | None, object]:
    if actual is None or pd.isna(actual):
        return first_valid_reference(("peer medyan", peer_reference), ("train medyan", train_reference))
    peer_component = float(component_contributions.get("peer_pct", 0.0))
    raw_component = float(component_contributions.get("raw_pct", 0.0))
    if peer_reference is not None and pd.notna(peer_reference):
        if peer_z is not None and pd.notna(peer_z) and abs(float(peer_z)) >= 2.0:
            return "peer medyan", peer_reference
        if peer_component >= raw_component * 0.75:
            return "peer medyan", peer_reference
    return first_valid_reference(
        ("musteri onceki ay", prior_reference),
        ("peer medyan", peer_reference),
        ("train medyan", train_reference),
    )


def financial_term_detail(
    feature: str,
    current_context: pd.Series,
    prior_context: pd.Series | None,
) -> str | None:
    term_column = financial_term_column_for(feature)
    if term_column is None or term_column not in current_context.index:
        return None
    current_term = clean_context_value(current_context.get(term_column))
    if not current_term:
        return None
    prior_term = None
    if prior_context is not None and not prior_context.empty and term_column in prior_context.index:
        prior_term = clean_context_value(prior_context.get(term_column))
    if prior_term and prior_term == current_term:
        return f"{term_column}={current_term}; onceki ayla ayni finansal term tasiniyor"
    if prior_term:
        return f"{term_column}={current_term}; onceki term={prior_term}"
    return f"{term_column}={current_term}"


def previous_reference_comment(
    *,
    feature: str,
    actual,
    prior_reference,
    peer_reference,
    term_detail: str | None,
) -> str | None:
    if actual is None or pd.isna(actual) or prior_reference is None or pd.isna(prior_reference):
        return None
    if term_detail and "ayni finansal term" in term_detail:
        return "onceki ay ayni finansal term oldugu icin trend kaniti zayif"
    peer_is_risk = peer_reference is not None and pd.notna(peer_reference) and is_riskier(feature, actual, peer_reference)
    prior_is_improving = not is_riskier(feature, actual, prior_reference)
    if prior_is_improving and peer_is_risk:
        return "onceki finansal veriye gore iyilesme var ama peer'e gore halen riskli"
    if peer_is_risk:
        return "peer'e gore risk sinyali onceki ay bilgisinden daha belirgin"
    return None


def financial_term_column_for(feature: str) -> str | None:
    lower = str(feature).lower()
    if lower.startswith("q_") or lower.endswith("_q") or "ara donem" in lower:
        return "financial_term_q"
    if lower.startswith("l1y_") or lower.endswith("_l1y") or "l1y" in lower:
        return "financial_term_l1y"
    return None


def clean_context_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value).strip() or None


def is_riskier(feature: str, actual, reference) -> bool:
    if actual is None or reference is None or pd.isna(actual) or pd.isna(reference):
        return False
    delta = float(actual) - float(reference)
    lower = feature.lower()
    if any(token in lower for token in INCREASE_IS_RISK_TOKENS):
        return delta > 0
    if any(token in lower for token in DECREASE_IS_RISK_TOKENS):
        return delta < 0
    return abs(delta) > 0


def classify_alert_type(
    row_details: list[dict],
    coverage_ratio: float,
    data_gap_score: float,
    alert_band: str,
) -> str:
    if alert_band == "NORMAL" and (data_gap_score >= 30.0 or coverage_ratio < 0.70):
        return "DATA_GAP"
    if not row_details:
        return "UNKNOWN"

    top_features = " ".join(str(detail.get("feature", "")) for detail in row_details[:3]).lower()
    if any(token in top_features for token in ("irb", "pd", "rating", "memzuc", "bank", "risk")):
        return "CREDIT_RISK"
    return "FINANCIAL_SIGNAL"


def review_queue_for(alert_band: str, alert_type: str, data_gap_score: float) -> str:
    if alert_band == "NORMAL":
        if alert_type == "DATA_GAP" or data_gap_score >= 30.0:
            return "DATA_QUALITY_REVIEW"
        return "NO_ACTION"
    if alert_band == "KIRMIZI":
        return "URGENT_FINANCIAL_REVIEW"
    if alert_band == "TURUNCU":
        return "FINANCIAL_REVIEW"
    return "WATCHLIST"


def direction_text(feature: str, actual, reference_label: str | None, reference_value) -> str:
    if actual is None or pd.isna(actual):
        return "deger missing oldugu icin yon yorumu yok"
    if reference_value is None or pd.isna(reference_value) or reference_label is None:
        return "referans olmadigi icin yon yorumu yok"
    delta = float(actual) - float(reference_value)
    if abs(delta) <= 1e-9:
        return f"{reference_label} ile ayni seviyede"
    movement = "artmis" if delta > 0 else "azalmis"
    risk_direction = "yon etkisi tanimsiz"
    lower = feature.lower()
    if any(token in lower for token in INCREASE_IS_RISK_TOKENS):
        risk_direction = "risk artisi" if delta > 0 else "risk azalisi"
    elif any(token in lower for token in DECREASE_IS_RISK_TOKENS):
        risk_direction = "risk artisi" if delta < 0 else "risk azalisi"
    return f"{reference_label} gore {movement}; {risk_direction}"


def lookup_prior_reference(prior_reference: pd.DataFrame, row_id: str, feature: str):
    if prior_reference.empty:
        return np.nan
    try:
        value = prior_reference.loc[row_id, feature]
    except KeyError:
        return np.nan
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return value


def lookup_prior_context(prior_context: pd.DataFrame, row_id: str) -> pd.Series | None:
    if prior_context.empty:
        return None
    try:
        value = prior_context.loc[row_id]
    except KeyError:
        return None
    if isinstance(value, pd.DataFrame):
        value = value.iloc[-1]
    return value


def first_valid_reference(*items: tuple[str, object]) -> tuple[str | None, object]:
    for label, value in items:
        if value is not None and not pd.isna(value):
            return label, value
    return None, np.nan


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def round_optional(value, digits: int):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def format_number(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


def update_latest_by_id(current: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if candidate.empty:
        return current
    if current.empty:
        combined = candidate
    else:
        combined = pd.concat([current, candidate], ignore_index=True)
    combined = combined.sort_values([ID_COLUMN, TIME_COLUMN])
    return combined.drop_duplicates(subset=[ID_COLUMN], keep="last").reset_index(drop=True)


def _keep_columns(columns: Iterable[str], numeric_source_columns: list[str]) -> list[str]:
    normalized = {str(column).strip().lower() for column in columns}
    keep = {ID_COLUMN, TIME_COLUMN}
    keep.update(column for column in CONTEXT_COLUMNS if column in normalized)
    keep.update(column for column in numeric_source_columns if column in normalized)
    keep.update(column for column in DERIVED_INPUT_COLUMNS if column in normalized)
    return sorted(keep)


def _resolve_scoring_month(scoring_month: str | None, month_profile: dict) -> pd.Timestamp:
    available = list(month_profile["month_counts"].keys())
    if scoring_month is None:
        return max(available)
    parsed = pd.to_datetime(scoring_month, dayfirst=True, errors="raise").normalize()
    if parsed not in month_profile["month_counts"]:
        available_text = ", ".join(month.strftime("%Y-%m-%d") for month in available[-5:])
        raise ValueError(f"Scoring month {parsed.date()} not found. Latest available months: {available_text}")
    return parsed


def _load_alert_bands(config: dict) -> dict[str, tuple[float, float]]:
    raw = config.get("alert_bands", {})
    result = {}
    for name, payload in raw.items():
        result[str(name).upper()] = (float(payload["min_score"]), float(payload["max_score"]))
    return result or {
        "NORMAL": (0, 60),
        "SARI": (60, 75),
        "TURUNCU": (75, 90),
        "KIRMIZI": (90, 100),
    }
