"""OpenAI-compatible LLM anomaly decision runner."""

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from engine.multivar_anomaly import ID_COLUMN, TIME_COLUMN, run_multivar_anomaly
from engine.config_loader import load_secrets
from llm.evidence_builder import (
    EvidenceConfig,
    build_evidence_from_result_rows,
    build_evidence_packages_from_oracle,
    build_evidence_packages,
    load_input_frame,
    write_jsonl,
)
from llm.oracle_output import (
    audit_llm_output_tables,
    ensure_llm_output_tables_in_oracle,
    write_llm_outputs_to_oracle,
)

try:
    from pydantic import BaseModel, Field
except ImportError:  # Runtime dependency is checked when the LLM chain is built.
    BaseModel = None
    Field = None


logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)
LLM_PAYLOAD_PREVIEW_CUSTOMERS = 3
LLM_PAYLOAD_PREVIEW_CHARS = 50000
LLM_ERROR_PAYLOAD_PREVIEW_CHARS = 8000
RAW_MODEL_RESPONSE_FILE = Path(os.environ.get("LLM_RAW_RESPONSE_FILE", "runtime/llm/raw_model_responses.jsonl"))
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
CA_BUNDLE_ENV_VARS = ("LLM_CA_BUNDLE", "LLM_SSL_CERT_FILE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
COMMON_CA_BUNDLE_PATHS = (
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/ssl/cert.pem",
    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
)
ML_SELECTION_SCORE_PRIORITY = (
    "ensemble_score",
    "anomaly_score",
    "autoencoder_score",
    "residual_score",
    "if_score",
)
LLM_FORBIDDEN_MODEL_OUTPUT_FIELDS = {
    "anomaly_score",
    "ensemble_score",
    "if_score",
    "residual_score",
    "autoencoder_score",
    "confidence",
    "alert_band",
    "alert_type",
    "review_queue",
    "rank_in_run",
    "selection_bucket",
    "selection_score",
    "selection_score_column",
    "selection_model",
    "is_anomaly",
}

SYSTEM_PROMPT = """Sen deneyimli bir banka risk yoneticisi ve finansal anomali uzmanisin.
Sana secilen scoring ayina ait musteri snapshot kayitlari verilecek.
Her kayit bir musteri icin tek scoring snapshot'idir; output'ta her input kaydi icin tek karar donmelisin.

Her snapshot icinde musterinin kendi tarihsel seyri, ayni snapshotlara ait peer serisi, trend, sezon ve veri kalitesi sinyalleri vardir.
Bu seriler ayrica karar satiri degildir; sadece secilen snapshot'in anomali olup olmadigini yorumlamak icin arka plan bilgisidir.

Kurallar:
- Degisken sozlugunu oku: is anlami, formul, risk yonu ve birimi dikkate al.
- Tum aciklama alanlarini Turkce yaz. reason_summary ve reason_1/2/3 icinde Ingilizce cumle kullanma.
- reason_summary ve reason_1/2/3 icinde karar verdigin degiskenler icin sayisal kanit yaz: current, previous veya history median, change_pct, history_z, peer_median ve peer_z alanlarindan mevcut olanlari kullan.
- Artis, azalis, sapma, trend kirilmasi veya peer uyumsuzlugu gibi ifadeleri sayisal karsilastirma vermeden kullanma.
- Cari degeri musterinin kendi gecmisiyle, ayni sezon gecmisiyle ve peer grubuyla karsilastir.
- snapshot_series.customer ile musterinin son snapshot degerlerini, snapshot_series.peer ile ayni snapshotlardaki peer median/support/quality bilgisini birlikte oku.
- Tek donem sicrama, kademeli trend bozulmasi, sezon etkisi ve veri kalitesi problemini ayir.
- Buyuk tutar tek basina anomali degildir; olcek, peer ve tarihsel davranisla birlikte yorumla.
- Missing veya stale finansal term sinyalini finansal bozulma gibi yazma; veri kalitesi veya inceleme nedeni olarak ayir.
- Peer kalitesi ZAYIF ise kesin hukum verme, manuel inceleme oner.
- Musterinin kendi tarihsel verisi yeterliyse peer tek basina anomali nedeni olamaz; peer sadece destekleyici kanittir.
- Peer kaynakli anomali ancak musteri history'si yetersizse veya musteri history'sindeki bozulmayi destekliyorsa kullanilabilir.
- risk_direction=HIGHER_IS_RISKY ise artis risk bozulmasi, azalis risk azalisi/iyilesmedir.
- risk_direction=LOWER_IS_RISKY ise azalis risk bozulmasi, artis risk azalisi/iyilesmedir.
- Risk yonunun tersine giden sapmalari riskli anomali nedeni yapma; gerekiyorsa olumlu/iyilesen sapma olarak not et ama riskli anomali flag'i verme.
- Rating grubunu risk sinyali olarak kullanabilirsin.
- IRB/model PD degerleri ve PD oranlari karar kaniti olarak kullanilmaz.
- Gelecek donem varsayimi yapma.

Her LLM isteginde tek musteri ve tek scoring snapshot vardir.
Feature'lar veya nedenler icin ayri results item'i uretme.
Tum feature sinyallerini birlestirip musteri-snapshot icin tek karar dondur.

Sadece su alanlari dondur: period_position, is_anomaly, anomaly_type, anomaly_score, reason_summary, reason_1,
reason_1_weight, reason_2, reason_2_weight, reason_3, reason_3_weight, risk_level.
anomaly_score 0.0 ile 1.0 arasinda anomali siddet skorudur; confidence degildir.
reason_summary tekil karar nedenini birlestirilmis olarak anlatir.
reason_1/2/3 en yuksek etkili uc nedeni, weight alanlari ise bu nedenlerin toplam karar icindeki goreli agirligini verir.
Agirliklar 0.0 ile 1.0 arasinda olmali ve mumkunse toplami 1.0'a yakin olmali.
reason_summary en fazla 800 karakter, reason_1/2/3 her biri en fazla 420 karakter olsun.
String degerlerinde satir sonu kullanma.
Sadece tek satir gecerli JSON object dondur. Markdown, kod blogu, aciklama metni, Python repr veya JSON string wrapper kullanma.
JSON'u tirnak icine alinmis string olarak dondurme; dogrudan { ile baslayan ve } ile biten object yaz."""

OUTPUT_CONTRACT = {
    "period_position": "0-based integer",
    "is_anomaly": "boolean",
    "anomaly_type": "ANI_RISK_ARTISI | FINANSAL_BOZULMA | PD_RISKI | KKB_RISKI | PEER_UYUMSUZLUGU | DATA_GAP | TREND_KIRILMASI | SEZON_DISI_SAPMA | NORMAL",
    "anomaly_score": "0.0-1.0 arasi anomali siddet skoru; guven skoru degildir",
    "reason_summary": "history, peer, trend, sezon ve veri kalitesini sayisal kanitlarla birlestiren Turkce tekil aciklama",
    "reason_1": "karara en cok etki eden Turkce neden; ilgili degiskenin current/previous/change/history_z/peer_z gibi sayisal kanitlarini icermeli",
    "reason_1_weight": "0.0-1.0",
    "reason_2": "karara ikinci en cok etki eden Turkce neden veya bos; doluysa sayisal kanit icermeli",
    "reason_2_weight": "0.0-1.0",
    "reason_3": "karara ucuncu en cok etki eden Turkce neden veya bos; doluysa sayisal kanit icermeli",
    "reason_3_weight": "0.0-1.0",
    "risk_level": "DUSUK | ORTA | YUKSEK | KRITIK",
}


if BaseModel is not None and Field is not None:
    class AnomalyRecord(BaseModel):
        period_position: int = Field(
            description="Tek scoring snapshot icin sira numarasi; her request tek snapshot oldugu icin 0 olmali"
        )
        is_anomaly: bool = Field(description="Bu musteri-donem kaydi anomali mi?")
        anomaly_type: str = Field(
            description=(
                "Anomali tipi: ANI_RISK_ARTISI, FINANSAL_BOZULMA, PD_RISKI, KKB_RISKI, "
                "PEER_UYUMSUZLUGU, DATA_GAP, TREND_KIRILMASI, SEZON_DISI_SAPMA veya NORMAL"
            )
        )
        anomaly_score: float = Field(description="Anomali siddet skoru, 0.0 ile 1.0 arasinda; confidence degildir")
        reason_summary: str = Field(
            description="Tekil karar nedeninin Turkce ozeti; current, previous/history median, change_pct, history_z ve peer_z gibi sayisal kanitlarla kiyaslamali olsun"
        )
        reason_1: str = Field(description="Karara en cok etki eden neden; sayisal kanit icermeli")
        reason_1_weight: float = Field(description="Birinci nedenin goreli agirligi, 0.0 ile 1.0 arasinda")
        reason_2: Optional[str] = Field(default=None, description="Karara ikinci en cok etki eden neden; doluysa sayisal kanit icermeli")
        reason_2_weight: float = Field(default=0.0, description="Ikinci nedenin goreli agirligi, 0.0 ile 1.0 arasinda")
        reason_3: Optional[str] = Field(default=None, description="Karara ucuncu en cok etki eden neden; doluysa sayisal kanit icermeli")
        reason_3_weight: float = Field(default=0.0, description="Ucuncu nedenin goreli agirligi, 0.0 ile 1.0 arasinda")
        risk_level: str = Field(description="Risk seviyesi: DUSUK, ORTA, YUKSEK veya KRITIK")

    AnomalyBatchResult = AnomalyRecord
else:
    AnomalyRecord = None
    AnomalyBatchResult = None


def configure_logging() -> None:
    """Enable visible console logs for CLI runs."""

    level_name = (os.environ.get("LLM_LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = Path(os.environ.get("LLM_LOG_FILE", "runtime/logs/cli/llm_anomaly.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("LLM console/file logging enabled: log_file=%s", log_file)


def load_local_env_files() -> None:
    """Load local env files without overriding already-set environment values."""

    candidates = [
        Path(".env"),
        Path(__file__).resolve().parent / ".env.local",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def build_langchain_structured_chain() -> Any:
    settings = load_llm_settings()
    validate_llm_settings(settings)
    schema = anomaly_batch_schema()
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain structured LLM dependencies are required. "
            "Install requirements.txt or pip install langchain-openai langchain-core pydantic httpx."
        ) from exc

    http_client = build_llm_http_client(settings)
    llm = ChatOpenAI(
        base_url=str(settings["base_url"]).rstrip("/"),
        api_key=str(settings["api_key"]),
        model=str(settings["model"]),
        temperature=0,
        max_retries=int(settings["max_retries"]),
        http_client=http_client,
        **optional_llm_kwargs(settings),
    )
    structured_llm = llm.with_structured_output(schema, include_raw=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", escape_prompt_literal_braces(SYSTEM_PROMPT)),
            (
                "human",
                escape_prompt_literal_braces(
                    "Asagidaki musteri-donem evidence kayitlarini analiz et:\n\n{input_records}",
                    allowed_variables={"input_records"},
                ),
            ),
        ]
    )
    logger.info(
        "LangChain structured LLM chain initialized: model=%s structured_call=with_structured_output_schema_only include_raw=True max_retries=%s max_tokens=%s http_trust_env=%s ssl_verify=%s ca_bundle=%s raw_response_file=%s",
        settings["model"],
        settings["max_retries"],
        settings.get("max_tokens"),
        settings["http_trust_env"],
        settings.get("ssl_verify", True),
        settings.get("ca_bundle"),
        RAW_MODEL_RESPONSE_FILE,
    )
    return prompt | structured_llm


def escape_prompt_literal_braces(text: str, *, allowed_variables: set[str] | None = None) -> str:
    """Escape literal braces so LangChain templates only keep allowed variables."""

    escaped = text.replace("{", "{{").replace("}", "}}")
    for variable in allowed_variables or set():
        escaped = escaped.replace("{{" + variable + "}}", "{" + variable + "}")
    return escaped


def build_llm_http_client(settings: dict[str, Any]) -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for the LLM client. Install requirements.txt or pip install httpx.") from exc

    verify: bool | str
    if not bool(settings.get("ssl_verify", True)):
        verify = False
        logger.info("LLM SSL verification disabled: ssl_verify=False")
    else:
        verify = str(settings["ca_bundle"]) if settings.get("ca_bundle") else True
    return httpx.Client(
        trust_env=bool(settings["http_trust_env"]),
        timeout=None,
        verify=verify,
    )


def anomaly_batch_schema() -> type:
    if AnomalyBatchResult is None:
        raise RuntimeError(
            "Pydantic is required for the LLM structured output flow. "
            "Install requirements.txt or pip install pydantic."
        )
    if not isinstance(AnomalyBatchResult, type):
        raise RuntimeError(f"Invalid LLM structured schema: expected class, got {type(AnomalyBatchResult).__name__}.")
    if BaseModel is not None:
        try:
            is_pydantic_model = issubclass(AnomalyBatchResult, BaseModel)
        except TypeError as exc:
            raise RuntimeError(
                f"Invalid LLM structured schema: AnomalyBatchResult is not a Pydantic class ({exc})."
            ) from exc
        if not is_pydantic_model:
            raise RuntimeError("Invalid LLM structured schema: AnomalyBatchResult does not inherit Pydantic BaseModel.")
    return AnomalyBatchResult


def optional_llm_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if settings.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(settings["max_tokens"])
    return kwargs


def validate_llm_settings(settings: dict[str, Any]) -> None:
    missing = [
        key
        for key in ("base_url", "api_key", "model")
        if not settings.get(key) or (isinstance(settings.get(key), str) and not str(settings.get(key)).strip())
    ]
    if missing:
        raise RuntimeError(
            "LLM settings are incomplete. Missing: "
            + ", ".join(missing)
            + ". Set them in secret/secrets.yaml llm.sections.<section> or env variables."
        )


def format_evidence_for_langchain(evidence_items: list[dict[str, Any]]) -> str:
    if len(evidence_items) != 1:
        raise RuntimeError(
            "LLM prompt contract expects exactly one scoring snapshot per request; "
            f"got {len(evidence_items)}."
        )
    lines = [
        "Gorev: Bu tek musteri ve tek scoring snapshot icin anomali karari ver.",
        "Cikti: Tek satir JSON object olmali; results listesi, feature listesi veya birden fazla karar dondurme.",
        "JSON disinda metin yazma. Markdown/code fence/Python repr kullanma. JSON'u string icine gomerek dondurme.",
        "Alanlar: period_position, is_anomaly, anomaly_type, anomaly_score, reason_summary, reason_1, reason_1_weight, reason_2, reason_2_weight, reason_3, reason_3_weight, risk_level.",
        "Tum aciklama icerigi Turkce olmali; reason_summary ve reason_1/2/3 Ingilizce cumle icermemeli.",
        "anomaly_score 0.0-1.0 arasi anomali siddet skorudur; confidence degildir.",
        "reason_summary tekil karar nedenini birlestirilmis olarak anlatir.",
        "reason_1/2/3 en yuksek etkili uc nedeni temsil eder; weight alanlari goreli karar agirligidir.",
        "Artis/azalis/sapma/trend/peer uyumsuzlugu dedigin her reason icinde sayisal kanit ver: current, previous/history_median, change_pct, history_z, peer_median, peer_z.",
        "Sayi yoksa o degiskeni reason olarak yazma. Reasonlarda 'belirgin artis' gibi soyut ifade tek basina yeterli degildir.",
        "reason_summary en fazla 800 karakter, reason_1/2/3 her biri en fazla 420 karakter olsun. String degerlerinde satir sonu kullanma.",
        "Musteri history'si yeterliyse peer tek basina anomali nedeni olamaz; peer sadece destekleyici kanittir.",
        'Ornek output sekli: {"period_position":0,"is_anomaly":false,"anomaly_type":"NORMAL","anomaly_score":0.0,"reason_summary":"...","reason_1":"...","reason_1_weight":1.0,"reason_2":"","reason_2_weight":0.0,"reason_3":"","reason_3_weight":0.0,"risk_level":"DUSUK"}',
        "Not: History, customer_series ve peer_series sadece bu snapshot'i yorumlama baglamidir.",
        "Kayit:",
    ]
    for index, item in enumerate(evidence_items):
        lines.extend(compact_evidence_lines(sanitize_evidence_for_llm(item), index=index))
    return "\n".join(lines)


def sanitize_evidence_for_llm(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if is_forbidden_llm_model_output_field(key):
                continue
            cleaned[key] = sanitize_evidence_for_llm(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_evidence_for_llm(item) for item in value]
    return value


def is_forbidden_llm_model_output_field(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized.startswith("ml_") or normalized in LLM_FORBIDDEN_MODEL_OUTPUT_FIELDS


def compact_evidence_lines(item: dict[str, Any], *, index: int) -> list[str]:
    context = item.get("context") or {}
    data_quality = item.get("data_quality") or {}
    peer_definition = item.get("peer_definition") or {}
    lines = [
        f"--- DONEM {index} ---",
        f"period_position={index}",
        f"mono_id={item.get('mono_id')} | cohort_dt={item.get('cohort_dt')}",
        "context=" + compact_json(context),
        "data_quality=" + compact_json(data_quality),
        "peer_definition=" + compact_json(peer_definition),
        "features:",
    ]
    for feature_index, feature in enumerate(item.get("features") or [], start=1):
        lines.append(compact_feature_line(feature, feature_index))
    return lines


def compact_feature_line(feature: dict[str, Any], index: int) -> str:
    dictionary = feature.get("dictionary") or {}
    history = feature.get("history") or {}
    trend = feature.get("trend") or {}
    seasonality = feature.get("seasonality") or {}
    snapshot_series = feature.get("snapshot_series") or {}
    peer = feature.get("peer") or {}
    data_quality = feature.get("data_quality") or {}
    parts = [
        f"{index}) name={feature.get('name')}",
        f"label={dictionary.get('label')}",
        f"category={dictionary.get('category')}",
        f"formula={dictionary.get('formula')}",
        f"risk_direction={dictionary.get('risk_direction')}",
        f"opposite_direction_meaning={dictionary.get('opposite_direction_meaning')}",
        f"current={feature.get('current_value')}",
        f"previous={feature.get('previous_value')}",
        f"change_pct={feature.get('change_pct')}",
        "history="
        + compact_json(
            {
                "period_count": history.get("period_count"),
                "median": history.get("median"),
                "p25": history.get("p25"),
                "p75": history.get("p75"),
                "robust_scale": history.get("robust_scale"),
                "history_z": feature_history_z(feature),
                "rolling_3m_median": history.get("rolling_3m_median"),
                "rolling_6m_median": history.get("rolling_6m_median"),
                "rolling_12m_median": history.get("rolling_12m_median"),
            }
        ),
        "trend="
        + compact_json(
            {
                "slope_6m": trend.get("slope_6m"),
                "slope_12m": trend.get("slope_12m"),
                "trend_break_flag": trend.get("trend_break_flag"),
                "trend_note": trend.get("trend_note"),
            }
        ),
        "seasonality="
        + compact_json(
            {
                "month_of_year": seasonality.get("month_of_year"),
                "same_month_last_year_value": seasonality.get("same_month_last_year_value"),
                "yoy_change_pct": seasonality.get("yoy_change_pct"),
                "same_month_customer_median": seasonality.get("same_month_customer_median"),
                "same_month_customer_z": seasonality.get("same_month_customer_z"),
                "seasonal_peer_median": seasonality.get("seasonal_peer_median"),
                "seasonal_peer_z": seasonality.get("seasonal_peer_z"),
                "seasonality_note": seasonality.get("seasonality_note"),
            }
        ),
        "snapshot_series="
        + compact_json(
            {
                "window_periods": snapshot_series.get("window_periods"),
                "customer": snapshot_series.get("customer"),
                "peer": snapshot_series.get("peer"),
            }
        ),
        "peer="
        + compact_json(
            {
                "level": peer.get("peer_definition_level"),
                "median": peer.get("peer_median"),
                "z": peer.get("peer_z"),
                "support": peer.get("peer_support"),
                "quality": peer.get("peer_quality"),
            }
        ),
        "feature_data_quality=" + compact_json(data_quality),
    ]
    return " | ".join(parts)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def invoke_langchain_structured_decision(chain: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    return invoke_langchain_structured_decisions(chain, [evidence])[0]


def invoke_langchain_structured_decisions(
    chain: Any,
    evidence_items: list[dict[str, Any]],
    *,
    payload_preview_index: int | None = None,
) -> list[dict[str, Any]]:
    if not evidence_items:
        return []
    if len(evidence_items) != 1:
        raise RuntimeError(
            "LLM decision contract is one customer snapshot per request; "
            f"got {len(evidence_items)} evidence items."
        )
    input_records = format_evidence_for_langchain(evidence_items)
    first_item = evidence_items[0] if evidence_items else {}
    logger.info(
        "LLM request payload prepared: mono_id=%s decision_items=%s first_cohort_dt=%s chars=%s total_features=%s formatter=compact_text",
        first_item.get("mono_id"),
        len(evidence_items),
        first_item.get("cohort_dt"),
        len(input_records),
        sum(len(item.get("features") or []) for item in evidence_items),
    )
    if payload_preview_index is not None and payload_preview_index <= LLM_PAYLOAD_PREVIEW_CUSTOMERS:
        logger.info(
            "========== LLM PAYLOAD PREVIEW %s/%s START | mono_id=%s chars=%s ==========\n%s",
            payload_preview_index,
            LLM_PAYLOAD_PREVIEW_CUSTOMERS,
            first_item.get("mono_id"),
            len(input_records),
            truncate_text(input_records, LLM_PAYLOAD_PREVIEW_CHARS),
        )
        logger.info(
            "========== LLM PAYLOAD PREVIEW %s/%s END | mono_id=%s ==========",
            payload_preview_index,
            LLM_PAYLOAD_PREVIEW_CUSTOMERS,
            first_item.get("mono_id"),
        )
    try:
        response = chain.invoke({"input_records": input_records})
    except Exception as exc:
        logger.exception(
            "LLM request failed before structured response: mono_id=%s decision_items=%s first_cohort_dt=%s chars=%s exception_type=%s exception=%r payload_preview=%s",
            first_item.get("mono_id"),
            len(evidence_items),
            first_item.get("cohort_dt"),
            len(input_records),
            type(exc).__name__,
            exc,
            truncate_text(input_records, LLM_ERROR_PAYLOAD_PREVIEW_CHARS),
        )
        raise
    response = unwrap_structured_response(
        response,
        first_item=first_item,
        decision_items=len(evidence_items),
        input_chars=len(input_records),
    )
    if response is None:
        raise RuntimeError(
            "LLM structured response returned None after HTTP OK. "
            "Expected one structured decision object with fields: period_position, is_anomaly, anomaly_type, anomaly_score, reason_summary, reason_1..3, reason_1_weight..3_weight, risk_level. "
            f"Raw model response was written to {RAW_MODEL_RESPONSE_FILE} if the endpoint returned one."
        )
    if (isinstance(response, dict) and "results" in response) or hasattr(response, "results"):
        records = response.get("results") if isinstance(response, dict) else getattr(response, "results", None)
        actual_count = len(records or [])
        raise RuntimeError(
            "LLM returned a results list, but the contract is a single customer-snapshot decision object: "
            f"actual_results={actual_count} mono_id={first_item.get('mono_id')}. "
            "This usually means the model treated feature reasons as separate decisions; check the raw response file."
        )
    decision = normalize_structured_decision(model_to_dict(response))
    position = int(decision["period_position"])
    if position != 0:
        raise RuntimeError(
            "LLM structured response period_position must be 0 for single-snapshot request: "
            f"period_position={position} mono_id={first_item.get('mono_id')}"
        )
    evidence = evidence_items[0]
    decision["period_position"] = 0
    if not decision.get("mono_id"):
        decision["mono_id"] = evidence.get("mono_id")
    if not decision.get("cohort_dt"):
        decision["cohort_dt"] = evidence.get("cohort_dt")
    return [decision]


def unwrap_structured_response(
    response: Any,
    *,
    first_item: dict[str, Any],
    decision_items: int,
    input_chars: int,
) -> Any:
    if not isinstance(response, dict) or ("parsed" not in response and "raw" not in response):
        return response
    raw_response = response.get("raw")
    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")
    raw_path = write_raw_model_response(
        raw_response,
        first_item=first_item,
        decision_items=decision_items,
        input_chars=input_chars,
        parsing_error=parsing_error,
    )
    parsed_from_raw_content = None if parsed is not None else parse_raw_structured_decision(raw_response)
    logger.info(
        "LLM raw model response captured: path=%s mono_id=%s parsed=%s parsed_from_raw_content=%s parsing_error=%s",
        raw_path,
        first_item.get("mono_id"),
        parsed is not None,
        parsed_from_raw_content is not None,
        type(parsing_error).__name__ if parsing_error else None,
    )
    if parsing_error:
        logger.warning(
            "LLM structured parsing error: mono_id=%s error=%r raw_response_file=%s",
            first_item.get("mono_id"),
            parsing_error,
            raw_path,
        )
    return parsed if parsed is not None else parsed_from_raw_content


def parse_raw_structured_decision(raw_response: Any) -> dict[str, Any] | None:
    """Parse plain JSON content when LangChain include_raw returns parsed=None."""

    for candidate in raw_response_json_candidates(raw_response):
        parsed = parse_json_like(candidate)
        if parsed is None:
            continue
        parsed = model_to_dict(parsed)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"results": parsed}
    return None


def raw_response_json_candidates(raw_response: Any) -> list[Any]:
    candidates: list[Any] = []
    if raw_response is None:
        return candidates
    content = raw_response.get("content") if isinstance(raw_response, dict) else getattr(raw_response, "content", None)
    candidates.extend(content_to_text_candidates(content))

    additional_kwargs = (
        raw_response.get("additional_kwargs")
        if isinstance(raw_response, dict)
        else getattr(raw_response, "additional_kwargs", None)
    )
    tool_calls = raw_response.get("tool_calls") if isinstance(raw_response, dict) else getattr(raw_response, "tool_calls", None)
    candidates.extend(tool_call_argument_candidates(tool_calls))
    if isinstance(additional_kwargs, dict):
        candidates.extend(tool_call_argument_candidates(additional_kwargs.get("tool_calls")))
    return [candidate for candidate in candidates if candidate not in (None, "")]


def content_to_text_candidates(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, dict):
        return [content.get("text") or content.get("content") or content]
    if isinstance(content, list):
        candidates: list[Any] = []
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                candidates.append(part)
                text_parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content") or part.get("input")
                if value is not None:
                    candidates.append(value)
                    if isinstance(value, str):
                        text_parts.append(value)
        if text_parts:
            candidates.append("\n".join(text_parts))
        return candidates
    return [str(content)]


def tool_call_argument_candidates(tool_calls: Any) -> list[Any]:
    if not tool_calls:
        return []
    candidates: list[Any] = []
    for call in tool_calls:
        if isinstance(call, dict):
            args = call.get("args")
            function = call.get("function")
            if isinstance(function, dict):
                args = args or function.get("arguments")
            candidates.append(args)
            continue
        args = getattr(call, "args", None)
        function = getattr(call, "function", None)
        if function is not None:
            args = args or getattr(function, "arguments", None)
        candidates.append(args)
    return candidates


def parse_json_like(value: Any) -> Any | None:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = strip_markdown_json_fence(text)
    for _ in range(3):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = parse_first_json_object(text)
        if isinstance(parsed, str):
            text = strip_markdown_json_fence(parsed.strip())
            continue
        return parsed
    return None


def strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_first_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    return None


def write_raw_model_response(
    raw_response: Any,
    *,
    first_item: dict[str, Any],
    decision_items: int,
    input_chars: int,
    parsing_error: Any,
) -> Path:
    RAW_MODEL_RESPONSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mono_id": first_item.get("mono_id"),
        "first_cohort_dt": first_item.get("cohort_dt"),
        "decision_items": decision_items,
        "input_chars": input_chars,
        "parsing_error": repr(parsing_error) if parsing_error else None,
        "raw_response": extract_raw_model_response(raw_response),
    }
    with RAW_MODEL_RESPONSE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return RAW_MODEL_RESPONSE_FILE


def extract_raw_model_response(raw_response: Any) -> dict[str, Any]:
    if raw_response is None:
        return {"type": None, "content": None}
    content = getattr(raw_response, "content", None)
    additional_kwargs = getattr(raw_response, "additional_kwargs", None)
    response_metadata = getattr(raw_response, "response_metadata", None)
    tool_calls = getattr(raw_response, "tool_calls", None)
    return {
        "type": type(raw_response).__name__,
        "content": content,
        "additional_kwargs": additional_kwargs,
        "response_metadata": response_metadata,
        "tool_calls": tool_calls,
        "repr": repr(raw_response),
    }


def normalize_structured_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Normalize one customer-snapshot decision for Oracle output."""

    required_fields = [
        "period_position",
        "is_anomaly",
        "anomaly_type",
        "anomaly_score",
        "reason_summary",
        "reason_1",
        "reason_1_weight",
        "risk_level",
    ]
    missing_fields = [field for field in required_fields if field not in decision or decision.get(field) is None]
    if missing_fields:
        raise RuntimeError(
            "LLM structured response is missing required single-decision fields: "
            f"{','.join(missing_fields)}"
        )

    reason_summary = str(decision.get("reason_summary") or "").strip()
    decision["anomaly_type"] = str(decision.get("anomaly_type") or "NORMAL").strip() or "NORMAL"
    decision["risk_level"] = str(decision.get("risk_level") or "DUSUK").strip() or "DUSUK"
    decision["anomaly_score"] = bounded_unit_float(decision.get("anomaly_score"))
    decision["reason_summary"] = reason_summary
    for index in range(1, 4):
        reason_key = f"reason_{index}"
        weight_key = f"reason_{index}_weight"
        reason_value = decision.get(reason_key)
        decision[reason_key] = str(reason_value).strip() if reason_value is not None else None
        decision[weight_key] = bounded_unit_float(decision.get(weight_key))
    decision.setdefault("recommended_action", "Manuel incele" if bool(decision.get("is_anomaly")) else "Izle")
    decision["main_reasons"] = top_reason_records(decision)
    return decision


def top_reason_records(decision: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    numeric_evidence = decision.get("_reason_numeric_evidence") or {}
    for index in range(1, 4):
        reason = decision.get(f"reason_{index}")
        if not reason:
            continue
        weight = bounded_unit_float(decision.get(f"reason_{index}_weight"))
        evidence_text = numeric_evidence.get(index) or numeric_evidence.get(str(index))
        if evidence_text:
            evidence_text = f"weight={weight:.4f}; {evidence_text}"
        else:
            evidence_text = f"weight={weight:.4f}"
        records.append(
            {
                "feature": f"REASON_{index}",
                "evidence": evidence_text,
                "interpretation": reason,
            }
        )
    if not records:
        records.append(
            {
                "feature": "GENEL_DEGERLENDIRME",
                "evidence": "",
                "interpretation": decision.get("reason_summary") or "LLM structured karar aciklamasi bos dondu.",
            }
        )
    return records


def bounded_unit_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def enrich_decision_reasons_with_numeric_evidence(
    decision: dict[str, Any],
    evidence_item: dict[str, Any],
) -> dict[str, Any]:
    features = [
        feature
        for feature in (evidence_item.get("feature_details") or evidence_item.get("features") or [])
        if isinstance(feature, dict)
    ]
    if not features:
        decision["main_reasons"] = top_reason_records(decision)
        return decision

    numeric_evidence: dict[int, str] = {}
    used_feature_indexes: set[int] = set()
    for reason_index in range(1, 4):
        reason_key = f"reason_{reason_index}"
        reason = text_or_empty(decision.get(reason_key))
        if not reason:
            continue
        feature_index, feature = select_reason_feature(reason, features, used_feature_indexes, reason_index - 1)
        if feature is None:
            continue
        used_feature_indexes.add(feature_index)
        evidence_text = numeric_evidence_for_feature(feature)
        if not evidence_text:
            continue
        numeric_evidence[reason_index] = evidence_text
        if not has_structured_numeric_evidence(reason):
            decision[reason_key] = append_reason_numeric_evidence(reason, evidence_text)

    summary = text_or_empty(decision.get("reason_summary"))
    if summary and not has_structured_numeric_evidence(summary):
        summary_feature = features[0]
        summary_evidence = numeric_evidence.get(1) or numeric_evidence_for_feature(summary_feature)
        if summary_evidence:
            decision["reason_summary"] = append_reason_numeric_evidence(summary, summary_evidence, prefix="Sayisal ozet")

    decision["_reason_numeric_evidence"] = numeric_evidence
    decision["main_reasons"] = top_reason_records(decision)
    return decision


def select_reason_feature(
    reason: str,
    features: list[dict[str, Any]],
    used_indexes: set[int],
    fallback_index: int,
) -> tuple[int, dict[str, Any] | None]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    reason_text = normalize_match_text(reason)
    reason_tokens = match_tokens(reason)
    for index, feature in enumerate(features):
        dictionary = feature.get("dictionary") or {}
        candidates = [
            feature.get("name"),
            dictionary.get("label"),
            dictionary.get("category"),
            dictionary.get("formula"),
        ]
        score = 0
        for candidate in candidates:
            candidate_text = normalize_match_text(candidate)
            if candidate_text and candidate_text in reason_text:
                score += 8
            candidate_tokens = match_tokens(candidate)
            if candidate_tokens:
                score += len(reason_tokens.intersection(candidate_tokens))
        if index in used_indexes:
            score -= 2
        scored.append((score, -index, feature))
    scored.sort(reverse=True, key=lambda row: (row[0], row[1]))
    best_score, best_negative_index, best_feature = scored[0]
    if best_score > 0:
        return -best_negative_index, best_feature
    if 0 <= fallback_index < len(features):
        return fallback_index, features[fallback_index]
    return 0, features[0]


def numeric_evidence_for_feature(feature: dict[str, Any]) -> str:
    dictionary = feature.get("dictionary") or {}
    history = feature.get("history") or {}
    peer = feature.get("peer") or {}
    trend = feature.get("trend") or {}
    seasonality = feature.get("seasonality") or {}
    label = text_or_empty(dictionary.get("label")) or text_or_empty(feature.get("name")) or "degisken"
    parts = [f"feature={label}"]
    add_numeric_part(parts, "current", feature.get("current_value"))
    add_numeric_part(parts, "previous", feature.get("previous_value"))
    add_numeric_part(parts, "change_pct", feature.get("change_pct"), suffix="%")
    add_numeric_part(parts, "history_median", history.get("median"))
    add_numeric_part(parts, "history_z", feature_history_z(feature))
    add_numeric_part(parts, "peer_median", peer.get("peer_median"))
    add_numeric_part(parts, "peer_z", peer.get("peer_z"))
    add_numeric_part(parts, "slope_6m", trend.get("slope_6m"))
    add_numeric_part(parts, "slope_12m", trend.get("slope_12m"))
    add_numeric_part(parts, "yoy_change_pct", seasonality.get("yoy_change_pct"), suffix="%")
    add_numeric_part(parts, "same_month_z", seasonality.get("same_month_customer_z"))
    add_text_part(parts, "risk_direction", dictionary.get("risk_direction"))
    return "; ".join(parts)


def add_numeric_part(parts: list[str], key: str, value: Any, *, suffix: str = "") -> None:
    formatted = format_reason_number(value)
    if formatted is not None:
        parts.append(f"{key}={formatted}{suffix}")


def add_text_part(parts: list[str], key: str, value: Any) -> None:
    text = text_or_empty(value)
    if text:
        parts.append(f"{key}={text}")


def append_reason_numeric_evidence(reason: str, evidence_text: str, *, prefix: str = "Sayisal kanit") -> str:
    combined = f"{reason.rstrip('. ')}. {prefix}: {evidence_text}."
    return combined[:1200]


def has_structured_numeric_evidence(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "sayisal kanit:" in lowered or "sayisal ozet:" in lowered:
        return True
    numeric_keys = {
        match.group(1).lower()
        for match in re.finditer(
            r"\b(current|cari|previous|onceki|change_pct|degisim|history_z|hist_z|peer_z|peer_median|history_median)\s*[=:]",
            text,
            flags=re.IGNORECASE,
        )
    }
    return len(numeric_keys) >= 3


def feature_history_z(feature: dict[str, Any]) -> float | None:
    history = feature.get("history") or {}
    current = clean_float(feature.get("current_value"))
    median = clean_float(history.get("median"))
    robust_scale = clean_float(history.get("robust_scale"))
    if current is None or median is None or robust_scale is None or robust_scale <= 0:
        return None
    return round((current - median) / robust_scale, 6)


def format_reason_number(value: Any) -> str | None:
    parsed = clean_float(value)
    if parsed is None:
        return None
    if parsed == 0:
        return "0"
    formatted = f"{parsed:.6g}"
    return formatted


def normalize_match_text(value: Any) -> str:
    return " ".join(sorted(match_tokens(value)))


def match_tokens(value: Any) -> set[str]:
    text = text_or_empty(value).lower().replace("_", " ")
    return {token for token in re.split(r"[^0-9a-zA-Z]+", text) if len(token) >= 3}


def group_evidence_by_customer(evidence_items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    group_map: dict[str, list[dict[str, Any]]] = {}
    for row_index, item in enumerate(evidence_items):
        customer_id = str(item.get("mono_id") or f"ROW_{row_index}")
        group = group_map.get(customer_id)
        if group is None:
            group = []
            group_map[customer_id] = group
            groups.append((customer_id, group))
        group.append(item)
    for _, group in groups:
        group.sort(key=evidence_period_sort_key)
    return groups


def evidence_period_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("cohort_dt") or ""), str(item.get("mono_id") or ""))


def model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [model_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: model_to_dict(item) for key, item in value.items()}
    return value


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def load_llm_settings() -> dict[str, Any]:
    """Load OpenAI-compatible LLM settings from env/local env files or secrets.yaml."""

    load_local_env_files()
    secret_settings = load_llm_secret_settings()
    settings = {
        "base_url": first_non_empty(
            os.environ.get("LLM_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            secret_settings.get("base_url"),
        ),
        "api_key": first_non_empty(
            os.environ.get("LLM_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            secret_settings.get("api_key"),
            secret_settings.get("openai_api_key"),
        ),
        "model": first_non_empty(
            os.environ.get("LLM_MODEL"),
            secret_settings.get("model"),
        ),
        "timeout_seconds": None,
        "max_retries": parse_non_negative_int(
            first_non_empty(
                os.environ.get("LLM_MAX_RETRIES"),
                secret_settings.get("max_retries"),
                0,
            ),
            name="LLM max_retries",
        ),
        "max_tokens": parse_optional_positive_int(
            first_non_empty(
                os.environ.get("LLM_MAX_TOKENS"),
                secret_settings.get("max_tokens"),
            ),
            name="LLM max_tokens",
        ),
        "http_trust_env": parse_bool(
            first_non_empty(
                os.environ.get("LLM_HTTP_TRUST_ENV"),
                secret_settings.get("http_trust_env"),
                False,
            ),
            name="LLM http_trust_env",
        ),
        "ssl_verify": parse_bool(
            first_non_empty(
                os.environ.get("LLM_SSL_VERIFY"),
                secret_settings.get("ssl_verify"),
                False,
            ),
            name="LLM ssl_verify",
        ),
        "ca_bundle": resolve_ca_bundle(secret_settings),
        "source": secret_settings.get("_source", "env/default"),
    }
    logger.info(
        "LLM settings resolved: base_url=%s model=%s key_source=%s timeout_seconds=%s max_retries=%s max_tokens=%s http_trust_env=%s proxy_env_present=%s ssl_verify=%s ca_bundle=%s structured_call=with_structured_output_schema_only client=langchain_structured",
        mask_url(str(settings["base_url"])),
        settings["model"],
        llm_key_source(secret_settings),
        settings["timeout_seconds"],
        settings["max_retries"],
        settings["max_tokens"],
        settings["http_trust_env"],
        proxy_env_present(),
        settings["ssl_verify"],
        settings["ca_bundle"],
    )
    return settings


def load_llm_secret_settings() -> dict[str, Any]:
    try:
        secrets = load_secrets()
    except FileNotFoundError:
        return {}
    llm = secrets.get("llm") if isinstance(secrets, dict) else None
    if not isinstance(llm, dict):
        return {}

    sections = llm.get("sections")
    if isinstance(sections, dict):
        section_name = (
            os.environ.get("LLM_SECTION")
            or llm.get("section")
            or llm.get("default_section")
            or next(iter(sections), None)
        )
        selected = sections.get(str(section_name)) if section_name is not None else None
        if not isinstance(selected, dict):
            available = ", ".join(str(key) for key in sections.keys())
            raise RuntimeError(f"LLM secret section not found: {section_name}. Available sections: {available}")
        result = dict(selected)
        result["_source"] = f"secret/secrets.yaml llm.sections.{section_name}"
        return result

    result = dict(llm)
    result["_source"] = "secret/secrets.yaml llm"
    return result


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "evet"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "hayir"}:
            return False
    raise RuntimeError(f"Invalid {name} value: {value}")


def proxy_env_present() -> bool:
    return any(bool(os.environ.get(name)) for name in PROXY_ENV_VARS)


def resolve_ca_bundle(secret_settings: dict[str, Any]) -> str | None:
    explicit = first_non_empty(
        os.environ.get("LLM_CA_BUNDLE"),
        os.environ.get("LLM_SSL_CERT_FILE"),
        secret_settings.get("ca_bundle"),
        secret_settings.get("ssl_ca_bundle"),
    )
    if explicit:
        return require_existing_path(explicit, name="LLM CA bundle")

    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        candidate = os.environ.get(env_name)
        if candidate and Path(str(candidate)).expanduser().exists():
            return str(Path(str(candidate)).expanduser())

    for candidate in COMMON_CA_BUNDLE_PATHS:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def require_existing_path(value: Any, *, name: str) -> str:
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise RuntimeError(f"{name} file not found: {value}")
    return str(path)


def parse_non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc
    if parsed < 0:
        raise RuntimeError(f"Invalid {name} value: {value}")
    return parsed


def parse_optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc
    if parsed <= 0:
        raise RuntimeError(f"Invalid {name} value: {value}")
    return parsed


def llm_key_source(secret_settings: dict[str, Any]) -> str:
    if os.environ.get("LLM_API_KEY"):
        return "env:LLM_API_KEY"
    if os.environ.get("OPENAI_API_KEY"):
        return "env:OPENAI_API_KEY"
    if secret_settings.get("api_key") or secret_settings.get("openai_api_key"):
        return str(secret_settings.get("_source", "secret/secrets.yaml"))
    return "missing"


def mask_url(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    host = rest.split("/", 1)[0]
    suffix = "/" + rest.split("/", 1)[1] if "/" in rest else ""
    if len(host) <= 8:
        masked_host = host[:2] + "***"
    else:
        masked_host = host[:4] + "***" + host[-4:]
    return f"{scheme}://{masked_host}{suffix}"


def run_decisions(evidence_items: list[dict[str, Any]], *, dry_run: bool = False) -> list[dict[str, Any]]:
    log_step("04", "LLM modelinden anomali karari aliniyor")
    decisions = []
    total = len(evidence_items)
    logger.info(
        "Starting LLM decision step: evidence_items=%s dry_run=%s call_pattern=one_chain_invoke_per_customer_snapshot",
        total,
        dry_run,
    )
    chain = None if dry_run else build_langchain_structured_chain()
    for index, item in enumerate(evidence_items, start=1):
        customer_id = str(item.get("mono_id") or f"ROW_{index}")
        history_periods = int((item.get("data_quality") or {}).get("customer_history_periods") or 0)
        logger.info(
            "LLM customer snapshot decision progress: %s/%s mono_id=%s cohort_dt=%s decision_items=1 customer_history_periods=%s",
            index,
            total,
            customer_id,
            item.get("cohort_dt"),
            history_periods,
        )
        if dry_run:
            decisions.append(
                {
                    "mono_id": customer_id,
                    "cohort_dt": item.get("cohort_dt"),
                    "dry_run": True,
                    "input_records": format_evidence_for_langchain([item]),
                }
            )
        else:
            try:
                decisions.extend(
                    invoke_langchain_structured_decisions(
                        chain,
                        [item],
                        payload_preview_index=index,
                    )
                )
            except Exception as exc:
                log_step_failed(
                    "04",
                    f"LLM decision failed at customer snapshot {index}/{total} mono_id={customer_id} cohort_dt={item.get('cohort_dt')}: {exc}",
                )
                raise
    logger.info("Completed LLM decision step: decisions=%s", len(decisions))
    log_step_done("04", f"llm_decisions={len(decisions)} dry_run={dry_run}")
    return decisions


def run_full_ml_scoring_for_llm_selection(
    *,
    table_key: str,
    scoring_month: str | None,
    max_train_rows: int | None,
    output_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir = Path(output_path).parent / "ml_full_scoring"
    log_step("00M", "LLM oncesi tum scoring cohort icin ML anomaly skorlamasi yapiliyor")
    logger.info(
        "Starting full ML scoring before LLM selection: table_key=%s scoring_month=%s max_train_rows=%s output_dir=%s",
        table_key,
        scoring_month or "latest",
        max_train_rows,
        output_dir,
    )
    summary = run_multivar_anomaly(
        source="oracle",
        table_key=table_key,
        scoring_month=scoring_month,
        max_train_rows=max_train_rows,
        output_dir=output_dir,
        persist_oracle_outputs=False,
    )
    score_frame = pd.read_csv(summary["scores_path"], encoding="utf-8-sig")
    logger.info(
        "Full ML scoring completed before LLM selection: scored_rows=%s scores_path=%s alert_counts=%s",
        summary.get("scored_rows"),
        summary.get("scores_path"),
        summary.get("alert_counts"),
    )
    log_step_done(
        "00M",
        f"scored_rows={summary.get('scored_rows')} scores_path={summary.get('scores_path')} alert_counts={summary.get('alert_counts')}",
    )
    return summary, score_frame


def attach_evidence_feature_details(
    decisions: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_lookup = {
        (str(item.get("mono_id") or "").strip(), pd.Timestamp(item.get("cohort_dt")).strftime("%Y-%m-%d")): item
        for item in evidence_items
        if item.get("mono_id") and item.get("cohort_dt")
    }
    attached = 0
    for decision in decisions:
        item = evidence_lookup.get(decision_lookup_key(decision))
        if item is None:
            continue
        details = item.get("feature_details") or item.get("features") or []
        decision["evidence_features"] = details
        decision["evidence_feature_count"] = len(details)
        decision["evidence_data_quality"] = item.get("data_quality") or {}
        decision["evidence_peer_definition"] = item.get("peer_definition") or {}
        enrich_decision_reasons_with_numeric_evidence(decision, item)
        attached += 1
    logger.info(
        "Attached evidence feature details to decisions: attached=%s/%s",
        attached,
        len(decisions),
    )
    return decisions


def select_ml_balanced_customer_ids(
    score_frame: pd.DataFrame,
    *,
    total_customers: int = 10,
    anomaly_customers: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    if total_customers <= 0:
        return [], pd.DataFrame()

    frame = score_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"ML selection score frame does not include {ID_COLUMN}.")
    if TIME_COLUMN in frame.columns:
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce").dt.strftime("%Y-%m-%d")

    score_source = resolve_ml_selection_score_column(frame)
    frame["_selection_score"] = pd.to_numeric(frame[score_source], errors="coerce").fillna(-1.0)
    if "alert_band" in frame.columns:
        frame["_alert_band"] = frame["alert_band"].astype(str).str.upper().fillna("")
        frame["_is_ml_anomaly"] = frame["_alert_band"].ne("NORMAL")
    elif "is_anomaly" in frame.columns:
        frame["_alert_band"] = ""
        frame["_is_ml_anomaly"] = frame["is_anomaly"].map(parse_selection_bool)
    else:
        raise ValueError("ML selection score frame must include alert_band or is_anomaly for anomaly/normal split.")
    frame = frame.sort_values("_selection_score", ascending=False).drop_duplicates(subset=[ID_COLUMN], keep="first")

    anomaly_target = balanced_anomaly_target(total_customers, anomaly_customers)
    normal_target = total_customers - anomaly_target
    anomaly_rows = frame[frame["_is_ml_anomaly"]].sort_values("_selection_score", ascending=False).head(anomaly_target)
    normal_rows = (
        frame[~frame[ID_COLUMN].isin(anomaly_rows[ID_COLUMN]) & ~frame["_is_ml_anomaly"]]
        .sort_values("_selection_score", ascending=True)
        .head(normal_target)
    )
    if len(anomaly_rows) < anomaly_target or len(normal_rows) < normal_target:
        raise ValueError(
            "ML balanced selection could not satisfy requested split: "
            f"total={total_customers} anomaly_target={anomaly_target} anomaly_available={len(anomaly_rows)} "
            f"normal_target={normal_target} normal_available={len(normal_rows)} score_column={score_source}"
        )
    selected = pd.concat([anomaly_rows, normal_rows], ignore_index=True)

    selected = selected.head(total_customers).copy()
    selected["selection_bucket"] = ["ML_HIGH_ANOMALY"] * len(anomaly_rows) + ["ML_NORMAL_REFERENCE"] * len(normal_rows)
    selected["selection_score"] = selected["_selection_score"]
    selected["selection_score_column"] = score_source
    selected["selection_model"] = ml_selection_model_name(score_source)
    customer_ids = selected[ID_COLUMN].astype(str).tolist()
    logger.info(
        "ML BALANCED CUSTOMER SELECTION | total=%s anomaly_requested=%s anomaly_selected=%s normal_selected=%s score_column=%s selection_model=%s selected_ids=%s",
        len(customer_ids),
        anomaly_target,
        int(selected["selection_bucket"].eq("ML_HIGH_ANOMALY").sum()),
        int(selected["selection_bucket"].eq("ML_NORMAL_REFERENCE").sum()),
        score_source,
        ml_selection_model_name(score_source),
        ",".join(customer_ids),
    )

    if output_path is not None:
        path = Path(output_path).parent / "ml_balanced_selected_customers.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.drop(columns=[column for column in selected.columns if column.startswith("_")], errors="ignore").to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("Wrote ML balanced customer selection to %s", path)

    return customer_ids, selected.drop(columns=[column for column in selected.columns if column.startswith("_")], errors="ignore")


def balanced_anomaly_target(total_customers: int, explicit_anomaly_customers: int | None = None) -> int:
    if explicit_anomaly_customers is None:
        return max(total_customers // 2, 0)
    return min(max(int(explicit_anomaly_customers), 0), total_customers)


def resolve_ml_selection_score_column(frame: pd.DataFrame) -> str:
    for column in ML_SELECTION_SCORE_PRIORITY:
        if column in frame.columns:
            return column
    raise ValueError(
        "ML selection score frame does not include any supported score column: "
        + ", ".join(ML_SELECTION_SCORE_PRIORITY)
    )


def ml_selection_model_name(score_column: str) -> str:
    mapping = {
        "ensemble_score": "ensemble",
        "anomaly_score": "ensemble_alias",
        "autoencoder_score": "autoencoder",
        "residual_score": "residual",
        "if_score": "isolation_forest",
    }
    return mapping.get(str(score_column), str(score_column))


def parse_selection_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    text = str(value).strip().upper()
    return text in {"1", "TRUE", "T", "YES", "Y", "EVET", "ANOMALY"}


def attach_ml_companion_scores(
    decisions: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    *,
    table_key: str,
    scoring_month: str | None,
    max_train_rows: int | None,
    output_path: str | Path,
    score_frame: pd.DataFrame | None = None,
    score_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append non-LLM anomaly scores for the same customer snapshots.

    These columns are persisted only for comparison. They are not sent to the
    LLM prompt and do not influence the LLM decision.
    """

    if not decisions:
        return decisions
    customer_ids = distinct_evidence_customer_ids(evidence_items)
    if not customer_ids:
        logger.warning("ML companion scoring skipped because evidence has no customer ids.")
        return decisions

    selected_month = scoring_month or first_evidence_month(evidence_items)
    summary = score_summary or {}
    if score_frame is None:
        companion_dir = Path(output_path).parent / "ml_companion"
        log_step("04M", "Ayni musteri snapshotlari icin ML anomaly skorlari uretiliyor")
        logger.info(
            "Starting ML companion scoring: table_key=%s scoring_month=%s customers=%s max_train_rows=%s output_dir=%s",
            table_key,
            selected_month or "latest",
            len(customer_ids),
            max_train_rows,
            companion_dir,
        )
        summary = run_multivar_anomaly(
            source="oracle",
            table_key=table_key,
            scoring_month=selected_month,
            max_train_rows=max_train_rows,
            output_dir=companion_dir,
            persist_oracle_outputs=False,
            score_customer_ids=customer_ids,
        )
        score_frame = pd.read_csv(summary["scores_path"], encoding="utf-8-sig")
    else:
        log_step("04M", "Full cohort ML skorlarindan LLM karar satirlarina karsilastirma kolonlari ekleniyor")
        logger.info(
            "Attaching precomputed ML full-cohort scores: scoring_month=%s decisions=%s score_rows=%s scores_path=%s",
            selected_month or "latest",
            len(decisions),
            len(score_frame),
            summary.get("scores_path"),
        )
    score_lookup = ml_score_lookup(score_frame)
    attached = 0
    for decision in decisions:
        key = decision_lookup_key(decision)
        row = score_lookup.get(key)
        if row is None:
            row = score_lookup.get((str(decision.get("mono_id")), ""))
        if row is None:
            continue
        ensemble_value = clean_float(row.get("ensemble_score"))
        if ensemble_value is None:
            ensemble_value = clean_float(row.get("anomaly_score"))
        decision["ml_ensemble_score"] = ensemble_value
        decision["ml_if_score"] = clean_float(row.get("if_score"))
        decision["ml_residual_score"] = clean_float(row.get("residual_score"))
        decision["ml_autoencoder_score"] = clean_float(row.get("autoencoder_score"))
        decision["ml_alert_band"] = text_or_empty(row.get("alert_band"))
        decision["ml_is_anomaly"] = text_or_empty(row.get("alert_band")).upper() != "NORMAL"
        attached += 1
    logger.info(
        "Completed ML companion scoring: attached=%s/%s scores_path=%s scored_rows=%s",
        attached,
        len(decisions),
        summary.get("scores_path"),
        summary.get("scored_rows"),
    )
    log_step_done("04M", f"ml_scores_attached={attached}/{len(decisions)} scores_path={summary.get('scores_path')}")
    return decisions


def distinct_evidence_customer_ids(evidence_items: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in evidence_items:
        customer_id = str(item.get("mono_id") or "").strip()
        if not customer_id or customer_id in seen:
            continue
        seen.add(customer_id)
        ids.append(customer_id)
    return ids


def first_evidence_month(evidence_items: list[dict[str, Any]]) -> str | None:
    for item in evidence_items:
        value = item.get("cohort_dt")
        if value:
            return str(value)
    return None


def ml_score_lookup(score_frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    frame = score_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        customer_id = str(row.get(ID_COLUMN) or "").strip()
        month_value = row.get(TIME_COLUMN)
        month = "" if pd.isna(month_value) else pd.Timestamp(month_value).strftime("%Y-%m-%d")
        payload = row.to_dict()
        lookup[(customer_id, month)] = payload
        lookup.setdefault((customer_id, ""), payload)
    return lookup


def decision_lookup_key(decision: dict[str, Any]) -> tuple[str, str]:
    customer_id = str(decision.get("mono_id") or "").strip()
    month_value = decision.get("cohort_dt")
    month = "" if month_value is None else pd.Timestamp(month_value).strftime("%Y-%m-%d")
    return customer_id, month


def clean_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build evidence and optionally run LLM anomaly decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-evidence")
    build_parser.add_argument("input_path")
    build_parser.add_argument("output_path")
    build_parser.add_argument("--from-results", action="store_true")
    build_parser.add_argument("--scoring-month")
    build_parser.add_argument("--max-customers", type=int)
    build_parser.add_argument("--top-features", type=int, default=12)
    build_parser.add_argument("--series-periods", type=int, default=6)

    oracle_parser = subparsers.add_parser("build-oracle")
    oracle_parser.add_argument("output_path")
    oracle_parser.add_argument("--scoring-month")
    oracle_parser.add_argument("--max-customers", type=int)
    oracle_parser.add_argument("--max-train-rows", type=int, default=300_000)
    oracle_parser.add_argument("--top-features", type=int, default=12)
    oracle_parser.add_argument("--series-periods", type=int, default=6)
    oracle_parser.add_argument("--table-key", default="multivar_input")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("input_path")
    run_parser.add_argument("output_path")
    run_parser.add_argument("--from-evidence", action="store_true")
    run_parser.add_argument("--from-results", action="store_true")
    run_parser.add_argument("--scoring-month")
    run_parser.add_argument("--max-customers", type=int)
    run_parser.add_argument("--top-features", type=int, default=12)
    run_parser.add_argument("--series-periods", type=int, default=6)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--persist-oracle", action="store_true")
    run_parser.add_argument("--evidence-source", default="file")

    run_oracle_parser = subparsers.add_parser("run-oracle")
    run_oracle_parser.add_argument("output_path")
    run_oracle_parser.add_argument("--scoring-month")
    run_oracle_parser.add_argument("--max-customers", type=int)
    run_oracle_parser.add_argument("--max-train-rows", type=int, default=300_000)
    run_oracle_parser.add_argument("--top-features", type=int, default=12)
    run_oracle_parser.add_argument("--series-periods", type=int, default=6)
    run_oracle_parser.add_argument("--table-key", default="multivar_input")
    run_oracle_parser.add_argument("--persist-oracle", action="store_true", default=True)
    run_oracle_parser.add_argument("--dry-run", action="store_true")
    run_oracle_parser.add_argument("--skip-ml-companion", action="store_true")
    run_oracle_parser.add_argument(
        "--customer-selection-mode",
        choices=["ml-balanced", "first"],
        default="ml-balanced",
        help="LLM'e gidecek musterileri secme yontemi. ml-balanced: once full ML skorla, N/2 yuksek anomaly + kalan normal referans sec.",
    )
    run_oracle_parser.add_argument("--ml-balanced-anomaly-count", type=int, default=None)

    ensure_parser = subparsers.add_parser("ensure-output-tables")
    ensure_parser.add_argument("--scoring-month")

    args = parser.parse_args(argv)
    logger.info("LLM anomaly CLI started: command=%s", args.command)
    if args.command == "build-evidence":
        logger.info("Loading input file: %s", args.input_path)
        frame = load_input_frame(args.input_path)
        if args.from_results:
            evidence = build_evidence_from_result_rows(frame, max_customers=args.max_customers)
        else:
            evidence = build_evidence_packages(
                frame,
                EvidenceConfig(
                    scoring_month=args.scoring_month,
                    max_customers=args.max_customers,
                    top_features=args.top_features,
                    series_periods=args.series_periods,
                ),
            )
        output_path = write_jsonl(evidence, args.output_path)
        logger.info("Wrote %s evidence packages to %s", len(evidence), output_path)
        print(f"wrote {len(evidence)} evidence packages to {output_path}")
        return 0

    if args.command == "build-oracle":
        logger.info(
            "Building Oracle evidence: table_key=%s scoring_month=%s max_customers=%s max_train_rows=%s top_features=%s",
            args.table_key,
            args.scoring_month or "latest",
            args.max_customers,
            args.max_train_rows,
            args.top_features,
        )
        evidence = build_evidence_packages_from_oracle(
            scoring_month=args.scoring_month,
            max_customers=args.max_customers,
            max_train_rows=args.max_train_rows,
            top_features=args.top_features,
            series_periods=args.series_periods,
            table_key=args.table_key,
        )
        output_path = write_jsonl(evidence, args.output_path)
        logger.info("Wrote %s Oracle evidence packages to %s", len(evidence), output_path)
        print(f"wrote {len(evidence)} Oracle evidence packages to {output_path}")
        return 0

    if args.command == "run-oracle":
        log_step("00", "LLM Oracle anomaly run basladi")
        llm_max_customers = args.max_customers
        ml_selection_summary: dict[str, Any] | None = None
        ml_selection_score_frame: pd.DataFrame | None = None
        selected_customer_ids: list[str] | None = None
        selection_rule: str | None = None
        if args.customer_selection_mode == "ml-balanced":
            llm_max_customers = args.max_customers or 10
            ml_selection_summary, ml_selection_score_frame = run_full_ml_scoring_for_llm_selection(
                table_key=args.table_key,
                scoring_month=args.scoring_month,
                max_train_rows=args.max_train_rows,
                output_path=args.output_path,
            )
            selected_customer_ids, selected_rows = select_ml_balanced_customer_ids(
                ml_selection_score_frame,
                total_customers=llm_max_customers,
                anomaly_customers=args.ml_balanced_anomaly_count,
                output_path=args.output_path,
            )
            anomaly_selected = int(selected_rows["selection_bucket"].eq("ML_HIGH_ANOMALY").sum())
            normal_selected = int(selected_rows["selection_bucket"].eq("ML_NORMAL_REFERENCE").sum())
            selection_score_column = str(selected_rows["selection_score_column"].iloc[0]) if not selected_rows.empty else ""
            selection_model = str(selected_rows["selection_model"].iloc[0]) if not selected_rows.empty else ""
            selection_rule = (
                f"ml_balanced_full_cohort_{selection_model}_"
                f"top_anomaly_{anomaly_selected}_plus_normal_{normal_selected}"
            )
            logger.info(
                "LLM customer selection is ML-balanced: selected_customers=%s anomaly_selected=%s normal_selected=%s score_column=%s selection_model=%s note='ML scores are used only before evidence build and are not added to the LLM prompt'",
                ",".join(selected_customer_ids),
                anomaly_selected,
                normal_selected,
                selection_score_column,
                selection_model,
            )
        logger.info(
            "Running Oracle-to-LLM flow: table_key=%s scoring_month=%s max_customers=%s customer_selection_mode=%s max_train_rows=%s top_features=%s dry_run=%s persist_oracle=%s",
            args.table_key,
            args.scoring_month or "latest",
            llm_max_customers,
            args.customer_selection_mode,
            args.max_train_rows,
            args.top_features,
            args.dry_run,
            args.persist_oracle and not args.dry_run,
        )
        if args.persist_oracle and not args.dry_run:
            logger.info("Ensuring Oracle output tables before LLM call so structured output target exists even if LLM fails.")
            ensure_llm_output_tables_in_oracle(scoring_month=args.scoring_month)
        evidence = build_evidence_packages_from_oracle(
            scoring_month=args.scoring_month,
            max_customers=llm_max_customers,
            max_train_rows=args.max_train_rows,
            top_features=args.top_features,
            series_periods=args.series_periods,
            table_key=args.table_key,
            selected_customer_ids=selected_customer_ids,
            selection_rule=selection_rule,
        )
        try:
            decisions = run_decisions(evidence, dry_run=args.dry_run)
        except Exception as exc:
            log_step_skipped("05", f"LLM karar uretilemedi; Oracle output tablolari doldurulmadi. reason={exc}")
            if evidence:
                audit_llm_output_tables(evidence[0].get("cohort_dt"))
            return 2
        decisions = attach_evidence_feature_details(decisions, evidence)
        if not args.dry_run and not args.skip_ml_companion:
            try:
                decisions = attach_ml_companion_scores(
                    decisions,
                    evidence,
                    table_key=args.table_key,
                    scoring_month=args.scoring_month,
                    max_train_rows=args.max_train_rows,
                    output_path=args.output_path,
                    score_frame=ml_selection_score_frame,
                    score_summary=ml_selection_summary,
                )
            except Exception as exc:
                log_step_failed("04M", f"ML companion scoring failed: {exc}")
                return 4
        output_path = write_jsonl(decisions, args.output_path)
        logger.info("Wrote %s LLM decision rows to %s", len(decisions), output_path)
        print(f"wrote {len(decisions)} LLM decision rows to {output_path}")
        if args.persist_oracle and not args.dry_run:
            log_step("05", "LLM kararlari Oracle output tablolarina yaziliyor")
            logger.info("Persisting LLM decisions to Oracle output tables.")
            try:
                oracle_result = write_llm_outputs_to_oracle(
                    decisions,
                    llm_model=current_model_name(),
                    evidence_source="oracle_input",
                )
            except Exception as exc:
                log_step_failed("05", f"Oracle output write failed: {exc}")
                if evidence:
                    audit_llm_output_tables(evidence[0].get("cohort_dt"))
                return 3
            logger.info("Oracle persistence completed: %s", oracle_result)
            print(json.dumps(oracle_result, ensure_ascii=False))
            log_step_done(
                "05",
                f"inserted_results={oracle_result.get('inserted_results')} inserted_reasons={oracle_result.get('inserted_reasons')} inserted_features={oracle_result.get('inserted_features')} results_table={oracle_result.get('results_table')} reasons_table={oracle_result.get('reasons_table')} features_table={oracle_result.get('features_table')}",
            )
        elif evidence:
            log_step("05", "Dry-run output tablo audit'i yapiliyor")
            logger.info("Oracle persistence skipped: dry_run=%s persist_oracle=%s", args.dry_run, args.persist_oracle)
            audit_llm_output_tables(evidence[0].get("cohort_dt"))
        return 0

    if args.command == "ensure-output-tables":
        log_step("00", "LLM Oracle output tablolari olusturuluyor veya kontrol ediliyor")
        result = ensure_llm_output_tables_in_oracle(scoring_month=args.scoring_month)
        log_step_done("00", "output table ensure completed")
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.from_evidence:
        logger.info("Reading evidence JSONL: %s", args.input_path)
        evidence = read_jsonl(args.input_path)
    elif args.from_results:
        logger.info("Loading existing result rows: %s", args.input_path)
        frame = load_input_frame(args.input_path)
        evidence = build_evidence_from_result_rows(frame, max_customers=args.max_customers)
    else:
        logger.info("Loading raw input rows: %s", args.input_path)
        frame = load_input_frame(args.input_path)
        evidence = build_evidence_packages(
            frame,
            EvidenceConfig(
                scoring_month=args.scoring_month,
                max_customers=args.max_customers,
                top_features=args.top_features,
                series_periods=args.series_periods,
            ),
        )
    try:
        decisions = run_decisions(evidence, dry_run=args.dry_run)
    except Exception as exc:
        log_step_skipped("05", f"LLM karar uretilemedi; Oracle output tablolari doldurulmadi. reason={exc}")
        if evidence:
            audit_llm_output_tables(evidence[0].get("cohort_dt"))
        return 2
    decisions = attach_evidence_feature_details(decisions, evidence)
    output_path = write_jsonl(decisions, args.output_path)
    logger.info("Wrote %s LLM decision rows to %s", len(decisions), output_path)
    print(f"wrote {len(decisions)} LLM decision rows to {output_path}")
    if args.persist_oracle and not args.dry_run:
        log_step("05", "LLM kararlari Oracle output tablolarina yaziliyor")
        logger.info("Persisting LLM decisions to Oracle output tables.")
        try:
            oracle_result = write_llm_outputs_to_oracle(
                decisions,
                llm_model=current_model_name(),
                evidence_source=args.evidence_source,
            )
        except Exception as exc:
            log_step_failed("05", f"Oracle output write failed: {exc}")
            if evidence:
                audit_llm_output_tables(evidence[0].get("cohort_dt"))
            return 3
        logger.info("Oracle persistence completed: %s", oracle_result)
        print(json.dumps(oracle_result, ensure_ascii=False))
        log_step_done(
            "05",
            f"inserted_results={oracle_result.get('inserted_results')} inserted_reasons={oracle_result.get('inserted_reasons')} inserted_features={oracle_result.get('inserted_features')} results_table={oracle_result.get('results_table')} reasons_table={oracle_result.get('reasons_table')} features_table={oracle_result.get('features_table')}",
        )
    elif evidence:
        log_step("05", "Dry-run output tablo audit'i yapiliyor")
        logger.info("Oracle persistence skipped: dry_run=%s persist_oracle=%s", args.dry_run, args.persist_oracle)
        audit_llm_output_tables(evidence[0].get("cohort_dt"))
    return 0


def log_step(step_no: str, title: str) -> None:
    logger.info("========== STEP %s START | %s ==========", step_no, title)


def log_step_done(step_no: str, detail: str) -> None:
    logger.info("========== STEP %s DONE | %s ==========", step_no, detail)


def log_step_failed(step_no: str, reason: str) -> None:
    logger.error("========== STEP %s FAILED | %s ==========", step_no, reason)


def log_step_skipped(step_no: str, reason: str) -> None:
    logger.warning("========== STEP %s SKIPPED | %s ==========", step_no, reason)


def current_model_name() -> str:
    return str(load_llm_settings().get("model") or "unknown")


if __name__ == "__main__":
    raise SystemExit(main())
