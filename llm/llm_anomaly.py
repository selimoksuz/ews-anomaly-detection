"""OpenAI-compatible LLM anomaly decision runner."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

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

SYSTEM_PROMPT = """Sen deneyimli bir banka risk yoneticisi ve finansal anomali uzmanisin.
Sana tek bir musteriye ait birden fazla doneme ait kredi risk kaydi verilecek.
Kayitlar kronolojik siraya gore siralanmistir.

Once musterinin tum donemlerini birlikte incele.
Bir donemin anomali olup olmadigina musterinin kendi tarihsel seyri, peer bilgisi, trend, sezon ve veri kalitesi sinyalleri isiginda karar ver.

Kurallar:
- Degisken sozlugunu oku: is anlami, formul, risk yonu ve birimi dikkate al.
- Cari degeri musterinin kendi gecmisiyle, ayni sezon gecmisiyle ve peer grubuyla karsilastir.
- Tek donem sicrama, kademeli trend bozulmasi, sezon etkisi ve veri kalitesi problemini ayir.
- Buyuk tutar tek basina anomali degildir; olcek, peer ve tarihsel davranisla birlikte yorumla.
- Missing veya stale finansal term sinyalini finansal bozulma gibi yazma; veri kalitesi veya inceleme nedeni olarak ayir.
- Peer kalitesi ZAYIF ise kesin hukum verme, manuel inceleme oner.
- Risk azalisi olan sapmalari anomali nedeni yapma.
- PD ve rating ayni risk bilgisinin farkli gosterimleri olabilir; ayni bilgiyi cift kanit gibi sayma.
- Gelecek donem varsayimi yapma.

Her donem icin period_position, mono_id, cohort_dt, is_anomaly, anomaly_type, risk_level, confidence,
seasonality_assessment, trend_assessment, peer_assessment, main_reasons, caveat ve recommended_action dondur.
Sonuc listesi verilen donem sayisiyla ayni uzunlukta olmali.
Sadece gecerli JSON dondur. Markdown kullanma."""

OUTPUT_CONTRACT = {
    "period_position": "0-based integer",
    "mono_id": "string",
    "cohort_dt": "YYYY-MM-DD",
    "is_anomaly": "boolean",
    "anomaly_type": "ANI_RISK_ARTISI | FINANSAL_BOZULMA | PD_RISKI | KKB_RISKI | PEER_UYUMSUZLUGU | DATA_GAP | TREND_KIRILMASI | SEZON_DISI_SAPMA | NORMAL",
    "risk_level": "DUSUK | ORTA | YUKSEK | KRITIK",
    "confidence": "0.0-1.0",
    "seasonality_assessment": "string",
    "trend_assessment": "string",
    "peer_assessment": "string",
    "main_reasons": [
        {
            "feature": "string",
            "evidence": "current, history, seasonality, peer numeric evidence",
            "interpretation": "short Turkish business interpretation",
        }
    ],
    "caveat": "string or null",
    "recommended_action": "Izle | Manuel incele | Portfoy yoneticisine gonder | Limit/risk gozden gecir | Veri kontrolu yap",
}


if BaseModel is not None and Field is not None:

    class AnomalyReasonRecord(BaseModel):
        feature: str = Field(description="Anomali gerekcesindeki degisken adi")
        evidence: str = Field(description="Current, history, seasonality ve peer sayisal kanit ozeti")
        interpretation: str = Field(description="Kanita dayali kisa Turkce is yorumu")

    class AnomalyRecord(BaseModel):
        period_position: int = Field(
            description="Musterinin kronolojik donem listesindeki sira numarasi, 0'dan baslar"
        )
        mono_id: str = Field(description="Musteri tekil numarasi")
        cohort_dt: str = Field(description="Skorlanan donem tarihi, YYYY-MM-DD")
        is_anomaly: bool = Field(description="Bu musteri-donem kaydi anomali mi?")
        anomaly_type: str = Field(
            description=(
                "Anomali tipi: ANI_RISK_ARTISI, FINANSAL_BOZULMA, PD_RISKI, KKB_RISKI, "
                "PEER_UYUMSUZLUGU, DATA_GAP, TREND_KIRILMASI, SEZON_DISI_SAPMA veya NORMAL"
            )
        )
        risk_level: str = Field(description="Risk seviyesi: DUSUK, ORTA, YUKSEK veya KRITIK")
        confidence: float = Field(description="Guven skoru, 0.0 ile 1.0 arasinda")
        seasonality_assessment: str = Field(description="Sezon/mevsimsellik yorumu")
        trend_assessment: str = Field(description="Trend ve kademeli bozulma yorumu")
        peer_assessment: str = Field(description="Peer grubuna gore ayrisma yorumu")
        main_reasons: List[AnomalyReasonRecord] = Field(description="Karari aciklayan ana feature gerekceleri")
        caveat: Optional[str] = Field(default=None, description="Varsa veri kalitesi veya karar kisiti")
        recommended_action: str = Field(
            description="Onerilen aksiyon: Izle, Manuel incele, Portfoy yoneticisine gonder, Limit/risk gozden gecir veya Veri kontrolu yap"
        )

    class AnomalyBatchResult(BaseModel):
        results: List[AnomalyRecord] = Field(
            description="Verilen musteri-donem evidence kayitlari icin kronolojik sira ile karar listesi"
        )
else:
    AnomalyReasonRecord = None
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
            "Install requirements.txt or pip install langchain-openai langchain-core pydantic."
        ) from exc

    llm = ChatOpenAI(
        base_url=str(settings["base_url"]).rstrip("/"),
        api_key=str(settings["api_key"]),
        model=str(settings["model"]),
        temperature=0,
        timeout=int(settings["timeout_seconds"]),
        max_retries=int(settings["max_retries"]),
        **optional_llm_kwargs(settings),
    )
    structured_method = str(settings["structured_method"])
    structured_llm = llm.with_structured_output(schema, method=structured_method)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Asagidaki musteri-donem evidence kayitlarini analiz et:\n\n{input_records}"),
        ]
    )
    logger.info(
        "LangChain structured LLM chain initialized: model=%s structured_method=%s max_retries=%s max_tokens=%s",
        settings["model"],
        structured_method,
        settings["max_retries"],
        settings.get("max_tokens"),
    )
    return prompt | structured_llm


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
    lines = [
        "Gorev: Her musteri-donem icin anomali karari ver.",
        "Cikti: AnomalyBatchResult formatinda JSON; en dis alan results listesi olmali.",
        "Alanlar: period_position, mono_id, cohort_dt, is_anomaly, anomaly_type, risk_level, confidence, seasonality_assessment, trend_assessment, peer_assessment, main_reasons, caveat, recommended_action.",
        "Kayitlar:",
    ]
    for index, item in enumerate(evidence_items):
        lines.extend(compact_evidence_lines(item, index=index))
    return "\n".join(lines)


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
    peer = feature.get("peer") or {}
    data_quality = feature.get("data_quality") or {}
    parts = [
        f"{index}) name={feature.get('name')}",
        f"label={dictionary.get('label')}",
        f"category={dictionary.get('category')}",
        f"formula={dictionary.get('formula')}",
        f"risk_direction={dictionary.get('risk_direction')}",
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


def invoke_langchain_structured_decisions(chain: Any, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evidence_items:
        return []
    input_records = format_evidence_for_langchain(evidence_items)
    first_item = evidence_items[0] if evidence_items else {}
    logger.info(
        "LLM request payload prepared: mono_id=%s periods=%s first_cohort_dt=%s chars=%s total_features=%s formatter=compact_text",
        first_item.get("mono_id"),
        len(evidence_items),
        first_item.get("cohort_dt"),
        len(input_records),
        sum(len(item.get("features") or []) for item in evidence_items),
    )
    response = chain.invoke({"input_records": input_records})
    records = response.results
    if not records:
        raise RuntimeError(f"LLM structured response did not include results: {truncate_text(str(response), 500)}")
    if len(records) != len(evidence_items):
        raise RuntimeError(
            "LLM structured response result count mismatch: "
            f"expected={len(evidence_items)} actual={len(records)} mono_id={first_item.get('mono_id')}"
        )
    decisions = []
    for record in records:
        decision = model_to_dict(record)
        position = int(decision["period_position"])
        if not 0 <= position < len(evidence_items):
            raise RuntimeError(
                "LLM structured response period_position is out of range: "
                f"period_position={position} expected_range=0..{len(evidence_items) - 1} mono_id={first_item.get('mono_id')}"
            )
        evidence = evidence_items[position]
        decision["period_position"] = position
        if not decision.get("mono_id"):
            decision["mono_id"] = evidence.get("mono_id")
        if not decision.get("cohort_dt"):
            decision["cohort_dt"] = evidence.get("cohort_dt")
        decisions.append(decision)
    return decisions


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
        "timeout_seconds": parse_timeout_seconds(
            first_non_empty(
                os.environ.get("LLM_TIMEOUT_SECONDS"),
                secret_settings.get("timeout_seconds"),
                120,
            )
        ),
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
        "structured_method": parse_structured_method(
            first_non_empty(
                os.environ.get("LLM_STRUCTURED_METHOD"),
                secret_settings.get("structured_method"),
                "function_calling",
            )
        ),
        "source": secret_settings.get("_source", "env/default"),
    }
    logger.info(
        "LLM settings resolved: base_url=%s model=%s key_source=%s timeout_seconds=%s max_retries=%s max_tokens=%s structured_method=%s client=langchain_structured",
        mask_url(str(settings["base_url"])),
        settings["model"],
        llm_key_source(secret_settings),
        settings["timeout_seconds"],
        settings["max_retries"],
        settings["max_tokens"],
        settings["structured_method"],
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


def parse_timeout_seconds(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid LLM timeout_seconds value: {value}") from exc
    if timeout <= 0:
        raise RuntimeError(f"Invalid LLM timeout_seconds value: {value}")
    return timeout


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on", "evet"}


def parse_structured_method(value: Any) -> str:
    method = str(value or "").strip().lower()
    allowed = {"function_calling", "json_mode", "json_schema"}
    if method not in allowed:
        raise RuntimeError(f"Invalid LLM structured_method value: {value}. Allowed: {', '.join(sorted(allowed))}")
    return method


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
    customer_groups = group_evidence_by_customer(evidence_items)
    logger.info(
        "Starting LLM decision step: evidence_items=%s customer_groups=%s dry_run=%s call_pattern=one_chain_invoke_per_customer",
        total,
        len(customer_groups),
        dry_run,
    )
    chain = None if dry_run else build_langchain_structured_chain()
    for index, (customer_id, customer_items) in enumerate(customer_groups, start=1):
        history_period_counts = [
            int((item.get("data_quality") or {}).get("customer_history_periods") or 0)
            for item in customer_items
        ]
        logger.info(
            "LLM customer decision progress: %s/%s mono_id=%s decision_items=%s customer_history_periods=%s first_cohort_dt=%s last_cohort_dt=%s",
            index,
            len(customer_groups),
            customer_id,
            len(customer_items),
            max(history_period_counts) if history_period_counts else 0,
            customer_items[0].get("cohort_dt") if customer_items else None,
            customer_items[-1].get("cohort_dt") if customer_items else None,
        )
        if dry_run:
            decisions.append(
                {
                    "mono_id": customer_id,
                    "period_count": len(customer_items),
                    "dry_run": True,
                    "input_records": format_evidence_for_langchain(customer_items),
                }
            )
        else:
            try:
                decisions.extend(invoke_langchain_structured_decisions(chain, customer_items))
            except Exception as exc:
                log_step_failed(
                    "04",
                    f"LLM decision failed at customer {index}/{len(customer_groups)} mono_id={customer_id}: {exc}",
                )
                raise
    logger.info("Completed LLM decision step: decisions=%s", len(decisions))
    log_step_done("04", f"llm_decisions={len(decisions)} dry_run={dry_run}")
    return decisions


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

    oracle_parser = subparsers.add_parser("build-oracle")
    oracle_parser.add_argument("output_path")
    oracle_parser.add_argument("--scoring-month")
    oracle_parser.add_argument("--max-customers", type=int)
    oracle_parser.add_argument("--max-train-rows", type=int, default=300_000)
    oracle_parser.add_argument("--top-features", type=int, default=12)
    oracle_parser.add_argument("--table-key", default="multivar_input")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("input_path")
    run_parser.add_argument("output_path")
    run_parser.add_argument("--from-evidence", action="store_true")
    run_parser.add_argument("--from-results", action="store_true")
    run_parser.add_argument("--scoring-month")
    run_parser.add_argument("--max-customers", type=int)
    run_parser.add_argument("--top-features", type=int, default=12)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--persist-oracle", action="store_true")
    run_parser.add_argument("--evidence-source", default="file")

    run_oracle_parser = subparsers.add_parser("run-oracle")
    run_oracle_parser.add_argument("output_path")
    run_oracle_parser.add_argument("--scoring-month")
    run_oracle_parser.add_argument("--max-customers", type=int)
    run_oracle_parser.add_argument("--max-train-rows", type=int, default=300_000)
    run_oracle_parser.add_argument("--top-features", type=int, default=12)
    run_oracle_parser.add_argument("--table-key", default="multivar_input")
    run_oracle_parser.add_argument("--persist-oracle", action="store_true", default=True)
    run_oracle_parser.add_argument("--dry-run", action="store_true")

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
            table_key=args.table_key,
        )
        output_path = write_jsonl(evidence, args.output_path)
        logger.info("Wrote %s Oracle evidence packages to %s", len(evidence), output_path)
        print(f"wrote {len(evidence)} Oracle evidence packages to {output_path}")
        return 0

    if args.command == "run-oracle":
        log_step("00", "LLM Oracle anomaly run basladi")
        logger.info(
            "Running Oracle-to-LLM flow: table_key=%s scoring_month=%s max_customers=%s max_train_rows=%s top_features=%s dry_run=%s persist_oracle=%s",
            args.table_key,
            args.scoring_month or "latest",
            args.max_customers,
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
            max_customers=args.max_customers,
            max_train_rows=args.max_train_rows,
            top_features=args.top_features,
            table_key=args.table_key,
        )
        try:
            decisions = run_decisions(evidence, dry_run=args.dry_run)
        except Exception as exc:
            log_step_skipped("05", f"LLM karar uretilemedi; Oracle output tablolari doldurulmadi. reason={exc}")
            if evidence:
                audit_llm_output_tables(evidence[0].get("cohort_dt"))
            return 2
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
                f"inserted_results={oracle_result.get('inserted_results')} inserted_reasons={oracle_result.get('inserted_reasons')} results_table={oracle_result.get('results_table')} reasons_table={oracle_result.get('reasons_table')}",
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
            ),
        )
    try:
        decisions = run_decisions(evidence, dry_run=args.dry_run)
    except Exception as exc:
        log_step_skipped("05", f"LLM karar uretilemedi; Oracle output tablolari doldurulmadi. reason={exc}")
        if evidence:
            audit_llm_output_tables(evidence[0].get("cohort_dt"))
        return 2
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
            f"inserted_results={oracle_result.get('inserted_results')} inserted_reasons={oracle_result.get('inserted_reasons')} results_table={oracle_result.get('results_table')} reasons_table={oracle_result.get('reasons_table')}",
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
