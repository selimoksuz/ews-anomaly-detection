"""OpenAI-compatible LLM anomaly decision runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llm.evidence_builder import (
    EvidenceConfig,
    build_evidence_from_result_rows,
    build_evidence_packages_from_oracle,
    build_evidence_packages,
    load_input_frame,
    write_jsonl,
)
from llm.oracle_output import write_llm_outputs_to_oracle


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sen banka kredi riski ve erken uyari anomalisi degerlendiren uzman bir analistsin.

Gorevin verilen musteri-donem evidence paketine gore kaydin anomali olup olmadigini belirlemektir.
Hazir anomaly score veya target yoktur. Karari sen vereceksin, ancak sadece verilen kanit paketine dayanacaksin.

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

Sadece gecerli JSON dondur. Markdown kullanma."""

OUTPUT_CONTRACT = {
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


def configure_logging() -> None:
    """Enable visible console logs for CLI runs."""

    level_name = (os.environ.get("LLM_LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


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


def build_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    user_payload = {
        "task": "Bu musteri-donem kaydi anomali mi? Karari evidence paketinden ver.",
        "output_contract": OUTPUT_CONTRACT,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_openai_compatible_chat(messages: list[dict[str, str]], *, timeout_seconds: int = 120) -> dict[str, Any]:
    load_local_env_files()
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY or OPENAI_API_KEY env variable is required.")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM call failed: HTTP {exc.code}: {body}") from exc

    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    return json.loads(content)


def run_decisions(evidence_items: list[dict[str, Any]], *, dry_run: bool = False) -> list[dict[str, Any]]:
    decisions = []
    total = len(evidence_items)
    logger.info("Starting LLM decision step: evidence_items=%s dry_run=%s", total, dry_run)
    for index, item in enumerate(evidence_items, start=1):
        logger.info(
            "LLM decision progress: %s/%s mono_id=%s cohort_dt=%s",
            index,
            total,
            item.get("mono_id"),
            item.get("cohort_dt"),
        )
        messages = build_messages(item)
        if dry_run:
            decisions.append(
                {
                    "mono_id": item.get("mono_id"),
                    "cohort_dt": item.get("cohort_dt"),
                    "dry_run": True,
                    "messages": messages,
                }
            )
        else:
            decisions.append(call_openai_compatible_chat(messages))
    logger.info("Completed LLM decision step: decisions=%s", len(decisions))
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
        evidence = build_evidence_packages_from_oracle(
            scoring_month=args.scoring_month,
            max_customers=args.max_customers,
            max_train_rows=args.max_train_rows,
            top_features=args.top_features,
            table_key=args.table_key,
        )
        decisions = run_decisions(evidence, dry_run=args.dry_run)
        output_path = write_jsonl(decisions, args.output_path)
        logger.info("Wrote %s LLM decision rows to %s", len(decisions), output_path)
        print(f"wrote {len(decisions)} LLM decision rows to {output_path}")
        if args.persist_oracle and not args.dry_run:
            logger.info("Persisting LLM decisions to Oracle output tables.")
            oracle_result = write_llm_outputs_to_oracle(
                decisions,
                llm_model=current_model_name(),
                evidence_source="oracle_input",
            )
            logger.info("Oracle persistence completed: %s", oracle_result)
            print(json.dumps(oracle_result, ensure_ascii=False))
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
    decisions = run_decisions(evidence, dry_run=args.dry_run)
    output_path = write_jsonl(decisions, args.output_path)
    logger.info("Wrote %s LLM decision rows to %s", len(decisions), output_path)
    print(f"wrote {len(decisions)} LLM decision rows to {output_path}")
    if args.persist_oracle and not args.dry_run:
        logger.info("Persisting LLM decisions to Oracle output tables.")
        oracle_result = write_llm_outputs_to_oracle(
            decisions,
            llm_model=current_model_name(),
            evidence_source=args.evidence_source,
        )
        logger.info("Oracle persistence completed: %s", oracle_result)
        print(json.dumps(oracle_result, ensure_ascii=False))
    return 0


def current_model_name() -> str:
    load_local_env_files()
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


if __name__ == "__main__":
    raise SystemExit(main())
