"""Config-driven lifecycle manager for development, tuning, and scoring."""

from __future__ import annotations

import copy
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from engine.calibration import ScoreCalibrator
from engine.config_loader import get_alert_bands, get_ensemble_weights, get_feature_list, load_config, load_secrets
from engine.data_loader import DataLoader
from engine.models import AnomalyModels
from engine.monitoring import MonitoringManager
from engine.output_writer import OutputWriter
from engine.registry import RegistryManager
from engine.retention import RetentionManager
from engine.scorer import AnomalyScorer
from engine.source_loader import SourceLoader
from engine.weight_tuning import WeightOptimizer
from engine.windowing import WindowResolver, summarize_window


class LifecycleManager:
    """High-level orchestration for development, tuning, evaluation, and live scoring."""

    def __init__(self, config_path=None, secrets_path=None):
        self.config = load_config(config_path)
        self.secrets = load_secrets(secrets_path)
        self.features = get_feature_list(self.config)
        self.id_column = self.config["pipeline"]["id_column"]
        self.time_column = self.config["pipeline"]["time_column"]
        self.development_cfg = self.config.get("development", {})
        self.retraining_cfg = self.config.get("retraining", {})
        self.live_scoring_cfg = self.config.get("live_scoring", {})
        self.calibration_cfg = self.config.get("calibration", {})
        self.weight_cfg = self.config.get("weight_optimization", {})
        self.batch_cfg = self.config.get("batch_execution", {})
        self.shadow_cfg = self.config.get("shadow_scoring", {})
        self.source_loader = SourceLoader(self.config, self.secrets)
        self.registry = RegistryManager(self.config)
        self.retention = RetentionManager(self.config)
        self.output_writer = OutputWriter(self.config, self.secrets)
        self.monitoring = MonitoringManager(self.config)
        self.data_loader = DataLoader(self.config)

    def develop(self, segment: Optional[str] = None):
        return self._run_training_lifecycle(run_type="develop", segment=segment)

    def retrain(self, segment: Optional[str] = None):
        return self._run_training_lifecycle(run_type="retrain", segment=segment)

    def compare(self, segment: Optional[str] = None, challenger_version: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("compare", segment_value, self.config)
        try:
            champion = self.registry.get_champion(segment_value)
            if champion is None:
                raise ValueError(f"No champion model found for segment '{segment_value}'.")
            challenger = self.registry.get_model(challenger_version) if challenger_version else self.registry.get_latest_candidate(segment_value)
            if challenger is None:
                raise ValueError(f"No challenger model found for segment '{segment_value}'.")

            comparison = self._build_comparison(champion, challenger)
            comparison_path = run.run_dir / "comparison.json"
            with open(comparison_path, "w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2, ensure_ascii=False)

            summary = {
                "champion_version": champion["model_version"],
                "challenger_version": challenger["model_version"],
                "recommended_model": comparison["recommendation"]["winner"],
                "comparison_path": str(comparison_path),
            }
            self.registry.finish_run(run, "completed", summary)
            return comparison
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def promote(self, segment: Optional[str] = None, model_version: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("promote", segment_value, self.config)
        try:
            target = self.registry.get_model(model_version) if model_version else self.registry.get_latest_candidate(segment_value)
            if target is None:
                raise ValueError(f"No candidate model available for segment '{segment_value}'.")
            self.registry.promote_model(segment_value, target["model_version"])
            summary = {"segment": segment_value, "promoted_model": target["model_version"]}
            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def cleanup(self):
        run = self.registry.start_run("cleanup", "ALL", self.config)
        try:
            deleted = self.retention.cleanup()
            self.registry.finish_run(run, "completed", {"deleted": deleted})
            return deleted
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def compare_preprocessing(self, segment: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("compare-preprocessing", segment_value, self.config)
        try:
            frames, _ = self._load_development_frames(segment_value)
            train_df = frames.get("train")
            calibration_df = frames.get(self.calibration_cfg.get("source_window", "calibration"))
            oot_df = frames.get(self.weight_cfg.get("validation_window", "oot"))
            if train_df is None or train_df.empty:
                raise ValueError("Train window is empty; cannot compare preprocessing.")
            if oot_df is None or oot_df.empty:
                raise ValueError("OOT window is empty; cannot compare preprocessing.")

            baseline_config = self._build_config_variant(preprocessing_enabled=False)
            robust_config = self._build_config_variant(preprocessing_enabled=True)
            default_weights = get_ensemble_weights(self.config)

            live_frame = self._load_live_frame(segment_value)
            evaluation_frames = [frame for name, frame in frames.items() if name in {"dev", "oot"} and frame is not None and not frame.empty]
            if not evaluation_frames:
                raise ValueError("No non-empty evaluation frames found for preprocessing comparison.")
            label_frame = self._load_label_frame(
                self.weight_cfg.get("source_name", "outcomes"),
                segment_value,
                start_date=min(pd.to_datetime(frame[self.time_column]).min() for frame in evaluation_frames),
                end_date=max(pd.to_datetime(frame[self.time_column]).max() for frame in evaluation_frames),
            )

            baseline_report = self._evaluate_preprocessing_variant(
                baseline_config,
                segment_value=segment_value,
                train_df=train_df,
                dev_df=frames.get("dev"),
                calibration_df=calibration_df,
                oot_df=oot_df,
                live_df=live_frame,
                label_frame=label_frame,
                weights=default_weights,
                tag="baseline",
            )
            robust_report = self._evaluate_preprocessing_variant(
                robust_config,
                segment_value=segment_value,
                train_df=train_df,
                dev_df=frames.get("dev"),
                calibration_df=calibration_df,
                oot_df=oot_df,
                live_df=live_frame,
                label_frame=label_frame,
                weights=default_weights,
                tag="robust",
            )

            sample = self._build_preprocessing_sample_comparison(
                baseline_report["live_scores"],
                robust_report["live_scores"],
            )

            comparison = {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "segment": segment_value,
                "fixed_weights": default_weights,
                "baseline": baseline_report["summary"],
                "robust": robust_report["summary"],
                "delta": {
                    "oot_primary_precision_delta": round(
                        robust_report["summary"]["outcomes"]["primary"]["precision_at_top_percent"]
                        - baseline_report["summary"]["outcomes"]["primary"]["precision_at_top_percent"],
                        4,
                    ),
                    "oot_primary_lift_delta": round(
                        robust_report["summary"]["outcomes"]["primary"]["lift_at_top_percent"]
                        - baseline_report["summary"]["outcomes"]["primary"]["lift_at_top_percent"],
                        4,
                    ),
                    "oot_tuned_precision_delta": round(
                        robust_report["summary"]["tuned_outcomes"]["primary"]["precision_at_top_percent"]
                        - baseline_report["summary"]["tuned_outcomes"]["primary"]["precision_at_top_percent"],
                        4,
                    ),
                    "oot_tuned_lift_delta": round(
                        robust_report["summary"]["tuned_outcomes"]["primary"]["lift_at_top_percent"]
                        - baseline_report["summary"]["tuned_outcomes"]["primary"]["lift_at_top_percent"],
                        4,
                    ),
                    "oot_ensemble_ks_delta": round(
                        robust_report["summary"]["stability"]["metrics"]["ensemble_score"]["ks_stat"]
                        - baseline_report["summary"]["stability"]["metrics"]["ensemble_score"]["ks_stat"],
                        4,
                    ),
                    "live_red_share_delta": round(
                        robust_report["summary"]["live_scores"]["band_share"]["KIRMIZI"]
                        - baseline_report["summary"]["live_scores"]["band_share"]["KIRMIZI"],
                        4,
                    ),
                },
                "sample_customer": sample,
            }

            json_path = run.run_dir / "preprocessing_comparison.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2, ensure_ascii=False)
            markdown_path = run.run_dir / "preprocessing_comparison.md"
            markdown_path.write_text(self._render_preprocessing_comparison_markdown(comparison), encoding="utf-8")

            summary = {
                "segment": segment_value,
                "comparison_path": str(json_path),
                "markdown_path": str(markdown_path),
                "sample_customer_id": sample.get("customer_id"),
            }
            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def reset_runtime(self):
        return self.retention.reset_runtime_state()

    def run_batch(self, segment: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("run-batch", segment_value, self.config)
        try:
            summary = {"segment": segment_value, "steps": {}}
            champion = self.registry.get_champion(segment_value)
            candidate_record = None

            if champion is None and self.batch_cfg.get("bootstrap_if_missing_champion", True):
                candidate_record = self.develop(segment=segment_value)
                summary["steps"]["bootstrap_develop"] = {
                    "model_version": candidate_record["model_version"],
                }
                if self.batch_cfg.get("tune_weights_enabled", True):
                    summary["steps"]["tune_weights"] = self.tune_weights(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                        apply=True,
                    )
                if self.batch_cfg.get("evaluate_outcomes_enabled", True):
                    summary["steps"]["evaluate_outcomes"] = self.evaluate_outcomes(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                    )
                if self.batch_cfg.get("bootstrap_promote", True):
                    summary["steps"]["promote"] = self.promote(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                    )
                if self.batch_cfg.get("score_live_enabled", True):
                    summary["steps"]["score_live"] = self.score_live(segment=segment_value)
            else:
                if self.batch_cfg.get("score_live_enabled", True):
                    summary["steps"]["score_live"] = self.score_live(segment=segment_value)

                if self.batch_cfg.get("refresh_candidate_enabled", True):
                    candidate_run_type = str(self.batch_cfg.get("candidate_run_type", "retrain")).lower()
                    if candidate_run_type == "develop":
                        candidate_record = self.develop(segment=segment_value)
                    else:
                        candidate_record = self.retrain(segment=segment_value)
                    summary["steps"]["candidate_refresh"] = {
                        "run_type": candidate_run_type,
                        "model_version": candidate_record["model_version"],
                    }

                if candidate_record is not None and self.batch_cfg.get("tune_weights_enabled", True):
                    summary["steps"]["tune_weights"] = self.tune_weights(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                        apply=True,
                    )

                if candidate_record is not None and self.batch_cfg.get("evaluate_outcomes_enabled", True):
                    summary["steps"]["evaluate_outcomes"] = self.evaluate_outcomes(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                    )

                comparison = None
                if candidate_record is not None and self.batch_cfg.get("compare_enabled", True):
                    comparison = self.compare(
                        segment=segment_value,
                        challenger_version=candidate_record["model_version"],
                    )
                    summary["steps"]["compare"] = comparison

                if (
                    candidate_record is not None
                    and comparison is not None
                    and self.batch_cfg.get("promote_if_recommended", False)
                    and comparison["recommendation"]["winner"] == candidate_record["model_version"]
                ):
                    summary["steps"]["promote"] = self.promote(
                        segment=segment_value,
                        model_version=candidate_record["model_version"],
                    )

            if self.batch_cfg.get("cleanup_after_run", False):
                summary["steps"]["cleanup"] = self.cleanup()

            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def score_live(self, segment: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("score-live", segment_value, self.config)
        try:
            champion = self.registry.get_champion(segment_value)
            if champion is None:
                raise ValueError(f"No champion model found for segment '{segment_value}'.")

            model = self._load_model(Path(champion["artifact_path"]))
            source_name = self.live_scoring_cfg.get("source_name", "live_scoring")
            snapshot_cfg = self.live_scoring_cfg.get("snapshot", {})
            selector = snapshot_cfg.get("selector", "latest")
            explicit_date = snapshot_cfg.get("explicit_date")

            frame = self.source_loader.load_frame(
                source_name,
                latest_snapshot=selector == "latest",
                snapshot_date=explicit_date if explicit_date else None,
                segment_column=self.development_cfg.get("segment_column"),
                segment_value=None if segment_value == "ALL" else segment_value,
            )
            if frame.empty:
                raise ValueError("No rows returned for live scoring.")
            frame = self.data_loader.validate_data(frame)

            scorer = self._build_scorer(model, champion, run.run_id, segment_value)
            results = scorer.score(frame)
            shadow_results = self._score_shadow_if_enabled(champion, frame, run.run_id, segment_value)
            if shadow_results is not None:
                results = results.merge(
                    shadow_results,
                    on=[self.id_column, self.time_column],
                    how="left",
                )
                results["score_delta"] = (results["anomaly_score"] - results["raw_shadow_score"]).round(2)
            effective_snapshot = pd.to_datetime(frame[self.time_column]).max()

            monitoring_payload = {
                "input": self.monitoring.summarize_input(frame, self.features),
                "scores": self.monitoring.summarize_scores(results),
            }
            monitoring_path = run.run_dir / "monitoring.json"
            self.monitoring.write_json(monitoring_path, monitoring_payload)

            output_summary = self.output_writer.write(
                results,
                effective_snapshot,
                run_id=run.run_id,
                segment=segment_value,
            )
            summary = {
                "segment": segment_value,
                "snapshot_date": effective_snapshot.date().isoformat(),
                "rows": int(len(results)),
                "model_version": champion["model_version"],
                "calibration_version": champion.get("calibration", {}).get("version"),
                "weight_version": champion.get("weighting", {}).get("weight_version"),
                "output": output_summary,
                "monitoring_path": str(monitoring_path),
            }
            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def tune_weights(
        self,
        segment: Optional[str] = None,
        model_version: Optional[str] = None,
        apply: Optional[bool] = None,
    ):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("tune-weights", segment_value, self.config)
        try:
            model_record = self._resolve_model_record(segment_value, model_version, prefer_candidate=True)
            model = self._load_model(Path(model_record["artifact_path"]))
            calibration_artifact = self._load_calibration_artifact(model_record)
            frames, _ = self._load_development_frames(segment_value)

            tuning_window = self.weight_cfg.get("training_window", "dev")
            validation_window = self.weight_cfg.get("validation_window", "oot")
            tuning_frame = frames.get(tuning_window)
            validation_frame = frames.get(validation_window)
            if tuning_frame is None or tuning_frame.empty:
                raise ValueError(f"Weight tuning window '{tuning_window}' is empty.")
            if validation_frame is None or validation_frame.empty:
                raise ValueError(f"Weight validation window '{validation_window}' is empty.")

            source_name = self.weight_cfg.get("source_name", "outcomes")
            target_column = self.weight_cfg.get("target_column", "label_30dpd_8w")
            monitoring_columns = list(self.weight_cfg.get("monitoring_columns", []))
            label_frame = self._load_label_frame(
                source_name,
                segment_value,
                start_date=min(
                    pd.to_datetime(tuning_frame[self.time_column]).min(),
                    pd.to_datetime(validation_frame[self.time_column]).min(),
                ),
                end_date=max(
                    pd.to_datetime(tuning_frame[self.time_column]).max(),
                    pd.to_datetime(validation_frame[self.time_column]).max(),
                ),
            )

            tuning_dataset = self._build_weight_dataset(
                model,
                tuning_frame,
                label_frame,
                calibration_artifact=calibration_artifact,
                model_record=model_record,
            )
            validation_dataset = self._build_weight_dataset(
                model,
                validation_frame,
                label_frame,
                calibration_artifact=calibration_artifact,
                model_record=model_record,
            )

            min_positive_rows = int(self.weight_cfg.get("min_positive_rows", 10))
            if tuning_dataset[target_column].sum() < min_positive_rows:
                raise ValueError(
                    f"Target '{target_column}' has fewer than {min_positive_rows} positive rows in tuning window."
                )

            optimizer = WeightOptimizer(self.config)
            artifact = optimizer.optimize(
                tuning_dataset,
                validation_dataset,
                target_column=target_column,
                monitoring_columns=monitoring_columns,
                model_version=model_record["model_version"],
                segment=segment_value,
            )

            artifact_path = run.artifact_dir / "weights.json"
            WeightOptimizer.save(artifact_path, artifact)
            should_apply = bool(self.weight_cfg.get("auto_apply", False)) if apply is None else bool(apply)
            weighting_record = {
                "status": "optimized" if should_apply else "candidate",
                "source": "target_optimization",
                "artifact_path": str(artifact_path),
                "weight_version": artifact["weight_version"],
                "target_column": artifact["target_column"],
                "monitoring_columns": artifact["monitoring_columns"],
                "objective": artifact["objective"],
                "weights": artifact["weights"],
                "tuning_metrics": artifact["tuning_metrics"],
                "validation_metrics": artifact["validation_metrics"],
            }
            update_payload = {"weighting": weighting_record} if should_apply else {"pending_weighting": weighting_record}
            self.registry.update_model(model_record["model_version"], update_payload)

            summary = {
                "segment": segment_value,
                "model_version": model_record["model_version"],
                "weight_version": artifact["weight_version"],
                "applied": should_apply,
                "artifact_path": str(artifact_path),
                "validation_metrics": artifact["validation_metrics"],
            }
            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def evaluate_outcomes(self, segment: Optional[str] = None, model_version: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run("evaluate-outcomes", segment_value, self.config)
        try:
            model_record = self._resolve_model_record(segment_value, model_version, prefer_candidate=False)
            model = self._load_model(Path(model_record["artifact_path"]))
            calibration_artifact = self._load_calibration_artifact(model_record)
            weights = self._resolve_active_weights(model_record)
            frames, _ = self._load_development_frames(segment_value)

            evaluation_windows = list(self.weight_cfg.get("evaluation_windows", ["dev", "oot"]))
            selected = [frames[name] for name in evaluation_windows if name in frames and not frames[name].empty]
            if not selected:
                raise ValueError("No development windows available for outcome evaluation.")
            evaluation_frame = pd.concat(selected, ignore_index=True)

            source_name = self.weight_cfg.get("source_name", "outcomes")
            target_column = self.weight_cfg.get("target_column", "label_30dpd_8w")
            monitoring_columns = list(self.weight_cfg.get("monitoring_columns", []))
            label_frame = self._load_label_frame(
                source_name,
                segment_value,
                start_date=pd.to_datetime(evaluation_frame[self.time_column]).min(),
                end_date=pd.to_datetime(evaluation_frame[self.time_column]).max(),
            )
            dataset = self._build_weight_dataset(
                model,
                evaluation_frame,
                label_frame,
                calibration_artifact=calibration_artifact,
                model_record=model_record,
            )

            evaluator = WeightOptimizer(self.config)
            metrics = evaluator.evaluate(
                dataset,
                weights,
                target_column=target_column,
                monitoring_columns=monitoring_columns,
            )
            artifact = {
                "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model_version": model_record["model_version"],
                "weight_version": model_record.get("weighting", {}).get("weight_version"),
                "target_column": target_column,
                "monitoring_columns": monitoring_columns,
                "windows": evaluation_windows,
                "metrics": metrics,
            }
            artifact_path = run.artifact_dir / "evaluation.json"
            with open(artifact_path, "w", encoding="utf-8") as handle:
                json.dump(artifact, handle, indent=2, ensure_ascii=False)

            self.registry.update_model(
                model_record["model_version"],
                {
                    "evaluation": {
                        "artifact_path": str(artifact_path),
                        "target_column": target_column,
                        "monitoring_columns": monitoring_columns,
                        "windows": evaluation_windows,
                        "metrics": metrics,
                    }
                },
            )
            summary = {
                "segment": segment_value,
                "model_version": model_record["model_version"],
                "evaluation_path": str(artifact_path),
                "metrics": metrics,
            }
            self.registry.finish_run(run, "completed", summary)
            return summary
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def _run_training_lifecycle(self, run_type: str, segment: Optional[str] = None):
        segment_value = self._resolve_segment(segment)
        run = self.registry.start_run(run_type, segment_value, self.config)
        try:
            frames, windows = self._load_development_frames(segment_value)
            train_df = frames.get("train")
            if train_df is None or train_df.empty:
                raise ValueError("Train window is empty; cannot fit model.")

            model = AnomalyModels(self.config)
            model.fit(train_df[self.features])

            model_version = self._build_model_version(segment_value, run_type)
            model_path = run.artifact_dir / "model.pkl"
            self._save_model(model, model_path)
            shadow_record = self._fit_shadow_branch(
                train_df=train_df,
                calibration_frame=frames.get(self.calibration_cfg.get("source_window", "calibration")),
                model_version=model_version,
                segment_value=segment_value,
                run=run,
            )

            calibration_record = self._fit_calibration(
                model,
                model_version=model_version,
                segment_value=segment_value,
                frame=frames.get(self.calibration_cfg.get("source_window", "calibration")),
                run=run,
            )
            calibration_artifact = self._load_calibration_artifact_from_record(calibration_record)

            stability = {}
            monitoring_payload = {"input": {}, "scores": {}}
            default_weights = get_ensemble_weights(self.config)
            for window_name, frame in frames.items():
                monitoring_payload["input"][window_name] = self.monitoring.summarize_input(frame, self.features)
                if frame.empty:
                    continue
                scorer = self._build_scorer(
                    model,
                    {
                        "model_version": model_version,
                        "calibration": calibration_record,
                        "weighting": {"weights": default_weights, "weight_version": None},
                    },
                    run.run_id,
                    segment_value,
                )
                monitoring_payload["scores"][window_name] = self.monitoring.summarize_scores(scorer.score(frame))
                if window_name != "train":
                    stability[window_name] = self._evaluate_stability_window(
                        model,
                        train_df,
                        frame,
                        calibration_artifact=calibration_artifact,
                        weights=default_weights,
                    )

            stability_path = run.artifact_dir / "stability.json"
            with open(stability_path, "w", encoding="utf-8") as handle:
                json.dump(stability, handle, indent=2, ensure_ascii=False)
            monitoring_path = run.run_dir / "monitoring.json"
            self.monitoring.write_json(monitoring_path, monitoring_payload)

            record = {
                "model_version": model_version,
                "segment": segment_value,
                "status": "candidate",
                "run_id": run.run_id,
                "created_at": run.created_at,
                "artifact_path": str(model_path),
                "artifact_dir": str(run.artifact_dir),
                "stability_path": str(stability_path),
                "monitoring_path": str(monitoring_path),
                "windows": {name: summarize_window(frame, self.time_column) for name, frame in frames.items()},
                "calibration": calibration_record,
                "shadow_scoring": shadow_record,
                "preprocessing": model.preprocessing_summary(),
                "weighting": {
                    "status": "config_default",
                    "source": "config",
                    "weight_version": None,
                    "weights": default_weights,
                },
            }
            self.registry.register_model(record)

            summary = {
                "model_version": model_version,
                "segment": segment_value,
                "artifact_path": str(model_path),
                "stability_path": str(stability_path),
                "monitoring_path": str(monitoring_path),
                "windows": {name: self._window_boundaries(spec) for name, spec in windows.items()},
                "calibration_status": calibration_record["status"],
                "shadow_status": shadow_record["status"],
                "preprocessing": model.preprocessing_summary(),
            }
            self.registry.finish_run(run, "completed", summary)
            return record
        except Exception as exc:
            self.registry.finish_run(run, "failed", {"reason": str(exc)})
            raise

    def _resolve_segment(self, segment: Optional[str]) -> str:
        if segment:
            return segment
        configured = self.development_cfg.get("segment_value")
        if configured:
            return configured
        return "ALL"

    def _build_model_version(self, segment: str, run_type: str) -> str:
        prefix = self.retraining_cfg.get("candidate_name_prefix", "challenger")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        if run_type == "develop":
            prefix = "develop"
        return f"{segment}-{prefix}-{timestamp}"

    def _save_model(self, model: AnomalyModels, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(model, handle)

    def _load_model(self, path: Path) -> AnomalyModels:
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _load_development_frames(self, segment_value: str):
        source_name = self.development_cfg.get("source_name", "development")
        segment_column = self.development_cfg.get("segment_column")
        snapshots = self.source_loader.list_snapshots(
            source_name,
            segment_column=segment_column,
            segment_value=None if segment_value == "ALL" else segment_value,
        )
        windows = WindowResolver(self.config).resolve(snapshots)

        frames = {}
        for offset, (name, spec) in enumerate(windows.items()):
            frame = self.source_loader.load_frame(
                source_name,
                start_date=spec.start,
                end_date=spec.end,
                segment_column=segment_column,
                segment_value=None if segment_value == "ALL" else segment_value,
            )
            frame = self._apply_sampling(frame, per_window_seed_offset=offset)
            if not frame.empty:
                frame = self.data_loader.validate_data(frame)
            frames[name] = frame
        return frames, windows

    def _load_label_frame(self, source_name: str, segment_value: str, *, start_date, end_date):
        frame = self.source_loader.load_frame(
            source_name,
            start_date=start_date,
            end_date=end_date,
        )
        if frame.empty:
            raise ValueError(f"Label source '{source_name}' returned no rows for the requested date range.")
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if self.id_column not in frame.columns or self.time_column not in frame.columns:
            raise ValueError(
                f"Label source '{source_name}' must contain '{self.id_column}' and '{self.time_column}'."
            )
        frame[self.time_column] = pd.to_datetime(frame[self.time_column])
        return frame

    def _fit_calibration(self, model, *, model_version: str, segment_value: str, frame, run):
        return self._fit_calibration_for_config(
            self.config,
            model,
            model_version=model_version,
            segment_value=segment_value,
            frame=frame,
            run=run,
            file_name="calibration.json",
        )

    def _fit_calibration_for_config(self, calibrator_config, model, *, model_version: str, segment_value: str, frame, run, file_name: str):
        if not bool(self.calibration_cfg.get("enabled", False)):
            return {"enabled": False, "status": "disabled", "version": None}
        if frame is None or frame.empty:
            return {"enabled": True, "status": "missing_window", "version": None}
        min_rows = int(self.calibration_cfg.get("min_rows", 500))
        if len(frame) < min_rows:
            return {
                "enabled": True,
                "status": "insufficient_rows",
                "version": None,
                "rows": int(len(frame)),
                "required_rows": min_rows,
            }

        X = model.transform(frame[self.features])
        calibrator = ScoreCalibrator(calibrator_config)
        artifact = calibrator.fit(
            {
                "ae_raw": model.raw_ae_scores(X),
                "if_raw": model.raw_if_scores(X),
                "md_raw": model.raw_md_scores(X),
            },
            model_version=model_version,
            segment=segment_value,
            window=summarize_window(frame, self.time_column),
        )
        path = run.artifact_dir / file_name
        ScoreCalibrator.save(path, artifact)
        return {
            "enabled": True,
            "status": "fitted",
            "version": artifact.version,
            "artifact_path": str(path),
            "window": artifact.window,
            "rows": artifact.n_rows,
        }

    def _build_scorer(self, model, model_record: dict, run_id: str, segment_value: str):
        weighting = model_record.get("weighting", {})
        metadata = {
            "run_id": run_id,
            "segment": segment_value,
            "model_version": model_record.get("model_version"),
            "calibration_version": model_record.get("calibration", {}).get("version"),
            "weight_version": weighting.get("weight_version"),
        }
        return AnomalyScorer(
            self.config,
            model,
            calibration_artifact=self._load_calibration_artifact(model_record),
            weights=weighting.get("weights", get_ensemble_weights(self.config)),
            metadata=metadata,
        )

    def _build_weight_dataset(self, model, feature_frame, label_frame, *, calibration_artifact, model_record):
        scorer = self._build_scorer(model, model_record, run_id="offline", segment_value=model_record["segment"])
        scores = scorer.score(feature_frame)
        merged = scores.merge(
            label_frame,
            on=[self.id_column, self.time_column],
            how="inner",
            suffixes=("", "_label"),
        )
        if merged.empty:
            raise ValueError("No overlap found between score frame and label frame.")
        return merged

    def _resolve_model_record(self, segment_value: str, model_version: Optional[str], *, prefer_candidate: bool):
        if model_version:
            return self.registry.get_model(model_version)
        if prefer_candidate:
            candidate = self.registry.get_latest_candidate(segment_value)
            if candidate is not None:
                return candidate
        champion = self.registry.get_champion(segment_value)
        if champion is not None:
            return champion
        raise ValueError(f"No model available for segment '{segment_value}'.")

    def _load_calibration_artifact(self, model_record: dict):
        return self._load_calibration_artifact_from_record(model_record.get("calibration", {}))

    @staticmethod
    def _load_calibration_artifact_from_record(calibration_record: dict):
        if calibration_record.get("status") != "fitted":
            return None
        artifact_path = calibration_record.get("artifact_path")
        if not artifact_path:
            return None
        return ScoreCalibrator.load(Path(artifact_path))

    def _resolve_active_weights(self, model_record: dict):
        return model_record.get("weighting", {}).get("weights", get_ensemble_weights(self.config))

    def _apply_sampling(self, frame: pd.DataFrame, *, per_window_seed_offset: int = 0) -> pd.DataFrame:
        sampling_cfg = self.development_cfg.get("sampling", {})
        if not sampling_cfg.get("enabled", False) or frame.empty:
            return frame.reset_index(drop=True)

        seed = int(sampling_cfg.get("random_seed", 42)) + per_window_seed_offset
        rng = np.random.default_rng(seed)
        result = frame.copy()
        max_rows_per_snapshot = sampling_cfg.get("max_rows_per_snapshot")
        max_rows = sampling_cfg.get("max_rows")

        if max_rows_per_snapshot:
            pieces = []
            for _, group in result.groupby(self.time_column):
                take = min(len(group), int(max_rows_per_snapshot))
                idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
                pieces.append(group.loc[idx])
            result = pd.concat(pieces, ignore_index=True) if pieces else result.iloc[0:0]

        if max_rows and len(result) > int(max_rows):
            idx = rng.choice(result.index.to_numpy(), size=int(max_rows), replace=False)
            result = result.loc[idx]

        return result.sort_values([self.time_column, self.id_column]).reset_index(drop=True)

    def _evaluate_stability_window(self, model, train_df, reference_df, *, calibration_artifact, weights):
        train_scores = self._compute_model_scores(
            model,
            train_df,
            calibration_artifact=calibration_artifact,
            weights=weights,
        )
        reference_scores = self._compute_model_scores(
            model,
            reference_df,
            calibration_artifact=calibration_artifact,
            weights=weights,
        )

        report = {
            "train_rows": int(len(train_df)),
            "reference_rows": int(len(reference_df)),
            "metrics": {},
            "ensemble_alert_share": {
                "train": self._band_share(train_scores["ensemble_score"]),
                "reference": self._band_share(reference_scores["ensemble_score"]),
            },
            "weights": weights,
        }

        for metric_name in train_scores:
            train_values = train_scores[metric_name]
            ref_values = reference_scores[metric_name]
            ks_stat, ks_pval = ks_2samp(train_values, ref_values)
            train_summary = self._summarize_distribution(train_values)
            ref_summary = self._summarize_distribution(ref_values)
            mean_ratio = None
            if abs(train_summary["mean"]) > 1e-12:
                mean_ratio = round(float(ref_summary["mean"] / train_summary["mean"]), 4)
            report["metrics"][metric_name] = {
                "train": train_summary,
                "reference": ref_summary,
                "mean_ratio": mean_ratio,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
            }
        return report

    def _compute_model_scores(self, model, frame, *, calibration_artifact, weights):
        X = model.transform(frame[self.features])
        raw_scores = {
            "ae_raw": model.raw_ae_scores(X),
            "if_raw": model.raw_if_scores(X),
            "md_raw": model.raw_md_scores(X),
        }
        if calibration_artifact is not None:
            calibrated = ScoreCalibrator(self.config).apply(raw_scores, calibration_artifact)
        else:
            calibrated = {
                "ae_cal": model.ae_scores(X),
                "if_cal": model.if_scores(X),
                "md_cal": model.md_scores(X),
            }

        ensemble_score = np.clip(
            weights["autoencoder"] * calibrated["ae_cal"]
            + weights["isolation_forest"] * calibrated["if_cal"]
            + weights["mahalanobis"] * calibrated["md_cal"],
            self.config.get("scoring", {}).get("score_min", 0),
            self.config.get("scoring", {}).get("score_max", 100),
        )
        return {**raw_scores, **calibrated, "ensemble_score": ensemble_score}

    def _band_share(self, scores):
        bands = pd.Series(self._score_to_band(scores))
        total = len(bands)
        return {
            band: round(float((bands == band).sum() / total), 4)
            for band in ("NORMAL", "SARI", "TURUNCU", "KIRMIZI")
        }

    def _score_to_band(self, scores):
        bands = get_alert_bands(self.config)
        result = []
        for score in scores:
            assigned = "NORMAL"
            for band_name, (lower, upper) in bands.items():
                if lower <= score < upper or (band_name == "KIRMIZI" and score >= lower):
                    assigned = band_name
            result.append(assigned)
        return result

    @staticmethod
    def _summarize_distribution(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
        }

    def _build_comparison(self, champion: dict, challenger: dict):
        champion_stability = self._load_json(Path(champion["stability_path"]))
        challenger_stability = self._load_json(Path(challenger["stability_path"]))
        metric_path = self.retraining_cfg.get("promotion_metric", "stability.oot.metrics.ensemble_score.ks_pvalue")

        champion_payload = {
            "stability": champion_stability,
            "calibration": champion.get("calibration", {}),
            "weighting": champion.get("weighting", {}),
            "evaluation": champion.get("evaluation", {}),
        }
        challenger_payload = {
            "stability": challenger_stability,
            "calibration": challenger.get("calibration", {}),
            "weighting": challenger.get("weighting", {}),
            "evaluation": challenger.get("evaluation", {}),
        }
        champion_value = self._metric_value(champion_payload, metric_path)
        challenger_value = self._metric_value(challenger_payload, metric_path)
        winner = champion["model_version"] if champion_value is None or challenger_value is None else (
            challenger["model_version"] if challenger_value >= champion_value else champion["model_version"]
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metric_path": metric_path,
            "champion": {"model_version": champion["model_version"], "metric_value": champion_value, "payload": champion_payload},
            "challenger": {"model_version": challenger["model_version"], "metric_value": challenger_value, "payload": challenger_payload},
            "recommendation": {"winner": winner, "reason": "promotion_metric comparison"},
        }

    @staticmethod
    def _metric_value(payload: dict, metric_path: str):
        current = payload
        for part in metric_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _load_json(path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _window_boundaries(spec):
        return {
            "start": pd.Timestamp(spec.start).date().isoformat(),
            "end": pd.Timestamp(spec.end).date().isoformat(),
        }

    def _build_config_variant(self, *, preprocessing_enabled: bool) -> dict:
        variant = copy.deepcopy(self.config)
        variant.setdefault("preprocessing", {})
        variant["preprocessing"]["enabled"] = preprocessing_enabled
        if not preprocessing_enabled:
            variant["preprocessing"]["missing"] = {"strategy": "zero"}
            variant["preprocessing"]["winsorization"] = {"enabled": False}
            variant["preprocessing"]["log1p"] = {"enabled": False, "features": []}
            variant["preprocessing"]["hard_bounds"] = {"enabled": False, "rules": []}
            variant["preprocessing"]["scaler"] = {"type": "standard"}
        return variant

    def _build_shadow_config(self) -> dict:
        variant = copy.deepcopy(self.config)
        variant.setdefault("preprocessing", {})
        variant["preprocessing"]["enabled"] = True
        if not bool(self.shadow_cfg.get("use_shared_hard_bounds", True)):
            variant["preprocessing"]["hard_bounds"] = {"enabled": False, "rules": []}
        variant["preprocessing"]["winsorization"] = {
            "enabled": bool(self.shadow_cfg.get("winsorization_enabled", False))
        }
        variant["preprocessing"]["log1p"] = {
            "enabled": bool(self.shadow_cfg.get("log1p_enabled", False)),
            "features": self.config.get("preprocessing", {}).get("log1p", {}).get("features", []),
        }
        variant["preprocessing"]["scaler"] = {
            "type": self.shadow_cfg.get("scaler_type", "standard")
        }
        return variant

    def _load_live_frame(self, segment_value: str) -> pd.DataFrame:
        source_name = self.live_scoring_cfg.get("source_name", "input_features")
        snapshot_cfg = self.live_scoring_cfg.get("snapshot", {})
        selector = snapshot_cfg.get("selector", "latest")
        explicit_date = snapshot_cfg.get("explicit_date")
        frame = self.source_loader.load_frame(
            source_name,
            latest_snapshot=selector == "latest",
            snapshot_date=explicit_date if explicit_date else None,
            segment_column=self.development_cfg.get("segment_column"),
            segment_value=None if segment_value == "ALL" else segment_value,
        )
        if frame.empty:
            raise ValueError("No rows returned for live scoring comparison.")
        return self.data_loader.validate_data(frame)

    def _fit_shadow_branch(self, *, train_df, calibration_frame, model_version: str, segment_value: str, run):
        if not bool(self.shadow_cfg.get("enabled", False)):
            return {"enabled": False, "status": "disabled", "version": None}

        shadow_config = self._build_shadow_config()
        shadow_model = AnomalyModels(shadow_config)
        shadow_model.fit(train_df[self.features])
        shadow_model_path = run.artifact_dir / "shadow_model.pkl"
        self._save_model(shadow_model, shadow_model_path)

        calibration_record = self._fit_calibration_for_config(
            shadow_config,
            shadow_model,
            model_version=f"{model_version}-shadow",
            segment_value=segment_value,
            frame=calibration_frame,
            run=run,
            file_name="shadow_calibration.json",
        )
        return {
            "enabled": True,
            "status": "fitted",
            "artifact_path": str(shadow_model_path),
            "model_version": f"{model_version}-shadow",
            "mode": self.shadow_cfg.get("mode", "raw_shadow"),
            "preprocessing": shadow_model.preprocessing_summary(),
            "calibration": calibration_record,
        }

    def _evaluate_preprocessing_variant(
        self,
        variant_config: dict,
        *,
        segment_value: str,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame | None,
        calibration_df: pd.DataFrame | None,
        oot_df: pd.DataFrame,
        live_df: pd.DataFrame,
        label_frame: pd.DataFrame,
        weights: dict,
        tag: str,
    ) -> dict:
        model = AnomalyModels(variant_config)
        model.fit(train_df[self.features])

        calibration_artifact = None
        if bool(variant_config.get("calibration", {}).get("enabled", False)) and calibration_df is not None and not calibration_df.empty:
            X_cal = model.transform(calibration_df[self.features])
            calibration_artifact = ScoreCalibrator(variant_config).fit(
                {
                    "ae_raw": model.raw_ae_scores(X_cal),
                    "if_raw": model.raw_if_scores(X_cal),
                    "md_raw": model.raw_md_scores(X_cal),
                },
                model_version=f"{tag}-offline",
                segment=segment_value,
                window=summarize_window(calibration_df, self.time_column),
            ).to_dict()

        scorer = AnomalyScorer(
            variant_config,
            model,
            calibration_artifact=calibration_artifact,
            weights=weights,
            metadata={
                "run_id": f"compare-preprocessing-{tag}",
                "segment": segment_value,
                "model_version": f"{tag}-offline",
                "calibration_version": calibration_artifact["version"] if calibration_artifact else None,
                "weight_version": "fixed-config",
            },
        )
        stability = self._evaluate_stability_window(
            model,
            train_df,
            oot_df,
            calibration_artifact=calibration_artifact,
            weights=weights,
        )

        evaluation_frame = pd.concat(
            [frame for frame in (oot_df, ) if frame is not None and not frame.empty],
            ignore_index=True,
        )
        scored_eval = scorer.score(evaluation_frame)
        scored_live = scorer.score(live_df)
        eval_dataset = scored_eval.merge(
            label_frame,
            on=[self.id_column, self.time_column],
            how="inner",
            suffixes=("", "_label"),
        )
        if eval_dataset.empty:
            raise ValueError("No overlap found between evaluated scores and label frame for preprocessing comparison.")

        outcome_metrics = WeightOptimizer(variant_config).evaluate(
            eval_dataset,
            weights,
            target_column=self.weight_cfg.get("target_column", "label_30dpd_8w"),
            monitoring_columns=list(self.weight_cfg.get("monitoring_columns", [])),
        )
        tuned_artifact = None
        if dev_df is not None and not dev_df.empty:
            scored_dev = scorer.score(dev_df)
            tune_dataset = scored_dev.merge(
                label_frame,
                on=[self.id_column, self.time_column],
                how="inner",
                suffixes=("", "_label"),
            )
            if not tune_dataset.empty:
                optimizer = WeightOptimizer(variant_config)
                tuned_artifact = optimizer.optimize(
                    tune_dataset,
                    eval_dataset,
                    target_column=self.weight_cfg.get("target_column", "label_30dpd_8w"),
                    monitoring_columns=list(self.weight_cfg.get("monitoring_columns", [])),
                    model_version=f"{tag}-offline",
                    segment=segment_value,
                )
        summary = {
            "preprocessing": model.preprocessing_summary(),
            "stability": stability,
            "outcomes": outcome_metrics,
            "tuned_weights": tuned_artifact["weights"] if tuned_artifact else weights,
            "tuned_outcomes": tuned_artifact["validation_metrics"] if tuned_artifact else outcome_metrics,
            "live_scores": self.monitoring.summarize_scores(scored_live),
            "train_inspection": model.preprocessor.inspect_frame(train_df[self.features]),
            "oot_inspection": model.preprocessor.inspect_frame(oot_df[self.features]),
            "live_inspection": model.preprocessor.inspect_frame(live_df[self.features]),
        }
        return {
            "summary": summary,
            "live_scores": scored_live,
        }

    def _score_shadow_if_enabled(self, model_record: dict, frame: pd.DataFrame, run_id: str, segment_value: str):
        shadow_record = model_record.get("shadow_scoring", {})
        if shadow_record.get("status") != "fitted":
            return None
        shadow_model_path = shadow_record.get("artifact_path")
        if not shadow_model_path:
            return None
        shadow_model = self._load_model(Path(shadow_model_path))
        shadow_config = self._build_shadow_config()
        shadow_scorer = AnomalyScorer(
            shadow_config,
            shadow_model,
            calibration_artifact=self._load_calibration_artifact_from_record(
                shadow_record.get("calibration", {})
            ),
            weights=self._resolve_active_weights(model_record),
            metadata={
                "run_id": run_id,
                "segment": segment_value,
                "model_version": shadow_record.get("model_version"),
                "calibration_version": shadow_record.get("calibration", {}).get("version"),
                "weight_version": model_record.get("weighting", {}).get("weight_version"),
            },
        )
        scored = shadow_scorer.score(frame)
        renamed = scored[
            [
                self.id_column,
                self.time_column,
                "anomaly_score",
                "alert_band",
                "ae_score",
                "if_score",
                "md_score",
            ]
        ].rename(
            columns={
                "anomaly_score": "raw_shadow_score",
                "alert_band": "raw_shadow_alert_band",
                "ae_score": "raw_shadow_ae_score",
                "if_score": "raw_shadow_if_score",
                "md_score": "raw_shadow_md_score",
            }
        )
        return renamed

    def _build_preprocessing_sample_comparison(self, baseline_scores: pd.DataFrame, robust_scores: pd.DataFrame) -> dict:
        merged = baseline_scores.merge(
            robust_scores,
            on=[self.id_column, self.time_column],
            suffixes=("_old", "_new"),
        )
        candidates = merged[
            (merged["alert_band_new"] == "KIRMIZI")
            & (merged["anomaly_score_new"] < 100)
        ].copy()
        both_red = candidates[candidates["alert_band_old"] == "KIRMIZI"]
        if not both_red.empty:
            candidates = both_red
        if candidates.empty:
            candidates = merged[
                (merged["alert_band_new"] == "KIRMIZI")
                | (merged["alert_band_old"] == "KIRMIZI")
            ].copy()
        if candidates.empty:
            raise ValueError("No red-band comparison sample found in live scores.")
        candidates["selection_rank"] = candidates["anomaly_score_new"].rank(method="first", ascending=False)
        row = candidates.sort_values(["anomaly_score_new", "anomaly_score_old"], ascending=False).iloc[0]
        return {
            "customer_id": row[self.id_column],
            "snapshot_date": pd.Timestamp(row[self.time_column]).date().isoformat(),
            "old": self._sample_payload(row, suffix="_old"),
            "new": self._sample_payload(row, suffix="_new"),
        }

    @staticmethod
    def _sample_payload(row: pd.Series, *, suffix: str) -> dict:
        return {
            "anomaly_score": round(float(row[f"anomaly_score{suffix}"]), 2),
            "alert_band": row[f"alert_band{suffix}"],
            "ae_score": round(float(row[f"ae_score{suffix}"]), 2),
            "if_score": round(float(row[f"if_score{suffix}"]), 2),
            "md_score": round(float(row[f"md_score{suffix}"]), 2),
            "reason_1": row.get(f"reason_1{suffix}"),
            "reason_2": row.get(f"reason_2{suffix}"),
            "reason_3": row.get(f"reason_3{suffix}"),
        }

    @staticmethod
    def _render_preprocessing_comparison_markdown(payload: dict) -> str:
        return "\n".join(
            [
                "# Preprocessing Comparison",
                "",
                f"- Segment: `{payload['segment']}`",
                f"- Fixed weights: `{payload['fixed_weights']}`",
                "",
                "## Primary Metrics",
                "",
                f"- Baseline precision@top%: `{payload['baseline']['outcomes']['primary']['precision_at_top_percent']}`",
                f"- Robust precision@top%: `{payload['robust']['outcomes']['primary']['precision_at_top_percent']}`",
                f"- Baseline lift@top%: `{payload['baseline']['outcomes']['primary']['lift_at_top_percent']}`",
                f"- Robust lift@top%: `{payload['robust']['outcomes']['primary']['lift_at_top_percent']}`",
                f"- Baseline tuned precision@top%: `{payload['baseline']['tuned_outcomes']['primary']['precision_at_top_percent']}`",
                f"- Robust tuned precision@top%: `{payload['robust']['tuned_outcomes']['primary']['precision_at_top_percent']}`",
                f"- OOT ensemble KS delta: `{payload['delta']['oot_ensemble_ks_delta']}`",
                "",
                "## Sample Customer",
                "",
                f"- Customer: `{payload['sample_customer']['customer_id']}`",
                f"- Snapshot: `{payload['sample_customer']['snapshot_date']}`",
                f"- Old score/band: `{payload['sample_customer']['old']['anomaly_score']}` / `{payload['sample_customer']['old']['alert_band']}`",
                f"- New score/band: `{payload['sample_customer']['new']['anomaly_score']}` / `{payload['sample_customer']['new']['alert_band']}`",
            ]
        )
