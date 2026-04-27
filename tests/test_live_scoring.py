import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import contextmanager

import pandas as pd
import yaml

from engine.lifecycle import LifecycleManager


class LiveScoringSelectionTests(unittest.TestCase):
    def _make_manager(self) -> LifecycleManager:
        manager = LifecycleManager.__new__(LifecycleManager)
        manager.live_scoring_cfg = {
            "source_name": "input_features",
            "snapshot": {
                "selector": "today",
                "explicit_date": None,
                "start_date": None,
                "end_date": None,
            },
        }
        manager.development_cfg = {"segment_column": "segment"}
        manager.source_loader = Mock()
        manager.data_loader = Mock()
        manager.features = []
        return manager

    def test_today_selector_reads_current_day_only(self):
        manager = self._make_manager()
        today_frame = pd.DataFrame({"customer_id": ["C1"], "snapshot_date": [pd.Timestamp("2026-04-14")]})
        manager.source_loader.load_frame.side_effect = [today_frame]

        frame = manager._load_live_source_frame("ALL")

        self.assertEqual(len(frame), 1)
        self.assertEqual(manager.source_loader.load_frame.call_count, 1)
        _, kwargs = manager.source_loader.load_frame.call_args
        self.assertTrue(kwargs["current_day"])

    def test_today_selector_returns_empty_when_current_day_is_empty(self):
        manager = self._make_manager()
        manager.source_loader.load_frame.side_effect = [pd.DataFrame()]

        frame = manager._load_live_source_frame("ALL")

        self.assertTrue(frame.empty)
        self.assertEqual(manager.source_loader.load_frame.call_count, 1)
        _, kwargs = manager.source_loader.load_frame.call_args
        self.assertTrue(kwargs["current_day"])

    def test_explicit_date_overrides_selector(self):
        manager = self._make_manager()
        manager.live_scoring_cfg["snapshot"]["explicit_date"] = "2026-04-08"
        explicit_frame = pd.DataFrame({"customer_id": ["C3"], "snapshot_date": [pd.Timestamp("2026-04-08")]})
        manager.source_loader.load_frame.side_effect = [explicit_frame]

        frame = manager._load_live_source_frame("ALL")

        self.assertEqual(len(frame), 1)
        _, kwargs = manager.source_loader.load_frame.call_args
        self.assertEqual(kwargs["snapshot_date"], "2026-04-08")
        self.assertNotIn("current_day", kwargs)

    def test_previous_month_end_selector_resolves_snapshot_date(self):
        manager = self._make_manager()
        manager.live_scoring_cfg["snapshot"]["selector"] = "previous_month_end"
        manager._previous_month_end = Mock(return_value=pd.Timestamp("2026-03-31"))
        previous_frame = pd.DataFrame({"customer_id": ["C4"], "snapshot_date": [pd.Timestamp("2026-03-31")]})
        manager.source_loader.load_frame.side_effect = [previous_frame]

        frame = manager._load_live_source_frame("ALL")

        self.assertEqual(len(frame), 1)
        _, kwargs = manager.source_loader.load_frame.call_args
        self.assertEqual(pd.Timestamp(kwargs["snapshot_date"]), pd.Timestamp("2026-03-31"))

    def test_successful_run_clears_explicit_date_from_config_file(self):
        manager = self._make_manager()
        manager.config = {
            "live_scoring": {
                "snapshot": {
                    "selector": "today",
                    "explicit_date": "2026-04-08",
                    "start_date": None,
                    "end_date": None,
                }
            }
        }
        manager.live_scoring_cfg = manager.config["live_scoring"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "pipeline_config.yaml"
            config_path.write_text(yaml.safe_dump(manager.config, sort_keys=False, allow_unicode=True), encoding="utf-8")
            manager.config_path = config_path

            manager._reset_live_explicit_date_after_success()

            persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertIsNone(manager.live_scoring_cfg["snapshot"]["explicit_date"])
            self.assertIsNone(persisted["live_scoring"]["snapshot"]["explicit_date"])

    def test_score_live_marks_empty_scope_as_skipped(self):
        manager = LifecycleManager.__new__(LifecycleManager)
        manager.config = {"pipeline": {"id_column": "customer_id", "time_column": "snapshot_date"}}
        manager.development_cfg = {"segment_value": "SEG_A"}
        manager.live_scoring_cfg = {
            "snapshot": {
                "selector": "today",
                "explicit_date": None,
                "start_date": None,
                "end_date": None,
            }
        }
        manager.last_materialization_summary = {"live_scoring": {"quality": {"derived_scope": {"status": "empty"}}}}
        manager.registry = Mock()
        run = SimpleNamespace(run_id="run-1", run_type="score-live", segment="SEG_A")
        manager.registry.start_run.return_value = run
        manager.registry.get_champion.return_value = {"artifact_path": "artifact.pkl", "model_version": "champion-v1"}
        manager._load_model = Mock(return_value=Mock())
        manager._resolve_model_feature_names = Mock(return_value=["feature_a"])
        manager._load_live_source_frame = Mock(return_value=pd.DataFrame())
        manager._resolve_segment = Mock(return_value="SEG_A")

        @contextmanager
        def fake_logging(_run):
            yield Path("runtime/runs/run-1/logs/scoring.log")

        manager._activate_run_logging = fake_logging

        result = manager.score_live(segment="SEG_A")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["rows"], 0)
        manager.registry.finish_run.assert_called_once()
        finish_args = manager.registry.finish_run.call_args.args
        self.assertEqual(finish_args[1], "skipped")


if __name__ == "__main__":
    unittest.main()
