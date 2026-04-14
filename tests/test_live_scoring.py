import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile

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


if __name__ == "__main__":
    unittest.main()
