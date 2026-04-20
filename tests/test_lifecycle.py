from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.lifecycle import LifecycleManager


class _RecordingRegistry:
    def __init__(self):
        self.calls = []

    def finish_run(self, run, status, summary=None):
        self.calls.append((run, status, summary or {}))


class LifecycleLoggingTests(unittest.TestCase):
    def test_logging_setup_failure_marks_run_failed(self):
        run = SimpleNamespace(run_type="develop", run_id="run-123", segment="SEG_A")
        manager = LifecycleManager.__new__(LifecycleManager)
        manager.config = {}
        manager.registry = _RecordingRegistry()

        with patch("engine.lifecycle.attach_run_file_logger", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                with manager._activate_run_logging(run):
                    self.fail("run body should not execute when logger setup fails")

        self.assertEqual(len(manager.registry.calls), 1)
        called_run, status, summary = manager.registry.calls[0]
        self.assertIs(called_run, run)
        self.assertEqual(status, "failed")
        self.assertEqual(summary["reason"], "Run logging setup failed: disk full")


if __name__ == "__main__":
    unittest.main()
