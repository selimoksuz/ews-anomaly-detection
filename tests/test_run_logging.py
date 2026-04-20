import tempfile
import unittest
from pathlib import Path

from engine.run_logging import get_run_log_path


class RunLoggingTests(unittest.TestCase):
    def test_legacy_meta_dir_keeps_run_logs_under_meta_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {
                "registry": {
                    "meta_dir": str(root / "meta"),
                }
            }

            path = get_run_log_path(config, category="development", run_id="run-123")

            self.assertEqual(path, root / "meta" / "runs" / "run-123" / "logs" / "development.log")
            self.assertTrue(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
