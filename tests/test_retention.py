import json
import tempfile
import unittest
from pathlib import Path

from engine.retention import RetentionManager


class RetentionManagerTests(unittest.TestCase):
    def test_reset_runtime_state_clears_runtime_directories_and_recreates_registries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {
                "registry": {
                    "meta_dir": str(root / "meta"),
                    "artifacts_dir": str(root / "artifacts"),
                    "logs_dir": str(root / "logs"),
                    "run_registry_file": str(root / "meta" / "run_registry.json"),
                    "model_registry_file": str(root / "meta" / "model_registry.json"),
                    "champion_registry_file": str(root / "meta" / "champions.json"),
                    "registry_lock_file": str(root / "meta" / ".registry.lock"),
                },
                "monitoring": {
                    "directory": str(root / "meta" / "monitoring"),
                },
                "sources": {
                    "outputs": {
                        "csv": {
                            "directory": str(root / "output" / "live_scores"),
                        }
                    }
                },
            }

            for path in (
                root / "logs" / "old.log",
                root / "artifacts" / "ALL" / "model.pkl",
                root / "meta" / "runs" / "run-a" / "manifest.json",
                root / "meta" / "monitoring" / "summary.json",
                root / "output" / "live_scores" / "old.csv",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            retention = RetentionManager(config)
            result = retention.reset_runtime_state()

            self.assertEqual(result["registry_files"], 3)
            self.assertTrue((root / "logs").exists())
            self.assertTrue((root / "artifacts").exists())
            self.assertTrue((root / "meta" / "runs").exists())
            self.assertEqual(json.loads((root / "meta" / "run_registry.json").read_text(encoding="utf-8")), [])
            self.assertEqual(json.loads((root / "meta" / "model_registry.json").read_text(encoding="utf-8")), [])
            self.assertEqual(json.loads((root / "meta" / "champions.json").read_text(encoding="utf-8")), {})


if __name__ == "__main__":
    unittest.main()
