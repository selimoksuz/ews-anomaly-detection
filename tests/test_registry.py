import tempfile
import unittest
from pathlib import Path

from engine.registry import RegistryManager


class RegistryManagerTests(unittest.TestCase):
    def test_run_and_model_registry_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {
                "registry": {
                    "meta_dir": str(root / "meta"),
                    "artifacts_dir": str(root / "artifacts"),
                    "run_registry_file": str(root / "meta" / "run_registry.json"),
                    "model_registry_file": str(root / "meta" / "model_registry.json"),
                    "champion_registry_file": str(root / "meta" / "champions.json"),
                }
            }

            registry = RegistryManager(config)
            self.assertEqual(registry.runs_dir, root / "meta" / "runs")
            run = registry.start_run("develop", "SEG_A", {"foo": "bar"})
            self.assertEqual(run.run_dir, root / "meta" / "runs" / run.run_id)
            self.assertEqual(run.manifest_path, run.run_dir / "manifest.json")
            registry.finish_run(run, "completed", {"ok": True})

            registry.register_model(
                {
                    "model_version": "SEG_A-develop-1",
                    "segment": "SEG_A",
                    "status": "candidate",
                    "created_at": run.created_at,
                    "artifact_path": str(run.artifact_dir / "model.pkl"),
                }
            )
            registry.promote_model("SEG_A", "SEG_A-develop-1")

            champion = registry.get_champion("SEG_A")
            self.assertIsNotNone(champion)
            self.assertEqual(champion["model_version"], "SEG_A-develop-1")

    def test_corrupted_run_registry_is_rebuilt_from_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {
                "registry": {
                    "meta_dir": str(root / "meta"),
                    "artifacts_dir": str(root / "artifacts"),
                    "run_registry_file": str(root / "meta" / "run_registry.json"),
                    "model_registry_file": str(root / "meta" / "model_registry.json"),
                    "champion_registry_file": str(root / "meta" / "champions.json"),
                }
            }

            registry = RegistryManager(config)
            self.assertEqual(registry.runs_dir, root / "meta" / "runs")
            run = registry.start_run("develop", "SEG_A", {"foo": "bar"})
            registry.finish_run(run, "completed", {"ok": True})

            run_registry_path = root / "meta" / "run_registry.json"
            run_registry_path.write_text('[] trailing-garbage', encoding="utf-8")

            rebuilt = registry.rebuild_run_registry()
            self.assertEqual(len(rebuilt), 1)
            self.assertEqual(rebuilt[0]["run_id"], run.run_id)

            recovered = registry.start_run("score-live", "SEG_A", {"foo": "bar"})
            self.assertEqual(recovered.run_type, "score-live")


if __name__ == "__main__":
    unittest.main()
