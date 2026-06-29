import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.config_loader import load_secrets, resolve_secrets_path


class ConfigLoaderTests(unittest.TestCase):
    def test_missing_env_secret_path_falls_back_to_current_project_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            (project_root / "config").mkdir()
            (project_root / "secret").mkdir()
            (project_root / "config" / "pipeline_config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
            secrets_path = project_root / "secret" / "secrets.yaml"
            secrets_path.write_text("oracle: {}\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_SECRETS_PATH": str(project_root / "missing" / "secrets.yaml"),
                        "RISK_PIPELINE_SECRETS_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_secrets_path(), secrets_path.resolve())
                    self.assertEqual(load_secrets(), {"oracle": {}})
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
