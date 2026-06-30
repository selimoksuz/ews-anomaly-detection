import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.config_loader import load_config, load_secrets, resolve_config_path, resolve_secrets_path


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

    def test_secret_path_can_be_discovered_from_parent_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "ews-anomaly-detection"
            project_root.mkdir()
            (project_root / "config").mkdir()
            (project_root / "config" / "pipeline_config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
            parent_secret = workspace / "secrets.yaml"
            parent_secret.write_text("oracle:\n  user: TEST\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_SECRETS_PATH": "",
                        "RISK_PIPELINE_SECRETS_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_secrets_path(), parent_secret.resolve())
                    self.assertEqual(load_secrets(), {"oracle": {"user": "TEST"}})
            finally:
                os.chdir(previous_cwd)

    def test_secret_path_accepts_notebook_workspace_spaced_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "ews-anomaly-detection"
            project_root.mkdir()
            (project_root / "config").mkdir()
            (project_root / "config" / "pipeline_config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
            parent_secret = workspace / "secrets .yaml"
            parent_secret.write_text("oracle:\n  user: TEST_SPACE\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_SECRETS_PATH": "",
                        "RISK_PIPELINE_SECRETS_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_secrets_path(), parent_secret.resolve())
                    self.assertEqual(load_secrets(), {"oracle": {"user": "TEST_SPACE"}})
            finally:
                os.chdir(previous_cwd)

    def test_config_path_can_be_discovered_from_parent_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "ews-anomaly-detection"
            project_root.mkdir()
            parent_config = workspace / "pipeline_config.yaml"
            parent_config.write_text("pipeline:\n  id_column: mono_id\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_CONFIG_PATH": "",
                        "RISK_PIPELINE_CONFIG_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_config_path(), parent_config.resolve())
                    self.assertEqual(load_config(), {"pipeline": {"id_column": "mono_id"}})
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
