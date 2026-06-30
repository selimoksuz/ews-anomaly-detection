import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import engine.config_loader as config_loader
from engine.config_loader import load_config, load_secrets, resolve_config_path, resolve_secrets_path
from engine.oracle_io import OracleConnector


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
            parent_config.write_text(
                "pipeline:\n  id_column: mono_id\noracle:\n  section: TEST\n",
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.object(config_loader, "PROJECT_ROOT", project_root), patch.object(
                    config_loader,
                    "DEFAULT_CONFIG_PATH",
                    project_root / "missing" / "pipeline_config.yaml",
                ), patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_CONFIG_PATH": "",
                        "RISK_PIPELINE_CONFIG_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_config_path(), parent_config.resolve())
                    config = load_config()
                    self.assertEqual(config["pipeline"]["id_column"], "mono_id")
                    self.assertEqual(config["oracle"]["section"], "TEST")
                    self.assertIn("llm_feature_details", config["oracle"]["tables"])
            finally:
                os.chdir(previous_cwd)

    def test_parent_config_without_pipeline_schema_is_ignored_when_repo_config_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "ews-anomaly-detection"
            (project_root / "config").mkdir(parents=True)
            repo_config = project_root / "config" / "pipeline_config.yaml"
            repo_config.write_text(
                "pipeline:\n  id_column: repo_mono\noracle:\n  section: REPO\n",
                encoding="utf-8",
            )
            parent_config = workspace / "pipeline_config.yaml"
            parent_config.write_text("model:\n  name: other_project\n", encoding="utf-8")

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
                    self.assertEqual(resolve_config_path(), repo_config.resolve())
                    config = load_config()
                    self.assertEqual(config["pipeline"]["id_column"], "repo_mono")
                    self.assertEqual(config["oracle"]["section"], "REPO")
            finally:
                os.chdir(previous_cwd)

    def test_invalid_env_config_path_falls_back_to_repo_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "ews-anomaly-detection"
            (project_root / "config").mkdir(parents=True)
            repo_config = project_root / "config" / "pipeline_config.yaml"
            repo_config.write_text(
                "pipeline:\n  id_column: repo_mono\noracle:\n  section: REPO\n",
                encoding="utf-8",
            )
            invalid_config = workspace / "pipeline_config.yaml"
            invalid_config.write_text("model:\n  name: other_project\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "EWS_ANOMALY_CONFIG_PATH": str(invalid_config),
                        "RISK_PIPELINE_CONFIG_PATH": "",
                    },
                ):
                    self.assertEqual(resolve_config_path(), repo_config.resolve())
                    config = load_config()
                    self.assertEqual(config["pipeline"]["id_column"], "repo_mono")
                    self.assertEqual(config["oracle"]["section"], "REPO")
            finally:
                os.chdir(previous_cwd)

    def test_oracle_connector_merges_non_pipeline_config_with_defaults(self):
        connector = OracleConnector(
            pipeline_config={"model": {"name": "other_project"}},
            secrets={
                "oracle": {
                    "sections": {
                        "ORA_PRD_ZTUSER": {
                            "user": "user",
                            "password": "password",
                            "host": "host",
                            "port": 1521,
                            "service_name": "svc",
                        }
                    }
                }
            },
        )
        self.assertEqual(connector.pipeline_settings["id_column"], "mono_id")
        self.assertIn("llm_feature_details", connector.oracle_settings["tables"])
        self.assertEqual(connector.pipeline_config["model"]["name"], "other_project")


if __name__ == "__main__":
    unittest.main()
