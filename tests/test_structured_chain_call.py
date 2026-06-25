import os
import sys
import types
import unittest
from unittest.mock import patch

from llm import llm_anomaly


class FakePrompt:
    def __or__(self, other):
        return ("fake_chain", other)


class FakePromptTemplate:
    @staticmethod
    def from_messages(_messages):
        return FakePrompt()


class FakeChatOpenAI:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.structured_schema = None
        self.structured_kwargs = None
        FakeChatOpenAI.last_instance = self

    def with_structured_output(self, schema, **kwargs):
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return "fake_structured_llm"


class FakeResponse:
    results = [{"period_position": 0, "mono_id": "C1", "cohort_dt": "2026-05-31"}]


class FakeInvokeChain:
    def invoke(self, _payload):
        return FakeResponse()


class FailingInvokeChain:
    def invoke(self, _payload):
        raise ConnectionError("route closed")


class StructuredChainCallTests(unittest.TestCase):
    def fake_langchain_modules(self):
        fake_prompts = types.SimpleNamespace(ChatPromptTemplate=FakePromptTemplate)
        fake_core = types.SimpleNamespace(prompts=fake_prompts)
        fake_openai = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
        return {
            "langchain_core": fake_core,
            "langchain_core.prompts": fake_prompts,
            "langchain_openai": fake_openai,
        }

    def test_load_settings_ignores_legacy_timeout_sources(self):
        with patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": "120"}, clear=False), patch.object(
            llm_anomaly, "load_local_env_files"
        ), patch.object(
            llm_anomaly,
            "load_llm_secret_settings",
            return_value={
                "base_url": "https://manavgat.yzyonetim.zb/v1",
                "api_key": "test-key",
                "model": "gpt-oss-20b",
                "timeout_seconds": 120,
            },
        ):
            settings = llm_anomaly.load_llm_settings()

        self.assertIsNone(settings["timeout_seconds"])

    def test_structured_chain_uses_source_compatible_methodless_call(self):
        FakeChatOpenAI.last_instance = None
        settings = {
            "base_url": "https://manavgat.yzyonetim.zb/v1",
            "api_key": "test-key",
            "model": "gpt-oss-20b",
            "timeout_seconds": None,
            "max_retries": 0,
            "max_tokens": None,
        }

        with patch.dict(sys.modules, self.fake_langchain_modules()), patch.object(
            llm_anomaly, "load_llm_settings", return_value=settings
        ), patch.object(llm_anomaly, "validate_llm_settings"), patch.object(
            llm_anomaly, "anomaly_batch_schema", return_value=object
        ):
            llm_anomaly.build_langchain_structured_chain()

        self.assertIsNotNone(FakeChatOpenAI.last_instance)
        self.assertNotIn("timeout", FakeChatOpenAI.last_instance.kwargs)
        self.assertEqual(FakeChatOpenAI.last_instance.structured_schema, object)
        self.assertEqual(FakeChatOpenAI.last_instance.structured_kwargs, {})

    def test_structured_chain_ignores_configured_timeout_for_source_compatibility(self):
        FakeChatOpenAI.last_instance = None
        settings = {
            "base_url": "https://manavgat.yzyonetim.zb/v1",
            "api_key": "test-key",
            "model": "gpt-oss-20b",
            "timeout_seconds": 600,
            "max_retries": 0,
            "max_tokens": None,
        }

        with patch.dict(sys.modules, self.fake_langchain_modules()), patch.object(
            llm_anomaly, "load_llm_settings", return_value=settings
        ), patch.object(llm_anomaly, "validate_llm_settings"), patch.object(
            llm_anomaly, "anomaly_batch_schema", return_value=object
        ):
            llm_anomaly.build_langchain_structured_chain()

        self.assertIsNotNone(FakeChatOpenAI.last_instance)
        self.assertNotIn("timeout", FakeChatOpenAI.last_instance.kwargs)

    def test_first_customer_payload_preview_is_logged(self):
        evidence = [{"mono_id": "C1", "cohort_dt": "2026-05-31", "features": [1, 2, 3]}]
        with patch.object(llm_anomaly, "format_evidence_for_langchain", return_value="period_position=0 | mono_id=C1"):
            with self.assertLogs(llm_anomaly.logger, level="INFO") as captured:
                llm_anomaly.invoke_langchain_structured_decisions(
                    FakeInvokeChain(),
                    evidence,
                    payload_preview_index=1,
                )

        log_text = "\n".join(captured.output)
        self.assertIn("LLM PAYLOAD PREVIEW 1/3 START", log_text)
        self.assertIn("period_position=0 | mono_id=C1", log_text)
        self.assertIn("LLM PAYLOAD PREVIEW 1/3 END", log_text)

    def test_connection_error_log_includes_payload_context(self):
        evidence = [{"mono_id": "C1", "cohort_dt": "2026-05-31", "features": [1, 2, 3]}]
        with patch.object(llm_anomaly, "format_evidence_for_langchain", return_value="period_position=0 | mono_id=C1"):
            with self.assertLogs(llm_anomaly.logger, level="ERROR") as captured:
                with self.assertRaises(ConnectionError):
                    llm_anomaly.invoke_langchain_structured_decisions(
                        FailingInvokeChain(),
                        evidence,
                        payload_preview_index=1,
                    )

        log_text = "\n".join(captured.output)
        self.assertIn("exception_type=ConnectionError", log_text)
        self.assertIn("mono_id=C1", log_text)
        self.assertIn("payload_preview=period_position=0 | mono_id=C1", log_text)


if __name__ == "__main__":
    unittest.main()
