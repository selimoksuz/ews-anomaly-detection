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


class StructuredChainCallTests(unittest.TestCase):
    def test_structured_chain_uses_source_compatible_methodless_call(self):
        FakeChatOpenAI.last_instance = None
        settings = {
            "base_url": "https://manavgat.yzyonetim.zb/v1",
            "api_key": "test-key",
            "model": "gpt-oss-20b",
            "timeout_seconds": 120,
            "max_retries": 0,
            "max_tokens": None,
        }

        with patch.object(llm_anomaly, "load_llm_settings", return_value=settings), patch.object(
            llm_anomaly, "validate_llm_settings"
        ), patch.object(llm_anomaly, "anomaly_batch_schema", return_value=object), patch(
            "langchain_openai.ChatOpenAI", FakeChatOpenAI
        ), patch("langchain_core.prompts.ChatPromptTemplate", FakePromptTemplate):
            llm_anomaly.build_langchain_structured_chain()

        self.assertIsNotNone(FakeChatOpenAI.last_instance)
        self.assertEqual(FakeChatOpenAI.last_instance.structured_schema, object)
        self.assertEqual(FakeChatOpenAI.last_instance.structured_kwargs, {})


if __name__ == "__main__":
    unittest.main()
