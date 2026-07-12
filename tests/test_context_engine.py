from __future__ import annotations

import unittest
from typing import Any

from cortext_context import BRIDGE_HEADER, CortextContextEngine


class FakeProvider:
  def __init__(self, snapshot: str = "- Bailey is allergic to bee stings.") -> None:
    self.snapshot = snapshot; self.queries: list[str] = []
  def context_snapshot(self, query: str) -> str:
    self.queries.append(query); return self.snapshot


def _transcript() -> list[dict[str, Any]]:
  messages: list[dict[str, Any]] = [{"role": "system", "content": "You are Hermes."}]
  messages += [{"role": "user", "content": f"early message {i}"} for i in range(3)]
  for i in range(10):
    messages.append({"role": "user", "content": f"middle question {i}"})
    messages.append({"role": "assistant", "content": f"middle answer {i}"})
  messages.append({"role": "assistant", "content": "",
                   "tool_calls": [{"id": "call-1", "function": {"name": "terminal"}}]})
  messages.append({"role": "tool", "tool_call_id": "call-1", "content": "42 minutes"})
  messages.append({"role": "user", "content": "latest question"})
  messages.append({"role": "assistant", "content": "latest answer"})
  return messages


class ContextEngineTests(unittest.TestCase):
  def setUp(self) -> None:
    self.provider = FakeProvider(); self.engine = CortextContextEngine(self.provider)

  def test_compress_keeps_head_tail_and_injects_snapshot(self) -> None:
    messages = _transcript()
    compacted = self.engine.compress(messages)
    self.assertLess(len(compacted), len(messages))
    self.assertEqual(compacted[0]["role"], "system")
    self.assertEqual(compacted[1]["content"], "early message 0")
    self.assertEqual(compacted[-1]["content"], "latest answer")
    bridge = next(m for m in compacted if BRIDGE_HEADER in str(m.get("content")))
    self.assertIn("Bailey is allergic", bridge["content"])
    self.assertEqual(self.engine.compression_count, 1)

  def test_compress_uses_focus_topic_then_last_user_message(self) -> None:
    messages = _transcript()
    self.engine.compress(messages, focus_topic="the vet appointment")
    self.assertEqual(self.provider.queries[-1], "the vet appointment")
    self.engine.compress(messages)
    self.assertEqual(self.provider.queries[-1], "latest question")

  def test_tool_pairs_stay_valid(self) -> None:
    compacted = self.engine.compress(_transcript())
    calls = {cid for m in compacted if m.get("role") == "assistant"
             for tc in m.get("tool_calls") or [] for cid in [tc.get("id")] if cid}
    results = {m.get("tool_call_id") for m in compacted if m.get("role") == "tool"}
    self.assertEqual(calls, results)

  def test_should_compress_thresholds(self) -> None:
    self.assertFalse(self.engine.should_compress(10_000))  # no context length known
    self.engine.context_length = 100_000; self.engine.threshold_tokens = 75_000
    self.assertFalse(self.engine.should_compress(74_999))
    self.assertTrue(self.engine.should_compress(75_000))

  def test_nothing_to_compress_returns_input(self) -> None:
    short = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    self.assertEqual(self.engine.compress(short), short)
    self.assertFalse(self.engine.has_content_to_compress(short))
    self.assertTrue(self.engine.has_content_to_compress(_transcript()))


if __name__ == "__main__":
  unittest.main()
