"""Cortext context engine: compaction as a local memory operation.

The built-in Hermes compactor summarizes discarded turns with an auxiliary
LLM call. This engine replaces that: every turn is already in Cortext's
durable store (the memory provider ingests turns and tool results), so
compaction just swaps the middle of the transcript for a recalled memory
snapshot — working memory plus query-relevant long-term memories. No LLM
call, milliseconds, and nothing is irreversibly lost.
"""
from __future__ import annotations

import logging
from typing import Any

try:
  from agent.context_engine import ContextEngine
except ImportError:  # pragma: no cover — outside a Hermes install
  class ContextEngine: pass

logger = logging.getLogger(__name__)

BRIDGE_HEADER = ("[Earlier conversation was archived to durable memory. "
                 "Context recalled from memory:]")


def _text_of(message: dict[str, Any]) -> str:
  content = message.get("content")
  if isinstance(content, str): return content
  if isinstance(content, list):
    return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
  return ""


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
  ids = []
  for call in message.get("tool_calls") or []:
    call_id = call.get("id") or (call.get("function") or {}).get("id") if isinstance(call, dict) else None
    if call_id: ids.append(call_id)
  return ids


def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Keep tool_call/tool_result pairing valid after cutting the middle."""
  surviving = {cid for m in messages if m.get("role") == "assistant" for cid in _tool_call_ids(m)}
  kept: list[dict[str, Any]] = []
  for message in messages:
    if message.get("role") == "tool" and message.get("tool_call_id") not in surviving: continue
    kept.append(message)
  result_ids = {m.get("tool_call_id") for m in kept if m.get("role") == "tool"}
  patched: list[dict[str, Any]] = []
  for message in kept:
    patched.append(message)
    if message.get("role") == "assistant":
      for cid in _tool_call_ids(message):
        if cid not in result_ids:
          patched.append({"role": "tool", "tool_call_id": cid,
                          "content": "[Result archived to durable memory — recalled context above]"})
  return patched


class CortextContextEngine(ContextEngine):
  """Compaction without a summarizer: protected head/tail stay verbatim, the
  middle is replaced by a Cortext memory snapshot."""

  def __init__(self, provider: Any) -> None:
    self._provider = provider
    self.last_prompt_tokens = 0; self.last_completion_tokens = 0; self.last_total_tokens = 0
    self.threshold_tokens = 0; self.context_length = 0; self.compression_count = 0
    self.threshold_percent = 0.75; self.protect_first_n = 3; self.protect_last_n = 6

  @property
  def name(self) -> str: return "cortext"

  # -- Token tracking ------------------------------------------------------

  def update_from_response(self, usage: dict[str, Any]) -> None:
    self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
    self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
    self.last_total_tokens = int(usage.get("total_tokens") or 0)

  def on_session_start(self, session_id: str, **kwargs: Any) -> None:
    model = str(kwargs.get("model") or "")
    if not model: return
    try:
      from agent.model_metadata import get_model_context_length
      self.context_length = int(get_model_context_length(model) or 0)
    except Exception:
      self.context_length = 0
    self.threshold_tokens = int(self.context_length * self.threshold_percent) if self.context_length else 0

  def should_compress(self, prompt_tokens: int = None) -> bool:
    tokens = self.last_prompt_tokens if prompt_tokens is None else int(prompt_tokens or 0)
    return bool(self.threshold_tokens) and tokens >= self.threshold_tokens

  def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
    return bool(self._split(messages)[2])

  # -- Compaction ----------------------------------------------------------

  def _split(self, messages: list[dict[str, Any]]) -> tuple[list, list, list, list]:
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    head = rest[:self.protect_first_n]
    body = rest[self.protect_first_n:]
    tail_start = max(0, len(body) - self.protect_last_n)
    # Never start the tail on a tool result or mid tool-exchange: back up to a
    # user message so the kept suffix is a self-contained exchange.
    while 0 < tail_start < len(body) and body[tail_start].get("role") != "user":
      tail_start -= 1
    return system, head, body[:tail_start], body[tail_start:]

  def compress(self, messages: list[dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> list[dict[str, Any]]:
    system, head, middle, tail = self._split(messages)
    if not middle: return messages
    query = (focus_topic or "").strip()
    if not query:
      query = next((_text_of(m) for m in reversed(middle + tail) if m.get("role") == "user" and _text_of(m).strip()), "")
    snapshot = ""
    try:
      snapshot = self._provider.context_snapshot(query)
    except Exception as exc:
      logger.warning("Cortext context engine recall failed: %s", exc)
    bridge = {"role": "user", "content": (BRIDGE_HEADER + "\n" + snapshot) if snapshot
              else "[Earlier conversation was archived to durable memory.]"}
    compacted = _sanitize_tool_pairs(system + head + [bridge] + tail)
    self.compression_count += 1
    logger.info("Cortext compaction: %d -> %d messages (no LLM call)", len(messages), len(compacted))
    return compacted
