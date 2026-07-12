"""Compaction ablation: Hermes's default summarizer vs the Cortext engine.

Both engines compact the identical transcript with identical protect windows
(first 3, last 6 non-system messages), forced. A model then answers the
benchmark probes from each compacted context; we score expected-fact groups
in the answers. Arms:

  full      no compaction (upper bound)
  default   built-in ContextCompressor (aux-LLM summary of the middle)
  cortext   CortextContextEngine (middle replaced by memory snapshot)

Usage: python -m bench.compaction_ablation   (needs OPENAI_API_KEY in .env)
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from bench import harness, live_judge, scenario

MODEL = "gpt-5.4-mini"
PROTECT_FIRST, PROTECT_LAST = 3, 6


FILLER_TOPICS = [
  "the history of the espresso machine", "how tides work", "the plot of a heist movie",
  "sourdough starters", "why the sky is blue", "keyboard switch types", "marathon training",
  "the difference between alligators and crocodiles", "how noise-cancelling headphones work",
  "famous bridges", "the rules of cricket", "how vaccines are developed", "olive oil grades",
  "the tallest buildings in the world", "how compilers optimize loops", "birdwatching basics",
  "the origins of jazz", "how batteries degrade", "types of pasta", "the water cycle",
]


def build_transcript(filler_rounds: int = 2) -> list[dict]:
  """Scenario facts early, then a long stretch of distractor chatter, so the
  facts sit squarely inside the region compaction will discard."""
  messages = [{"role": "system", "content": "You are a helpful personal assistant."}]
  for session in scenario.SESSIONS:
    for user, assistant in session:
      messages.append({"role": "user", "content": user})
      messages.append({"role": "assistant", "content": assistant})
  for round_index in range(filler_rounds):
    for topic in FILLER_TOPICS:
      messages.append({"role": "user", "content": f"Tell me something interesting about {topic}."})
      messages.append({"role": "assistant", "content": (
        f"Here's a detail about {topic}: it is a richer subject than most people expect, "
        f"with a surprising amount of engineering, history, and folklore behind it. "
        f"(round {round_index + 1}) " + "It rewards a closer look. " * 10)})
  return messages


def chat_with_messages(messages: list[dict], question: str) -> str:
  import os, urllib.request
  payload = {"model": MODEL, "messages": messages + [{"role": "user", "content": question}]}
  req = urllib.request.Request(
    (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")) + "/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    method="POST")
  with urllib.request.urlopen(req, timeout=180, context=live_judge._ssl_context()) as resp:
    body = json.loads(resp.read().decode())
  content = body["choices"][0]["message"]["content"]
  return content if isinstance(content, str) else str(content)


def probe_arm(name: str, messages: list[dict], prefetch=None) -> dict:
  """Ask every probe against the given context. ``prefetch`` mimics the
  per-turn memory injection Hermes performs when a memory provider is
  active: its packet for the question is added as prior context."""
  probes = {}
  packet_tokens = []
  for probe in scenario.PROBES:
    if probe.get("expect_unknown"):
      continue
    context = list(messages)
    if prefetch is not None:
      packet = prefetch(probe["question"]) or ""
      packet_tokens.append(harness.est_tokens(packet))
      if packet:
        context.append({"role": "system", "content": "Prior context from durable memory:\n" + packet})
    answer = chat_with_messages(context, probe["question"] + " Answer only from this conversation; if you don't know, say so.")
    matched, missed = harness.match_groups(answer, probe["expected"])
    stale, _ = harness.match_groups(answer, probe.get("stale", []))
    probes[probe["id"]] = {"matched": matched, "missed": missed, "stale": stale, "answer": answer}
  hits = sum(len(p["matched"]) for p in probes.values())
  total = sum(len(p["matched"]) + len(p["missed"]) for p in probes.values())
  stale = sum(len(p["stale"]) for p in probes.values())
  mean_packet = round(sum(packet_tokens) / len(packet_tokens)) if packet_tokens else 0
  return {"arm": name, "hits": hits, "total": total, "stale": stale,
          "context_tokens": harness.est_tokens(json.dumps(messages)) + mean_packet,
          "probes": probes}


def run_cortext_arm(transcript: list[dict]) -> tuple[list[dict], float, int]:
  from cortext_context import CortextContextEngine
  from provider import CortextMemoryProvider
  import os
  home = Path(tempfile.mkdtemp(prefix="ablate-cortext-"))
  os.environ["HERMES_HOME"] = str(home)
  provider = CortextMemoryProvider()
  provider.initialize("ablation", hermes_home=str(home), platform="cli",
                      agent_context="primary", user_id="ablate", agent_identity="agent")
  turn, last_user = 0, ""
  for message in transcript:
    if message["role"] == "user":
      turn += 1; last_user = message["content"]
      provider.on_turn_start(turn, last_user)
    elif message["role"] == "assistant" and turn:
      provider.sync_turn(last_user, message["content"])
  provider._drain()
  engine = CortextContextEngine(provider)
  engine.protect_first_n, engine.protect_last_n = PROTECT_FIRST, PROTECT_LAST
  with harness.NetworkMeter() as net:
    started = time.perf_counter()
    compacted = engine.compress(list(transcript))
    elapsed = time.perf_counter() - started
  provider.shutdown()
  # Cold-start reader on the same store, as Hermes prefetch would use it.
  reader = CortextMemoryProvider()
  reader.initialize("ablation-probe", hermes_home=str(home), platform="cli",
                    agent_context="primary", user_id="ablate", agent_identity="agent")
  return compacted, elapsed, len(net.connections), reader


def run_default_arm(transcript: list[dict]) -> tuple[list[dict], float, int]:
  import os
  # Route the compressor's auxiliary-LLM chain to OpenAI: isolate from the
  # user's real ~/.hermes (whose main provider may be misconfigured) and
  # strip competing provider keys so the chain resolves the custom endpoint
  # (OPENAI_BASE_URL + OPENAI_API_KEY).
  home = tempfile.mkdtemp(prefix="ablate-default-home-")
  os.environ["HERMES_HOME"] = home
  (Path(home) / "config.yaml").write_text(
    "model:\n  provider: custom\n  name: gpt-5.4-mini\n"
    "  base_url: https://api.openai.com/v1\n")
  os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
  for key in list(os.environ):
    if key.endswith("_API_KEY") and key != "OPENAI_API_KEY":
      os.environ.pop(key)
  from agent.context_compressor import ContextCompressor
  compressor = ContextCompressor(
    model=MODEL,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    protect_first_n=PROTECT_FIRST, protect_last_n=PROTECT_LAST,
    config_context_length=16_000, quiet_mode=True,
  )
  with harness.NetworkMeter() as net:
    started = time.perf_counter()
    compacted = compressor.compress(list(transcript), current_tokens=20_000, force=True)
    elapsed = time.perf_counter() - started
  return compacted, elapsed, len(net.connections)


def run_default_failed_aux_arm(transcript: list[dict]) -> tuple[list[dict], float]:
  """The default compactor when its auxiliary-LLM chain is unavailable — the
  shipped fallback drops the middle and inserts a placeholder."""
  import os
  from agent.context_compressor import ContextCompressor
  os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="ablate-failed-home-")
  real_key = os.environ.pop("OPENAI_API_KEY", "")
  real_base = os.environ.pop("OPENAI_BASE_URL", "")
  try:
    compressor = ContextCompressor(
      model=MODEL, protect_first_n=PROTECT_FIRST, protect_last_n=PROTECT_LAST,
      config_context_length=16_000, quiet_mode=True)
    started = time.perf_counter()
    compacted = compressor.compress(list(transcript), current_tokens=20_000, force=True)
    return compacted, time.perf_counter() - started
  finally:
    if real_key: os.environ["OPENAI_API_KEY"] = real_key
    if real_base: os.environ["OPENAI_BASE_URL"] = real_base


def main() -> int:
  harness.load_dotenv()
  transcript = build_transcript()
  print(f"transcript: {len(transcript)} messages, ~{harness.est_tokens(json.dumps(transcript))} tokens")

  results = [probe_arm("full (no compaction)", transcript)]

  compacted, seconds, connections = run_default_arm(transcript)
  print(f"default compactor: {len(transcript)} -> {len(compacted)} messages in {seconds:.2f}s, {connections} net connections")
  results.append(probe_arm("default summarizer", compacted) | {"compaction_s": round(seconds, 2), "compaction_net": connections})

  compacted, seconds = run_default_failed_aux_arm(transcript)
  print(f"default (aux down): {len(transcript)} -> {len(compacted)} messages in {seconds:.2f}s")
  results.append(probe_arm("default summarizer (aux LLM down)", compacted) | {"compaction_s": round(seconds, 2), "compaction_net": 0})

  compacted, seconds, connections, reader = run_cortext_arm(transcript)
  print(f"cortext engine:    {len(transcript)} -> {len(compacted)} messages in {seconds:.2f}s, {connections} net connections")
  def cortext_prefetch(question: str) -> str:
    reader._cache = ("", "")
    return reader.prefetch(question)
  results.append(probe_arm("cortext engine + provider", compacted, prefetch=cortext_prefetch)
                 | {"compaction_s": round(seconds, 2), "compaction_net": connections})
  reader.shutdown()

  out = REPO_ROOT / "bench" / "results-compaction"
  out.mkdir(parents=True, exist_ok=True)
  (out / "results.json").write_text(json.dumps(results, indent=2))

  print("\n| Arm | Facts recalled post-compaction | Stale leaks | Context tokens | Compaction time | Compaction net calls |")
  print("| --- | --- | ---: | ---: | ---: | ---: |")
  for r in results:
    print(f"| {r['arm']} | {r['hits']}/{r['total']} | {r['stale']} | {r['context_tokens']} | "
          f"{r.get('compaction_s', '—')} | {r.get('compaction_net', '—')} |")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
