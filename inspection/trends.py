"""Trend inference: gpt-4o reasoning over what XTrace remembers.

XTrace stores what happened; this module is the analyst that reads it. After
every run, the current result plus everything memory recalls about it (similar
past failures, operator feedback, chef guidance, watch items) goes to gpt-4o,
which answers the question a head chef would ask: is this a one-off or a
trend, what is the root cause, and what single action fixes the line?

Two depths:
  analyze_run(result, context)   per-plate, runs inline after each verdict
  analyze_trends()               the deep read across recent kitchen memory,
                                 for the "long-running issues" button

Both return parsed dicts and degrade to None on any API failure; a broken
analyst never blocks a verdict.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import memory  # noqa: E402

RUN_PROMPT = """You are the line analyst for Seefu, an automated plate-check
station in a robot-run kitchen. Below is the CURRENT inspection result and what
KITCHEN MEMORY recalls about it. Judge whether this is a one-off or a trend,
name the most likely root cause on the line (ingredient hopper empty or
misloaded, portioner miscalibrated, wrong recipe loaded, camera or lighting
drift), and give the ONE action the kitchen should take now.

Reply with STRICT JSON only, no fences:
{"trend": "<one line: one-off, emerging, or established pattern and why>",
 "root_cause": "<one line, the likeliest physical cause>",
 "action": "<one imperative line>",
 "long_running": true|false,
 "confidence": "low"|"medium"|"high"}
Ground every claim in the evidence given. If memory shows nothing similar,
say it is a one-off with low confidence and a simple action."""

TRENDS_PROMPT = """You are the quality analyst for Seefu, an automated
plate-check station in a robot-run kitchen. Below is the kitchen's accumulated
memory: extracted facts about past plate failures and fixes, operator verdict
feedback, chef standing instructions, and a composed shift summary. Identify
the LONG-RUNNING issues: problems that recur across plates or shifts, not
one-off slips. For each, cite the evidence, name the physical root cause, and
give the fix that ends the pattern rather than patching one plate.

Reply with STRICT JSON only, no fences:
{"summary": "<two sentences on the overall state of the line>",
 "issues": [{"issue": "<one line>",
             "evidence": "<one line citing the remembered events>",
             "root_cause": "<one line>",
             "fix": "<one imperative line>",
             "priority": "high"|"medium"|"low"}]}
List at most 4 issues, highest priority first. An empty issues list is a valid
answer for a healthy line."""

_client = None


def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _ask(prompt, evidence, max_tokens):
    client = _get_client()
    if client is None:
        return None
    from openai import APIConnectionError, APIStatusError, APITimeoutError
    try:
        resp = client.with_options(timeout=14.0, max_retries=0).responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt + "\n\n" + evidence}]}],
            max_output_tokens=max_tokens,
        )
        text = resp.output_text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (APITimeoutError, APIConnectionError, APIStatusError, ValueError):
        return None


def run_evidence(result, mem):
    """Serialize the current result + recalled memory into the analyst's brief."""
    mem = mem or {}
    lines = ["CURRENT RESULT:"]
    lines.append(f"  dish {result.get('dish_id')}  verdict {result.get('verdict')}  "
                 f"score {result.get('score')} vs threshold {result.get('threshold')}")
    for f in result.get("findings", []):
        lines.append(f"  finding at {f.get('location') or 'unmapped'}: "
                     f"{f.get('description')}  (fix: {f.get('fix') or 'none given'})")
    if result.get("note"):
        lines.append(f"  station note: {result['note']}")
    lines.append("\nKITCHEN MEMORY RECALL:")
    lines.append(f"  similar past failures: {mem.get('recurrences', 0)}")
    for t in mem.get("matched", []) or []:
        lines.append(f"  remembered: {t}")
    for t in mem.get("lessons", []) or []:
        lines.append(f"  chef guidance: {t}")
    for t in mem.get("watch", []) or []:
        lines.append(f"  watch item: {t}")
    return "\n".join(lines)


def analyze_run(result, mem):
    """Per-plate trend read. Skipped when there is nothing to reason about."""
    if os.environ.get("SEEFU_TREND_ANALYSIS", "1") == "0":
        return None
    mem = mem or {}
    worth_it = (result.get("verdict") == "defect"
                or mem.get("recurrences") or mem.get("watch"))
    if not worth_it:
        return None
    return _ask(RUN_PROMPT, run_evidence(result, mem), max_tokens=220)


def analyze_trends():
    """The deep read: every recent memory in one brief, systemic issues out."""
    if not memory.enabled():
        return {"enabled": False, "summary": None, "issues": []}
    recent = memory.recent_memories(limit=20).get("memories", [])
    brief = memory.shift_briefing()
    watch = memory.watch_items()
    lessons = memory.trigger_lessons("trend-scan")
    lines = ["REMEMBERED EVENTS (newest first):"]
    for m in recent:
        lines.append(f"  [{m.get('type')}] {m.get('text')}")
    if lessons:
        lines.append("CHEF STANDING INSTRUCTIONS:")
        lines += [f"  {t}" for t in lessons]
    if watch:
        lines.append("OPERATOR WATCH ITEMS (passes that were wrong):")
        lines += [f"  {t}" for t in watch]
    if brief.get("context"):
        lines.append("COMPOSED SHIFT SUMMARY:\n" + str(brief["context"]))
    data = _ask(TRENDS_PROMPT, "\n".join(lines), max_tokens=500)
    if data is None:
        return {"enabled": True, "summary": None, "issues": [],
                "error": "analysis unavailable"}
    data.setdefault("issues", [])
    data["enabled"] = True
    return data
