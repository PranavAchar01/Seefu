"""Unit tests for inspection/trends.py and the feedback side of core/memory.py.
No network: the OpenAI client is injected, XTrace calls are stubbed at _post.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inspection"))

from core import memory
import trends

RUN_JSON = json.dumps({"trend": "established pattern: third garnish miss",
                       "root_cause": "scallion hopper empty",
                       "action": "refill the scallion station",
                       "long_running": True, "confidence": "high"})


class FakeClient:
    def __init__(self, text=RUN_JSON, exc=None):
        self.text, self.exc, self.calls = text, exc, 0

    def with_options(self, **_kw):
        return self

    @property
    def responses(self):
        return self

    def create(self, **_kw):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return type("R", (), {"output_text": self.text})()


def timeout_exc():
    import httpx
    from openai import APITimeoutError
    return APITimeoutError(request=httpx.Request("POST", "https://api.test"))


RESULT = {"dish_id": "DISH-009", "verdict": "defect", "score": 47.6,
          "threshold": 49.4,
          "findings": [{"location": "centre of dish",
                        "description": "Missing scallion garnish",
                        "fix": "Re-garnish with scallion and send"}]}
MEM = {"recurrences": 2, "matched": ["Dish DISH-006 is missing scallions"],
       "lessons": [], "watch": []}


class TrendsBase(unittest.TestCase):
    def setUp(self):
        self._client = trends._client

    def tearDown(self):
        trends._client = self._client


class TestAnalyzeRun(TrendsBase):
    def test_defect_run_returns_parsed_analysis(self):
        trends._client = FakeClient()
        out = trends.analyze_run(RESULT, MEM)
        self.assertTrue(out["long_running"])
        self.assertIn("hopper", out["root_cause"])
        self.assertIn("refill", out["action"])

    def test_clean_pass_with_no_memory_is_skipped(self):
        fake = FakeClient()
        trends._client = fake
        out = trends.analyze_run({"verdict": "pass", "findings": []}, None)
        self.assertIsNone(out)
        self.assertEqual(fake.calls, 0)

    def test_pass_with_watch_items_is_analyzed(self):
        trends._client = FakeClient()
        out = trends.analyze_run({"verdict": "pass", "findings": []},
                                 {"watch": ["a pass was wrong before"]})
        self.assertIsNotNone(out)

    def test_fenced_json_is_tolerated(self):
        trends._client = FakeClient(text="```json\n" + RUN_JSON + "\n```")
        self.assertIsNotNone(trends.analyze_run(RESULT, MEM))

    def test_api_failure_degrades_to_none(self):
        trends._client = FakeClient(exc=timeout_exc())
        self.assertIsNone(trends.analyze_run(RESULT, MEM))

    def test_env_kill_switch(self):
        trends._client = FakeClient()
        with mock.patch.dict(os.environ, {"SEEFU_TREND_ANALYSIS": "0"}):
            self.assertIsNone(trends.analyze_run(RESULT, MEM))

    def test_evidence_carries_findings_and_recall(self):
        text = trends.run_evidence(RESULT, MEM)
        for needle in ("Missing scallion garnish", "similar past failures: 2",
                       "DISH-006", "Re-garnish"):
            self.assertIn(needle, text)


class TestFeedbackMemory(unittest.TestCase):
    def canned(self, rows):
        return {"data": [{"score": s, "text": t, "type": "fact"} for s, t in rows]}

    def test_count_overrules_counts_only_wrong_hold_feedback(self):
        rows = self.canned([
            (0.7, "Operator verdict feedback: the DEFECT verdict on dish "
                  "DISH-005 was WRONG. Findings were: missing garnish."),
            (0.6, "Operator verdict feedback: the DEFECT verdict on dish "
                  "DISH-006 was WRONG. Findings were: missing garnish."),
            (0.9, "Operator verdict feedback: the DEFECT verdict on dish "
                  "DISH-007 was RIGHT. Findings were: missing garnish."),
            (0.2, "Operator verdict feedback: the DEFECT verdict was WRONG."),
            (0.8, "Dish DISH-002 is missing broccoli."),
        ])
        with mock.patch.object(memory, "enabled", return_value=True), \
             mock.patch.object(memory, "search", return_value=rows):
            self.assertEqual(memory.count_overrules("missing garnish"), 2)

    def test_watch_items_keep_only_wrong_pass_feedback(self):
        rows = self.canned([
            (0.7, "Operator verdict feedback: the PASS verdict on dish "
                  "DISH-008 was WRONG. Operator note: it missed the garnish."),
            (0.8, "Operator verdict feedback: the DEFECT verdict on dish "
                  "DISH-005 was WRONG."),
        ])
        with mock.patch.object(memory, "enabled", return_value=True), \
             mock.patch.object(memory, "search", return_value=rows):
            items = memory.watch_items()
        self.assertEqual(len(items), 1)
        self.assertIn("PASS", items[0])

    def test_record_feedback_serializes_verdict_and_note(self):
        sent = {}

        def fake_post(path, payload, timeout, params=None):
            sent.update(payload)
            return {"ok": True}

        case = {"dish_id": "DISH-009", "verdict": "defect",
                "findings": [{"description": "Missing scallion garnish"}]}
        with mock.patch.object(memory, "enabled", return_value=True), \
             mock.patch.object(memory, "_post", side_effect=fake_post):
            memory.record_feedback(case, False, "we are out of scallions")
        content = sent["messages"][0]["content"]
        self.assertIn("Operator verdict feedback", content)
        self.assertIn("WRONG", content)
        self.assertIn("Missing scallion garnish", content)
        self.assertIn("out of scallions", content)
        self.assertEqual(sent["conv_id"], "verdict-feedback")

    def test_disabled_memory_short_circuits(self):
        with mock.patch.object(memory, "enabled", return_value=False):
            self.assertEqual(memory.count_overrules("x"), 0)
            self.assertEqual(memory.watch_items(), [])
            self.assertIsNone(memory.record_feedback({}, True))


if __name__ == "__main__":
    unittest.main()
