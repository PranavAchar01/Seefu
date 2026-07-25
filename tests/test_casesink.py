"""Unit tests for core/casesink.py (LocalSink) - emit gating is the contract.

hub.emit and history are stubbed; CASES/CASE_SEQ redirected to a tempdir.
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import casesink


class SinkTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = (casesink.CASES, casesink.CASE_SEQ, casesink.hub.emit,
                      casesink.history.update_status, casesink.history.update_verified,
                      casesink.history.shift_summary)
        casesink.CASES = Path(self.tmp.name) / "cases"
        casesink.CASE_SEQ = Path(self.tmp.name) / ".case_seq"
        self.emitted, self.h_calls = [], []
        casesink.hub.emit = self.emitted.append
        casesink.history.update_status = lambda b, s: self.h_calls.append(("status", b, s))
        casesink.history.update_verified = lambda b, v: self.h_calls.append(("verified", b, v))
        casesink.history.shift_summary = lambda: {"inspected": 1, "autonomous": 1,
                                                  "escalated": 0, "top": []}
        self.sink = casesink.LocalSink()

    def tearDown(self):
        (casesink.CASES, casesink.CASE_SEQ, casesink.hub.emit,
         casesink.history.update_status, casesink.history.update_verified,
         casesink.history.shift_summary) = self._orig
        self.tmp.cleanup()

    def case_msgs(self):
        return [m for m in self.emitted if m["type"] == "case"]


class TestCreateCase(SinkTestBase):
    def test_ids_sequential_from_1041(self):
        a = self.sink.create_case("B1", "pass", [], "o.png")
        b = self.sink.create_case("B2", "defect", [], "o.png")
        self.assertEqual((a, b), ("1041", "1042"))

    def test_pass_creates_autonomous_defect_creates_escalated(self):
        a = self.sink.create_case("B1", "pass", [], "o.png")
        b = self.sink.create_case("B2", "defect", [], "o.png")
        self.assertEqual(json.loads((casesink.CASES / f"{a}.json").read_text())["status"],
                         "autonomous_resolution")
        self.assertEqual(json.loads((casesink.CASES / f"{b}.json").read_text())["status"],
                         "escalated")

    def test_create_emits_once_without_note_and_updates_history(self):
        self.sink.create_case("B1", "pass", [], "o.png")
        msgs = self.case_msgs()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["status"], "autonomous_resolution")
        self.assertIs(msgs[0]["verified"], False)
        self.assertNotIn("note", msgs[0])
        self.assertIn(("status", "B1", "autonomous_resolution"), self.h_calls)

    def test_next_id_unique_under_threads(self):
        ids = []
        threads = [threading.Thread(target=lambda: ids.append(self.sink._next_id()))
                   for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(set(ids)), 10)


class TestAddEvent(SinkTestBase):
    def test_appends_without_emitting(self):
        cid = self.sink.create_case("B1", "pass", [], "o.png")
        before = len(self.emitted)
        self.sink.add_event(cid, "operator note")
        case = json.loads((casesink.CASES / f"{cid}.json").read_text())
        self.assertEqual(case["events"][-1]["text"], "operator note")
        self.assertEqual(len(self.emitted), before)


class TestSetStatus(SinkTestBase):
    def test_transition_emits_case_and_shift(self):
        cid = self.sink.create_case("B1", "defect", [], "o.png")
        before = len(self.emitted)
        self.sink.set_status(cid, "quarantined")
        new = self.emitted[before:]
        self.assertEqual([m["type"] for m in new], ["case", "shift"])
        self.assertEqual(new[0]["status"], "quarantined")

    def test_same_status_without_note_is_silent(self):
        cid = self.sink.create_case("B1", "defect", [], "o.png")
        before = len(self.emitted)
        h_before = len(self.h_calls)
        self.sink.set_status(cid, "escalated")
        self.assertEqual(len(self.emitted), before)
        self.assertEqual(len(self.h_calls), h_before)

    def test_same_status_with_note_delivers_note(self):
        cid = self.sink.create_case("B1", "defect", [], "o.png")
        before = len(self.emitted)
        self.sink.set_status(cid, "escalated", note="J1 bent pins, operator review")
        new = self.emitted[before:]
        self.assertEqual(new[0]["note"], "J1 bent pins, operator review")

    def test_unknown_status_raises(self):
        cid = self.sink.create_case("B1", "pass", [], "o.png")
        with self.assertRaises(ValueError):
            self.sink.set_status(cid, "exploded")


class TestSetVerified(SinkTestBase):
    def test_verified_true_stores_note_and_emits(self):
        cid = self.sink.create_case("B1", "pass", [], "o.png")
        before = len(self.emitted)
        self.sink.set_verified(cid, True, "consistent")
        case = json.loads((casesink.CASES / f"{cid}.json").read_text())
        self.assertEqual(case["note"], "consistent")
        new = self.case_msgs()[-1]
        self.assertIs(new["verified"], True)
        self.assertEqual(new["note"], "consistent")
        self.assertGreater(len(self.emitted), before)
        self.assertIn(("verified", "B1", True), self.h_calls)

    def test_verified_false_keeps_note_out_and_stays_silent(self):
        cid = self.sink.create_case("B1", "pass", [], "o.png")
        before = len(self.emitted)
        self.sink.set_verified(cid, False, "verification unavailable: boom")
        case = json.loads((casesink.CASES / f"{cid}.json").read_text())
        self.assertEqual(case["note"], "")                      # events only
        self.assertIn("verification unavailable", case["events"][-1]["text"])
        self.assertEqual(len(self.emitted), before)


class TestEmitGating(SinkTestBase):
    def test_autonomous_unverified_note_suppressed(self):
        self.sink._emit({"case_id": "x", "status": "autonomous_resolution",
                         "verified": False, "note": "should not appear"})
        self.assertNotIn("note", self.emitted[-1])

    def test_escalated_unverified_note_included(self):
        self.sink._emit({"case_id": "x", "status": "escalated",
                         "verified": False, "note": "disposition"})
        self.assertEqual(self.emitted[-1]["note"], "disposition")

    def test_zendesk_id_rides_along(self):
        self.sink._emit({"case_id": "x", "status": "escalated",
                         "verified": True, "note": "", "zendesk_id": "88231"})
        self.assertEqual(self.emitted[-1]["zendesk_id"], "88231")


class TestGetSink(SinkTestBase):
    def test_default_is_local(self):
        self.assertIsInstance(casesink.get_sink(), casesink.LocalSink)


if __name__ == "__main__":
    unittest.main()
