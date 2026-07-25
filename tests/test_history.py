"""Unit tests for core/history.py — sqlite inspection log.

Pure unit tests: the module-level DB path constant is redirected into a
TemporaryDirectory in setUp and restored in tearDown, so the real
runs/history.db is never touched. No server, no camera, no network.

Note: the spec name "record()" corresponds to history.log_inspection().
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core import history


class HistoryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_db = history.DB
        # _conn() does DB.parent.mkdir(exist_ok=True) (no parents=True), so the
        # redirected path must be exactly one level below an existing dir.
        history.DB = Path(self._tmp.name) / "runs" / "history.db"

    def tearDown(self):
        history.DB = self._orig_db
        self._tmp.cleanup()

    # -- helpers ------------------------------------------------------------

    def seed(self, dish_id="B1", verdict="defect", designator="R5",
             defect="solder_bridge", score=0.9, status="pending"):
        return history.log_inspection(dish_id, verdict, designator, defect,
                                      score, status)

    def raw(self, sql, params=()):
        con = sqlite3.connect(history.DB)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def set_ts(self, row_id, ts):
        con = sqlite3.connect(history.DB)
        with con:
            con.execute("UPDATE inspections SET ts=? WHERE id=?", (ts, row_id))
        con.close()

    # -- log_inspection -----------------------------------------------------

    def test_log_inspection_inserts_row_and_returns_rowid(self):
        rid = self.seed(dish_id="B7", verdict="defect", designator="C3",
                        defect="tombstone", score=0.42, status="escalated")
        self.assertEqual(rid, 1)
        rows = self.raw("SELECT ts, dish_id, verdict, designator, defect, "
                        "score, status, verified FROM inspections")
        self.assertEqual(len(rows), 1)
        ts, dish, verdict, desig, defect, score, status, verified = rows[0]
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        self.assertEqual(dish, "B7")
        self.assertEqual(verdict, "defect")
        self.assertEqual(desig, "C3")
        self.assertEqual(defect, "tombstone")
        self.assertAlmostEqual(score, 0.42)
        self.assertEqual(status, "escalated")
        self.assertIsNone(verified)  # verified is not set on insert

    def test_log_inspection_ids_autoincrement(self):
        self.assertEqual(self.seed(), 1)
        self.assertEqual(self.seed(), 2)
        self.assertEqual(self.seed(), 3)

    def test_log_inspection_coerces_score_to_float(self):
        self.seed(score="0.5")
        (score,), = self.raw("SELECT score FROM inspections")
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.5)

    def test_log_inspection_none_score_inserts_null(self):
        # score is nullable in the schema; None must insert NULL, not raise
        row_id = self.seed(score=None)
        con = history._conn()
        stored = con.execute("SELECT score FROM inspections WHERE id=?",
                             (row_id,)).fetchone()[0]
        con.close()
        self.assertIsNone(stored)

    # -- shift_summary ------------------------------------------------------

    def test_shift_summary_empty_db_returns_zeros_and_empty_top(self):
        summary = history.shift_summary()
        self.assertEqual(summary, {"inspected": 0, "autonomous": 0,
                                   "escalated": 0, "top": []})

    def test_shift_summary_counts_mixed_statuses(self):
        # 2 autonomous, 1 escalated + 1 quarantined (both count as escalated),
        # 2 that are neither -> inspected counts everything.
        self.seed(dish_id="A1", verdict="defect", status="autonomous_resolution")
        self.seed(dish_id="A2", verdict="defect", status="autonomous_resolution")
        self.seed(dish_id="A3", verdict="defect", status="escalated")
        self.seed(dish_id="A4", verdict="defect", status="quarantined")
        self.seed(dish_id="A5", verdict="pass", designator=None, defect=None,
                  status="ok")
        self.seed(dish_id="A6", verdict="pass", designator=None, defect=None,
                  status="pending")
        summary = history.shift_summary()
        self.assertEqual(summary["inspected"], 6)
        self.assertEqual(summary["autonomous"], 2)
        self.assertEqual(summary["escalated"], 2)

    def test_shift_summary_top_designators_ranked_and_limited_to_three(self):
        for _ in range(4):
            self.seed(designator="R5")
        for _ in range(3):
            self.seed(designator="C2")
        for _ in range(2):
            self.seed(designator="U1")
        self.seed(designator="D9")  # 4th distinct designator, cut by LIMIT 3
        # verdict != 'defect' rows must not count, even with a designator:
        for _ in range(5):
            self.seed(verdict="pass", designator="R9", defect=None, status="ok")
        summary = history.shift_summary()
        self.assertEqual(summary["top"], [
            {"designator": "R5", "count": 4},
            {"designator": "C2", "count": 3},
            {"designator": "U1", "count": 2},
        ])

    def test_shift_summary_none_designator_rows_never_in_top(self):
        self.seed(designator=None)
        self.seed(designator=None)
        self.seed(designator=None)
        summary = history.shift_summary()
        self.assertEqual(summary["inspected"], 3)
        self.assertEqual(summary["top"], [])

    def test_shift_summary_reflects_status_updates(self):
        self.seed(dish_id="B1", status="pending")
        history.update_status("B1", "autonomous_resolution")
        summary = history.shift_summary()
        self.assertEqual(summary["autonomous"], 1)
        self.assertEqual(summary["escalated"], 0)

    # -- defect_history -----------------------------------------------------

    def test_defect_history_only_that_designator_and_only_defect_verdict(self):
        self.seed(dish_id="B1", designator="R5", defect="solder_bridge",
                  score=0.9, status="escalated")
        self.seed(dish_id="B2", designator="R5", defect="missing",
                  score=0.7, status="pending")
        self.seed(dish_id="B3", designator="C2", defect="tombstone",
                  score=0.8, status="pending")           # other designator
        self.seed(dish_id="B4", verdict="pass", designator="R5",
                  defect=None, status="ok")              # not a defect verdict
        rows = history.defect_history("R5")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["dish_id"] for r in rows}, {"B1", "B2"})
        for r in rows:
            self.assertEqual(set(r), {"ts", "dish_id", "defect", "score",
                                      "status"})

    def test_defect_history_newest_first_across_distinct_timestamps(self):
        a = self.seed(dish_id="OLD", defect="a")
        b = self.seed(dish_id="NEW", defect="b")
        c = self.seed(dish_id="MID", defect="c")
        self.set_ts(a, "2026-07-23 10:00:00")
        self.set_ts(b, "2026-07-23 12:00:00")
        self.set_ts(c, "2026-07-23 11:00:00")
        rows = history.defect_history("R5")
        self.assertEqual([r["dish_id"] for r in rows], ["NEW", "MID", "OLD"])

    def test_defect_history_same_second_ties_are_newest_first(self):
        # ts has 1-second resolution; the id tiebreaker keeps newest-first true
        # for rows logged within the same second
        a = self.seed(dish_id="FIRST")
        b = self.seed(dish_id="SECOND")
        self.set_ts(a, "2026-07-23 10:00:00")
        self.set_ts(b, "2026-07-23 10:00:00")
        rows = history.defect_history("R5")
        self.assertEqual([r["dish_id"] for r in rows], ["SECOND", "FIRST"])

    def test_defect_history_unknown_designator_returns_empty(self):
        self.seed(designator="R5")
        self.assertEqual(history.defect_history("NOPE"), [])

    def test_defect_history_none_designator_returns_empty(self):
        # CURRENT BEHAVIOR: SQL "designator = ?" with None compares against
        # NULL and never matches, so NULL-designator defect rows are
        # unreachable through this API.
        self.seed(designator=None)
        self.assertEqual(history.defect_history(None), [])

    # -- update_status / update_verified ------------------------------------

    def test_update_status_hits_all_rows_of_that_board_only(self):
        self.seed(dish_id="B1", status="pending")
        self.seed(dish_id="B1", status="pending")
        self.seed(dish_id="B2", status="pending")
        history.update_status("B1", "escalated")
        self.assertEqual(self.raw("SELECT status FROM inspections "
                                  "WHERE dish_id='B1'"),
                         [("escalated",), ("escalated",)])
        self.assertEqual(self.raw("SELECT status FROM inspections "
                                  "WHERE dish_id='B2'"),
                         [("pending",)])

    def test_update_verified_coerces_truthy_to_1_and_falsy_to_0(self):
        self.seed(dish_id="B1")
        self.seed(dish_id="B2")
        history.update_verified("B1", True)
        history.update_verified("B2", "")
        self.assertEqual(self.raw("SELECT verified FROM inspections "
                                  "WHERE dish_id='B1'"), [(1,)])
        self.assertEqual(self.raw("SELECT verified FROM inspections "
                                  "WHERE dish_id='B2'"), [(0,)])
        # any truthy value, not just True:
        history.update_verified("B2", "yes")
        self.assertEqual(self.raw("SELECT verified FROM inspections "
                                  "WHERE dish_id='B2'"), [(1,)])

    def test_update_verified_hits_only_matching_board(self):
        self.seed(dish_id="B1")
        self.seed(dish_id="B2")
        history.update_verified("B1", True)
        self.assertEqual(self.raw("SELECT verified FROM inspections "
                                  "WHERE dish_id='B2'"), [(None,)])

    def test_updates_on_missing_board_are_silent_noops(self):
        self.seed(dish_id="B1", status="pending")
        history.update_status("GHOST", "escalated")   # must not raise
        history.update_verified("GHOST", True)        # must not raise
        self.assertEqual(self.raw("SELECT status, verified FROM inspections"),
                         [("pending", None)])


if __name__ == "__main__":
    unittest.main()
