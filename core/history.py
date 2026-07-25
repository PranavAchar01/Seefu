"""Sqlite log of every inspection, for shift stats and batch reasoning.

Table matches the spec exactly:
  inspections(id, ts, dish_id, verdict, designator, defect, score, status, verified)
Query helpers: shift_summary(), defect_history(designator).
"""

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "runs/history.db"


def _conn():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        dish_id TEXT NOT NULL,
        verdict TEXT NOT NULL,
        designator TEXT,
        defect TEXT,
        score REAL,
        status TEXT,
        verified INTEGER
    )""")
    return con


def log_inspection(dish_id, verdict, designator, defect, score, status):
    con = _conn()
    with con:
        cur = con.execute(
            "INSERT INTO inspections (ts, dish_id, verdict, designator, defect, score, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), dish_id, verdict, designator, defect,
             float(score) if score is not None else None, status))
    con.close()
    return cur.lastrowid


def update_status(dish_id, status):
    con = _conn()
    with con:
        con.execute("UPDATE inspections SET status=? WHERE dish_id=?", (status, dish_id))
    con.close()


def update_verified(dish_id, verified):
    con = _conn()
    with con:
        con.execute("UPDATE inspections SET verified=? WHERE dish_id=?",
                    (1 if verified else 0, dish_id))
    con.close()


def shift_summary():
    """Counts + top failing designators, shaped for the dashboard shift message."""
    con = _conn()
    inspected = con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    autonomous = con.execute(
        "SELECT COUNT(*) FROM inspections WHERE status='autonomous_resolution'").fetchone()[0]
    escalated = con.execute(
        "SELECT COUNT(*) FROM inspections WHERE status IN ('escalated','quarantined')").fetchone()[0]
    top = [{"designator": d, "count": c} for d, c in con.execute(
        "SELECT designator, COUNT(*) c FROM inspections "
        "WHERE verdict='defect' AND designator IS NOT NULL "
        "GROUP BY designator ORDER BY c DESC LIMIT 3")]
    con.close()
    return {"inspected": inspected, "autonomous": autonomous, "escalated": escalated, "top": top}


def defect_history(designator):
    """All defect rows for one designator, newest first."""
    con = _conn()
    rows = [{"ts": ts, "dish_id": b, "defect": d, "score": s, "status": st}
            for ts, b, d, s, st in con.execute(
                "SELECT ts, dish_id, defect, score, status FROM inspections "
                "WHERE verdict='defect' AND designator=? ORDER BY ts DESC, id DESC",
                (designator,))]
    con.close()
    return rows
