"""Seed two prior plating failures (plates DISH-002 and -003, pass line A) so a
live third failure reads as a pattern, in sqlite and in the kitchen memory.
Seeding prior world-state is legitimate; the reasoning over it is live.

With XTRACE_API_KEY set, the same two failures are also ingested into XTrace,
so the memory panel's recall has genuine history to match against.

Idempotent in sqlite: re-running replaces the seed rows instead of duplicating
them. XTrace ingestion appends; run it once per demo reset.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv                      # noqa: E402
load_dotenv(ROOT / ".env")

from core import history, memory  # noqa: E402

# dated TODAY, a few hours back, so "this shift" stays true whenever it runs
TODAY = time.strftime("%Y-%m-%d")
SEEDS = [
    (f"{TODAY} 09:12:31", "DISH-002", "defect", "centre",
     "Scallion garnish missing from the beef.", 52.4, "escalated", 1),
    (f"{TODAY} 10:47:02", "DISH-003", "defect", "centre",
     "Scallion garnish missing, beef unseasoned in the centre.", 55.1, "escalated", 1),
]


def main():
    con = history._conn()
    with con:
        for ts, dish_id, verdict, zone, defect, score, status, verified in SEEDS:
            con.execute("DELETE FROM inspections WHERE dish_id=?", (dish_id,))
            con.execute(
                "INSERT INTO inspections (ts, dish_id, verdict, designator, defect, "
                "score, status, verified) VALUES (?,?,?,?,?,?,?,?)",
                (ts, dish_id, verdict, zone, defect, score, status, verified))
    con.close()
    rows = history.defect_history("centre")
    print(f"seeded; centre-zone defect history now has {len(rows)} rows:")
    for r in rows:
        print(f"  {r['ts']}  {r['dish_id']}  {r['defect']}")

    if memory.enabled():
        for ts, dish_id, verdict, zone, defect, score, status, _ in SEEDS:
            memory.record_inspection({
                "dish_id": dish_id, "verdict": verdict, "score": score,
                "threshold": 49.54,
                "findings": [{"location": zone, "description": defect,
                              "fix": "re-garnish with scallion and send"}],
            })
        print("ingesting the two seed failures into XTrace (async, a few seconds)...")
        memory.wait_for_ingest(3)
    else:
        print("XTRACE_API_KEY not set; sqlite seeded only")


if __name__ == "__main__":
    main()
