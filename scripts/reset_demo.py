"""Reset runtime demo state (history, cases, counters) and re-seed the J1 beat.

Run before rehearsals and before the venue demo:
  .venv/bin/python scripts/reset_demo.py
Does NOT touch the model, calibration, crop, or captured frames.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for target in ("runs/history.db", "runs/.dish_seq", "runs/.case_seq"):
    path = ROOT / target
    if path.exists():
        path.unlink()
        print(f"removed {target}")
cases = ROOT / "runs/cases"
if cases.exists():
    shutil.rmtree(cases)
    print("removed runs/cases/")
# case ids restart at 1041, so last rehearsal's alert audio + spoken text would
# otherwise be reused for a different dish on the next defect call
alerts = ROOT / "runs/alerts"
if alerts.exists():
    shutil.rmtree(alerts)
    print("removed runs/alerts/  (ElevenLabs alert audio + spoken text)")

subprocess.run([sys.executable, str(ROOT / "scripts/seed_history.py")], check=True)
print("demo state reset: dish seq restarts at DISH-004, cases at 1041")
