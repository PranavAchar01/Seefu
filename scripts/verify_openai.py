"""One-command OpenAI re-verification for the morning (after fixing billing).

Tests, with real API calls:
  1. gpt-4o vision description of a defect crop (M4 happy path)
  2. case verifier on the newest real case (M6 happy path)
  3. Realtime ephemeral token mint (M7)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inspection"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import cv2  # noqa: E402

results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")


print("[1/3] vision description (gpt-4o)...")
from explain import CACHE_FILE, crop_blob, describe_crop  # noqa: E402
if CACHE_FILE.exists():
    CACHE_FILE.unlink()                      # force a real call, not a cache hit
frames = sorted((ROOT / "data/normal").glob("*.png"))
ref = cv2.imread(str(frames[-1]))
frame = ref.copy()
h, w = frame.shape[:2]
cx, cy, r = int(w * 0.5), int(h * 0.5), 40   # synthetic foreign object at center
cv2.circle(frame, (cx, cy), r, (200, 120, 40), -1)
blob = {"cx": 0.5, "cy": 0.5, "r": r / max(w, h)}
t0 = time.time()
text, fix, cosmetic, how = describe_crop(crop_blob(frame, blob), crop_blob(ref, blob),
                                         "FALLBACK")
report("vision", how.startswith("vision"),
       f"{time.time()-t0:.1f}s  source={how}  text={text!r}  fix={fix!r}")

print("[2/3] case verifier...")
from verify import verify_case  # noqa: E402
cases = sorted((ROOT / "runs/cases").glob("*.json"))
if cases:
    t0 = time.time()
    verified, note = verify_case(cases[-1].stem)
    ok = not note.startswith("verification unavailable")
    report("verifier", ok, f"{time.time()-t0:.1f}s  verified={verified}  note={note!r}")
else:
    report("verifier", False, "no cases on disk - run an inspection first")

print("[3/3] Realtime token mint...")
try:
    from openai import OpenAI
    t0 = time.time()
    secret = OpenAI().realtime.client_secrets.create(
        expires_after={"anchor": "created_at", "seconds": 60},
        session={"type": "realtime", "model": "gpt-realtime-2.1"})
    report("token mint", secret.value.startswith("ek_"),
           f"{time.time()-t0:.1f}s  value={secret.value[:12]}...")
except Exception as e:
    report("token mint", False, f"{type(e).__name__}: {e}")

print(f"\n{'ALL GREEN - the demo beats are unblocked.' if all(results) else 'Something is still failing - check billing / see above.'}")
sys.exit(0 if all(results) else 1)
