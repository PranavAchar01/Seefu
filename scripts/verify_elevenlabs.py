"""Prove the ElevenLabs leg of the operator call, end to end, without Twilio.

Runs the REAL integration path core/phone.py uses on a live defect:
  1. key + account reachable (GET /v1/models, and confirm the configured model)
  2. build the spoken diagnosis from a realistic defect result
  3. synthesize it through synthesize_alert_audio() - a real POST to
     /v1/text-to-speech/{voice_id} - and check the bytes are actually an mp3

Prints the resolved text, model id, voice id and byte size. Exits non-zero with
a clear reason if the key is missing or the API errors, so `make verify-voice`
is a hard pass/fail before the demo.

Usage:  .venv/bin/python scripts/verify_elevenlabs.py  [--keep]
        --keep leaves the mp3 in runs/alerts/ instead of using a temp case id
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import httpx  # noqa: E402

from core import phone  # noqa: E402

CASE_ID = "verify" if "--keep" in sys.argv else "verifytmp"

# a realistic defect: two findings, one mapped to a component, one only located
# by the anomaly map - the case the spoken text has to handle gracefully
RESULT = {
    "dish_id": "DISH-005",
    "case_id": CASE_ID,
    "verdict": "defect",
    "score": 51.2,
    "threshold": 44.4,
    "findings": [
        {"designator": "J1", "component": "4-pin power header",
         "location": "upper left", "score": 51.2, "source": "component map + vision",
         "description": "Pins 2 and 3 are bent inward and touching each other"},
        {"designator": None, "component": None, "location": "lower right",
         "score": 47.9, "source": "anomaly map",
         "description": "Anomalous region at (0.71, 0.82)"},
    ],
}

results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")


def is_mp3(data):
    """ID3 tag or an MPEG frame sync - i.e. audio, not a JSON error page."""
    return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF
                                  and (data[1] & 0xE0) == 0xE0)


key = os.environ.get("ELEVENLABS_API_KEY")
voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or phone.ELEVEN_DEFAULT_VOICE
print(f"model={phone.eleven_model()}  voice={voice_id}  "
      f"format={phone.eleven_output_format()}")
print(f"voice_settings={phone.ELEVEN_VOICE_SETTINGS}\n")

if not key:
    print("FAIL  ELEVENLABS_API_KEY is not set.")
    print("      Put ELEVENLABS_API_KEY=... in .env (python-dotenv loads it), "
          "then re-run.")
    print("      Without it the operator call still happens, but on Twilio's "
          "built-in robot voice.")
    sys.exit(2)

print("[1/3] account + model availability (GET /v1/models)...")
try:
    r = httpx.get("https://api.elevenlabs.io/v1/models",
                  headers={"xi-api-key": key}, timeout=20.0)
    if r.status_code != 200:
        report("models", False, phone._tts_http_reason(r))
    else:
        ids = [m.get("model_id") for m in r.json()]
        have = phone.eleven_model() in ids
        report("models", have,
               f"{len(ids)} models available; {phone.eleven_model()} "
               f"{'is enabled on this key' if have else 'NOT in ' + str(ids)}")
except Exception as e:
    report("models", False, f"{type(e).__name__}: {e}")

print("[2/3] spoken diagnosis...")
text = phone.defect_alert_text(RESULT)
clean = ("upper left" in text and "lower right" in text
         and "J1" in text and "51.2" in text and "44.4" in text
         and "unmapped" not in text and "(0.71" not in text and "*" not in text)
report("diagnosis", clean,
       f"{len(text)} chars, {len(text.split())} words "
       f"(~{len(text.split()) / 2.6:.0f}s of speech)")
print("\n  ---- what the operator hears ----")
for line in text.split(". "):
    print(f"  {line.strip().rstrip('.')}.")
print("  ---------------------------------\n")

print("[3/3] real synthesis through core.phone.synthesize_alert_audio...")
# clear any cached mp3 for this text so this is a genuine API call
import hashlib  # noqa: E402
stamp = hashlib.sha1(text.encode()).hexdigest()[:8]
mp3 = ROOT / f"runs/alerts/{phone._safe_case(CASE_ID)}_{stamp}.mp3"
if mp3.exists():
    mp3.unlink()

import time  # noqa: E402
t0 = time.time()
out = phone.synthesize_alert_audio(text, CASE_ID)
elapsed = time.time() - t0
status = phone.tts_status()

if out is None:
    report("synthesis", False, f"{elapsed:.1f}s  {status['reason']}")
else:
    data = out.read_bytes()
    ok = len(data) > 2000 and is_mp3(data)
    report("synthesis", ok,
           f"{elapsed:.1f}s  {len(data)} bytes  mp3={is_mp3(data)}  "
           f"model={status['model']}  voice={status['voice_id']}  -> {out}")
    if ok:
        print(f"\n  play it:  afplay {out}")
    if CASE_ID == "verifytmp":
        out.unlink(missing_ok=True)
        phone._alert_text_file(CASE_ID).unlink(missing_ok=True)

if all(results):
    print("\nALL GREEN - the operator will hear the ElevenLabs voice read the "
          "real diagnosis.")
    sys.exit(0)
print("\nFAILED - the call would fall back to Twilio's built-in voice. See above.")
sys.exit(1)
