"""M5 websocket contract test.

Boots the real server, connects a websocket client like the dashboard does,
drives one clean and one defect inspection through /inspect, and asserts the
EXACT message shapes from the spec (section M5) - required keys present, no
stray keys beyond the spec's optionals, enums respected.

Run: .venv/bin/python tests/test_ws_contract.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
PORT = 8001
BASE = f"http://localhost:{PORT}"

SHAPES = {
    "context": {"required": {"type", "dish", "batch", "quadrant", "components", "bank",
                             "calibration"}, "optional": set()},
    "inspection_start": {"required": {"type", "dish_id"}, "optional": set()},
    "inspection_result": {"required": {"type", "dish_id", "verdict", "score", "threshold",
                                       "blobs", "findings"}, "optional": {"image_b64"}},
    "case": {"required": {"type", "case_id", "status", "verified"},
             "optional": {"zendesk_id", "note"}},
    "shift": {"required": {"type", "inspected", "autonomous", "escalated", "top"},
              "optional": set()},
    "agent": {"required": {"type", "state"}, "optional": {"text"}},
    "event": {"required": {"type", "kind", "source", "message"},
              "optional": {"ts", "dish_id"}},
}
CASE_STATUSES = {"autonomous_resolution", "escalated", "quarantined", "resolved"}

failures = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        failures.append(label)


def assert_shape(msg, kind):
    keys = set(msg.keys())
    spec = SHAPES[kind]
    missing = spec["required"] - keys
    stray = keys - spec["required"] - spec["optional"]
    check(not missing, f"{kind}: no missing keys {missing or ''}")
    check(not stray, f"{kind}: no stray keys {stray or ''}")


async def main():
    messages = []

    async with websockets.connect(f"ws://localhost:{PORT}/ws") as ws:

        async def collect():
            async for raw in ws:
                messages.append(json.loads(raw))

        task = asyncio.create_task(collect())

        clean = sorted((ROOT / "data/normal").glob("*.png"))[-1]
        # synthesize the defect frame here rather than pointing at a scratch file
        # on one laptop: the test has to run on any checkout
        synth_img = cv2.imread(str(clean))
        sh, sw = synth_img.shape[:2]
        cv2.circle(synth_img, (sw // 3, sh // 3), int(0.045 * sw), (150, 150, 155), -1)
        cv2.line(synth_img, (sw // 3 - 60, sh // 3 - 60), (sw // 3 + 60, sh // 3 + 60),
                 (25, 25, 30), 9)
        synth_bytes = cv2.imencode(".png", synth_img)[1].tobytes()

        async with httpx.AsyncClient(timeout=120) as client:
            for name, payload in (("clean.png", clean.read_bytes()),
                                  ("synth_defect.png", synth_bytes)):
                r = await client.post(f"{BASE}/inspect",
                                      files={"file": (name, payload, "image/png")})
                check(r.status_code == 200, f"POST /inspect {name} -> 200")
        await asyncio.sleep(2)
        task.cancel()

    by_type = {}
    for m in messages:
        by_type.setdefault(m.get("type"), []).append(m)
    print(f"\ncollected {len(messages)} ws messages: "
          f"{ {k: len(v) for k, v in by_type.items()} }")

    check(messages and messages[0]["type"] == "context", "first message is context")
    for kind in ("context", "inspection_start", "inspection_result", "case", "shift"):
        check(kind in by_type, f"{kind} emitted")
        for m in by_type.get(kind, []):
            assert_shape(m, kind)

    results = by_type.get("inspection_result", [])
    check(len(results) == 2, "two inspection_results")
    if len(results) == 2:
        clean_r, defect_r = results
        for r in results:
            check(0.0 <= r["score"] <= 1.0 and r["threshold"] == 0.5,
                  f"ws score normalized to 0..1 (got {r['score']} / {r['threshold']})")
        check(clean_r["verdict"] == "pass", "clean frame verdict pass")
        check(clean_r["findings"] == [], "clean frame has no findings")
        check(defect_r["verdict"] == "defect", "defect frame verdict defect")
        check(len(defect_r["blobs"]) >= 1, "defect frame has blobs")
        check(len(defect_r["findings"]) >= 1, "defect frame has findings")
        for f in defect_r["findings"]:
            check(0.0 <= f["score"] <= 1.0, "ws finding score normalized")
        for b in defect_r["blobs"]:
            check(set(b) == {"cx", "cy", "r", "score"}, "blob shape {cx,cy,r,score}")
            check(0 <= b["cx"] <= 1 and 0 <= b["cy"] <= 1, "blob coords normalized")
        for f in defect_r["findings"]:
            check(set(f) == {"designator", "component", "description", "score",
                             "source", "location"},
                  "finding shape {designator,component,description,score,source,location}")
            check(f["location"], "finding carries a geometric location label")
        check(defect_r.get("image_b64"), "defect result carries image_b64")

    case_msgs = by_type.get("case", [])
    for m in case_msgs:
        check(m["status"] in CASE_STATUSES, f"case status valid: {m['status']}")
    # one create-emit per case; re-emits allowed ONLY to deliver verified=True
    check(len({m["case_id"] for m in case_msgs}) == 2, "exactly two cases (one per dish)")
    seen_ids = set()
    for m in case_msgs:
        if m["case_id"] in seen_ids:
            check(m["verified"] is True,
                  f"case re-emit only carries verified=True (case {m['case_id']})")
        seen_ids.add(m["case_id"])
    check(not any(m.get("note") for m in case_msgs
                  if m["status"] == "autonomous_resolution" and not m["verified"]),
          "no note (false VERIFIED badge) on unverified pass cases")
    shifts = by_type.get("shift", [])
    if shifts:
        check(isinstance(shifts[-1]["top"], list), "shift.top is a list")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    # vision gate off: the synthetic screw's real-API classification is
    # nondeterministic; the contract under test is message shapes, not vision
    env = dict(os.environ, SEEFU_VISION_GATE="0", SEEFU_CALL_ON_DEFECT="0")
    server = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "inspection/serve.py"),
         "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(90):
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            sys.exit("server never came up")
        sys.exit(asyncio.run(main()))
    finally:
        server.terminate()
