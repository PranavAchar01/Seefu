"""Shared frame-ingest geometry: THE single place the fixed crop lives.

The crop is (x, y, w, h) in raw sensor pixels, stored in data/crop.json so the
venue redefinition tomorrow is one run of scripts/set_crop.py, no code edit.
Every frame enters the pipeline through ingest(): capture.py saves, live grabs
in serve.py, uploads, and calibration reads. Frames that are not sensor-sized
are assumed to be already cropped and pass through untouched.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROP_FILE = ROOT / "data/crop.json"

SENSOR_W, SENSOR_H = 1920, 1080


def load_crop():
    """Returns (x, y, w, h) or None when no crop is defined.
    A crop smaller than 50px per side is never legitimate (a corrupted file
    once cropped every frame to 1x1) - refuse it loudly instead of obeying."""
    if not CROP_FILE.exists():
        return None
    data = json.loads(CROP_FILE.read_text())
    crop = tuple(int(v) for v in (data["x"], data["y"], data["w"], data["h"]))
    if crop[2] < 50 or crop[3] < 50:
        if "degenerate" not in _warned_sizes:
            _warned_sizes.add("degenerate")
            print(f"WARNING: {CROP_FILE} holds a degenerate rect {crop} - "
                  "IGNORING it (no crop applied). Re-run scripts/set_crop.py.")
        return None
    return crop


def save_crop(x, y, w, h, note=""):
    x = min(max(0, int(x)), SENSOR_W - 1)
    y = min(max(0, int(y)), SENSOR_H - 1)
    w = max(1, min(int(w), SENSOR_W - x))
    h = max(1, min(int(h), SENSOR_H - y))
    CROP_FILE.write_text(json.dumps(
        {"x": x, "y": y, "w": w, "h": h, "note": note}, indent=2) + "\n")
    return x, y, w, h


_warned_sizes = set()


def ingest(frame):
    """Apply the fixed crop to a raw sensor frame; pass through anything else."""
    crop = load_crop()
    if crop is None:
        return frame
    h, w = frame.shape[:2]
    if (w, h) != (SENSOR_W, SENSOR_H):
        # already cropped, or a DIFFERENT camera resolution (venue webcam!) -
        # warn loudly once per size so a silent crop-skip can't poison the bank
        if (w, h) != (crop[2], crop[3]) and (w, h) not in _warned_sizes:
            _warned_sizes.add((w, h))
            print(f"WARNING: frame {w}x{h} matches neither the sensor "
                  f"({SENSOR_W}x{SENSOR_H}) nor the crop {crop[2]}x{crop[3]} - "
                  "passing through UNCROPPED. New camera? Re-run scripts/set_crop.py "
                  "and update SENSOR_W/H in core/frames.py.")
        return frame
    x, y, cw, ch = crop
    return frame[y:y + ch, x:x + cw]
