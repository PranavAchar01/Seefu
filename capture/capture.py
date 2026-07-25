"""Seefu capture tool.

Builds the normal set and grabs test shots with LOCKED camera settings.

Usage:
  python capture/capture.py                 # interactive: SPACE saves, q quits
  python capture/capture.py --burst 50      # 50 shots, 2 s countdown between each
  python capture/capture.py --dir data/test_defect
  python capture/capture.py --stats data/normal   # brightness stats over a folder
  python capture/capture.py --smoke         # headless: grab 10 frames, print stats

Interactive keys:
  SPACE save   q quit   p print current props
  e/E exposure -/+   f/F focus -/+   w/W wb temp -/+   g/G gain -/+
On quit the tool prints the final prop values; paste them into DEFAULTS below
so every later run reuses the tuned, locked settings.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.frames import ingest, load_crop  # noqa: E402

# Tuned on the rig: after first-run tuning, paste the printed values here.
# None means "do not set this prop" (leave the camera's own default).
DEFAULTS = {
    "CAP_PROP_AUTO_EXPOSURE": 0.0,   # 0 = manual/off where the backend honors it
    "CAP_PROP_AUTO_WB": 0.0,
    "CAP_PROP_AUTOFOCUS": 0.0,
    "CAP_PROP_EXPOSURE": None,
    "CAP_PROP_FOCUS": None,
    "CAP_PROP_WB_TEMPERATURE": None,
    "CAP_PROP_GAIN": None,
}

TUNABLE = ["CAP_PROP_EXPOSURE", "CAP_PROP_FOCUS", "CAP_PROP_WB_TEMPERATURE", "CAP_PROP_GAIN"]
STEP = {"CAP_PROP_EXPOSURE": 1.0, "CAP_PROP_FOCUS": 5.0, "CAP_PROP_WB_TEMPERATURE": 100.0, "CAP_PROP_GAIN": 1.0}

# Framing guide, normalized (x0, y0, x1, y1). Measured off the rig framing
# 2026-07-23: dish occupies ~x 7-88%, y 2-100% of frame. Nudge with --guide.
# With --crop, saved frames are trimmed to this rect — every later capture
# (test shots, live /inspect grabs) must then use the SAME rect.
GUIDE = (0.07, 0.02, 0.88, 1.00)


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}. On macOS grant Camera permission to your terminal "
                 "(System Settings > Privacy & Security > Camera), then retry.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def lock_settings(cap):
    """Apply DEFAULTS and report what the backend actually accepted."""
    print("Locking camera settings:")
    for name, value in DEFAULTS.items():
        if value is None:
            continue
        prop = getattr(cv2, name)
        ok = cap.set(prop, value)
        readback = cap.get(prop)
        print(f"  {name} = {value}  set()={'ok' if ok else 'REJECTED'}  readback={readback}")
    print("  (a REJECTED prop means this backend/camera ignores it; on macOS built-in "
        "cameras that is common - an external UVC webcam honors more of these)")


def print_props(cap):
    print("Current props (paste into DEFAULTS to hardcode):")
    print("{")
    for name in DEFAULTS:
        print(f'    "{name}": {cap.get(getattr(cv2, name))},')
    print("}")


def mean_brightness(frame):
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def save_frame(frame, out_dir):
    frame = ingest(frame)                  # fixed crop from data/crop.json, if defined
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.png"
    cv2.imwrite(str(path), frame)
    print(f"saved {path}  mean_brightness={mean_brightness(frame):.2f}")
    return path


def crop_frame(frame, guide):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = guide
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def show(img, view):
    if view != 1.0:
        img = cv2.resize(img, None, fx=view, fy=view)
    cv2.imshow("Seefu capture", img)


def draw_hud(frame, saved_count, note="", guide=GUIDE, cropping=False):
    h, w = frame.shape[:2]
    # fixed ingest crop, if defined: saved frames contain exactly this region
    crop = load_crop()
    if crop is not None and (w, h) == (1920, 1080):
        cx, cy, cw, ch = crop
        cv2.rectangle(frame, (cx, cy), (cx + cw - 1, cy + ch - 1), (255, 200, 60), 2)
        cv2.putText(frame, "INGEST CROP", (cx + 8, cy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 200, 60), 2, cv2.LINE_AA)
    # framing guide: dish should fill this rectangle
    x0, y0 = int(w * guide[0]), int(h * guide[1])
    x1, y1 = int(w * guide[2]) - 1, int(h * guide[3]) - 1
    color = (60, 160, 255) if cropping else (80, 200, 120)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    if cropping:
        cv2.putText(frame, "CROP", (x0 + 8, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2, cv2.LINE_AA)
    # big frame counter
    cv2.putText(frame, str(saved_count), (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                3.5, (255, 255, 255), 6, cv2.LINE_AA)
    if note:
        cv2.putText(frame, note, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (80, 200, 255), 2, cv2.LINE_AA)
    return frame


def nudge(cap, name, delta):
    prop = getattr(cv2, name)
    current = cap.get(prop)
    target = current + delta
    ok = cap.set(prop, target)
    print(f"{name}: {current} -> {target}  set()={'ok' if ok else 'REJECTED'}  readback={cap.get(prop)}")


def interactive(cap, out_dir, guide, view):
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break
        display = draw_hud(frame.copy(), saved, f"SPACE save -> {out_dir}   q quit", guide)
        show(display, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            save_frame(frame, out_dir)
            saved += 1
        elif key == ord("p"):
            print_props(cap)
        elif key in (ord("e"), ord("E"), ord("f"), ord("F"), ord("w"), ord("W"), ord("g"), ord("G")):
            name = {"e": "CAP_PROP_EXPOSURE", "f": "CAP_PROP_FOCUS",
                    "w": "CAP_PROP_WB_TEMPERATURE", "g": "CAP_PROP_GAIN"}[chr(key).lower()]
            sign = 1.0 if chr(key).isupper() else -1.0
            nudge(cap, name, sign * STEP[name])
    print_props(cap)
    print(f"guide = {guide}  (override with --guide {','.join(str(v) for v in guide)})")
    cv2.destroyAllWindows()


def burst(cap, out_dir, n, guide, view):
    saved = 0
    for i in range(n):
        deadline = time.time() + 2.0
        while time.time() < deadline:          # live countdown, keeps preview fresh
            ok, frame = cap.read()
            if not ok:
                sys.exit("frame grab failed during countdown")
            remaining = deadline - time.time()
            display = draw_hud(frame.copy(), saved, f"shot {i + 1}/{n} in {remaining:.1f}s", guide)
            show(display, view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                return
        ok, frame = cap.read()
        if not ok:
            sys.exit("frame grab failed")
        save_frame(frame, out_dir)
        saved += 1
        print("\a", end="", flush=True)        # beep
        flash = np.full_like(frame, 255)       # on-screen flash
        show(draw_hud(flash, saved, "", guide), view)
        cv2.waitKey(120)
    cv2.destroyAllWindows()


def stats(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        sys.exit(f"no images in {folder}")
    values = []
    for p in files:
        b = mean_brightness(cv2.imread(str(p)))
        values.append(b)
        print(f"{p.name}  mean_brightness={b:.2f}")
    print(f"\n{len(values)} frames  mean={np.mean(values):.2f}  std={np.std(values):.3f}  "
          f"range={max(values) - min(values):.2f}")
    print("std under ~1.0 means exposure is locked tight; a big range means settings drifted.")


def smoke(cap):
    """Headless acceptance check: grab 10 frames, print brightness stats."""
    for _ in range(5):                          # warm-up, let exposure settle if unlocked
        cap.read()
    values = []
    for i in range(10):
        ok, frame = cap.read()
        if not ok:
            sys.exit("frame grab failed")
        b = mean_brightness(frame)
        values.append(b)
        print(f"frame {i + 1}/10  mean_brightness={b:.2f}")
    print(f"\nmean={np.mean(values):.2f}  std={np.std(values):.3f}  range={max(values) - min(values):.2f}")


def main():
    ap = argparse.ArgumentParser(description="Seefu webcam capture with locked settings")
    ap.add_argument("--dir", default="data/normal", help="output folder for saved frames")
    ap.add_argument("--burst", type=int, help="capture N frames with a 2 s countdown between each")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--stats", metavar="FOLDER", help="print brightness stats for a folder and exit")
    ap.add_argument("--smoke", action="store_true", help="headless 10-frame brightness check")
    ap.add_argument("--guide", help="framing rect x0,y0,x1,y1 normalized (default from GUIDE)")
    ap.add_argument("--view", type=float, default=1.0, help="preview window scale, e.g. 0.6")
    args = ap.parse_args()

    guide = tuple(float(v) for v in args.guide.split(",")) if args.guide else GUIDE
    if len(guide) != 4 or not all(0.0 <= v <= 1.0 for v in guide):
        sys.exit("--guide needs four normalized values: x0,y0,x1,y1")

    if args.stats:
        stats(args.stats)
        return

    cap = open_camera(args.cam, args.width, args.height)
    lock_settings(cap)
    actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"capture resolution: {int(actual[0])}x{int(actual[1])}")

    try:
        if args.smoke:
            smoke(cap)
        elif args.burst:
            burst(cap, Path(args.dir), args.burst, guide, args.view)
        else:
            interactive(cap, Path(args.dir), guide, args.view)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
