"""Define the fixed ingest crop by dragging a rectangle on a current frame.

Usage:
  python scripts/set_crop.py            # uses the newest raw frame on disk
  python scripts/set_crop.py --live     # grabs a fresh frame from the camera
  python scripts/set_crop.py --show     # preview the currently saved crop

Drag with the mouse; adjust by dragging again. ENTER saves, q quits unsaved.
Include the dish plus ~1 cm of mat margin on all sides (reseat wiggle room,
the dish must never exit the crop). Exclude all wood.
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.frames import SENSOR_H, SENSOR_W, load_crop, save_crop  # noqa: E402

state = {"drag": False, "x0": 0, "y0": 0, "rect": None}


def on_mouse(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.update(drag=True, x0=x, y0=y, rect=None)
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
        state["rect"] = (min(state["x0"], x), min(state["y0"], y),
                         abs(x - state["x0"]), abs(y - state["y0"]))
    elif event == cv2.EVENT_LBUTTONUP:
        state["drag"] = False


def newest_raw_frame():
    for folder in ("data/normal_raw", "data/normal", "data/test_shots"):
        frames = sorted((ROOT / folder).glob("*.png")) if (ROOT / folder).exists() else []
        # only raw sensor-sized frames are valid canvases for a sensor-space crop
        for p in reversed(frames):
            img = cv2.imread(str(p))
            if img is not None and img.shape[1] == SENSOR_W and img.shape[0] == SENSOR_H:
                return img, p
    sys.exit("no raw 1920x1080 frame found; run with --live or capture one first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="grab a fresh camera frame")
    ap.add_argument("--show", action="store_true", help="preview the saved crop and exit")
    ap.add_argument("--view", type=float, default=0.6, help="window scale")
    args = ap.parse_args()

    if args.live:
        from capture.capture import lock_settings, open_camera
        cap = open_camera(0, SENSOR_W, SENSOR_H)
        lock_settings(cap)
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("camera grab failed")
        source = "live"
    else:
        frame, source = newest_raw_frame()

    view = args.view
    existing = load_crop()
    if existing and (args.show or True):     # always show existing rect as a start
        x, y, w, h = existing
        state["rect"] = (int(x * view), int(y * view), int(w * view), int(h * view))
    if args.show and not existing:
        sys.exit("no crop saved yet")

    win = "Seefu set_crop  (drag rect | ENTER save | q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    small = cv2.resize(frame, None, fx=view, fy=view)
    print(f"frame source: {source}. Drag dish + ~1cm mat margin, no wood. ENTER saves.")

    while True:
        display = small.copy()
        if state["rect"]:
            x, y, w, h = state["rect"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (80, 200, 255), 2)
            cv2.putText(display, f"{int(w/view)}x{int(h/view)} px", (x + 6, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("quit without saving")
            break
        if key in (13, 10) and state["rect"] and not args.show:
            x, y, w, h = state["rect"]
            saved = save_crop(x / view, y / view, w / view, h / view, note="human-defined")
            print(f"saved crop x,y,w,h = {saved} -> data/crop.json")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
