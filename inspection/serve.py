"""Seefu inspection service.

FastAPI app:
  POST /inspect      multipart image upload, or ?live=1 to grab a frame with the
                     capture module's locked settings
  GET  /health
Script mode:
  python inspection/serve.py --calibrate
                     scores every image in data/normal plus data/test_defect/,
                     prints both distributions sorted, proposes a midpoint
                     threshold and writes it into data/calibration.json

Scores are RAW PatchCore patch distances (unbounded, continuous), judged against
the calibrated threshold - not the model's internal 0..1 normalization, whose
normal-only fit clamps hard at 0.5/1.0 and hides separation.
Verified against installed anomalib 2.5.1 / torch 2.13:
Patchcore.load_from_checkpoint needs weights_only=False; anomaly map comes out
at pre-processor resolution and is upsampled here to the full frame.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# running as a script names this module __main__; alias it so agent/tools.py's
# `from serve import ...` binds THIS instance instead of loading a second copy
# (a second copy would re-load the PatchCore model and split all module state)
if "serve" not in sys.modules:
    sys.modules["serve"] = sys.modules[__name__]

from dotenv import load_dotenv                      # noqa: E402
load_dotenv(ROOT / ".env")

from capture.capture import open_camera, lock_settings   # noqa: E402
from core.frames import ingest                            # noqa: E402
from core import history, hub, memory                     # noqa: E402
from core.casesink import get_sink                        # noqa: E402

# demo unit labels (context panel); one plated dish per frame at the pass
DISH_NAME = "Sesame Beef Bowl"
BATCH = "PASS LINE A"
QUADRANT = "Full plate"

CKPT = ROOT / "results/Patchcore/dish/latest/weights/lightning/model.ckpt"
CALIB = ROOT / "data/calibration.json"
RUNS = ROOT / "runs"
DISH_SEQ = RUNS / ".dish_seq"

_model = None
_model_mtime = None


def get_model():
    """Loads once, and reloads automatically if the checkpoint changes on disk
    (e.g. `make bank` re-run at the venue while the server stays up)."""
    global _model, _model_mtime
    if not CKPT.exists():
        raise RuntimeError(f"no model at {CKPT}; run `make bank` first")
    mtime = CKPT.stat().st_mtime
    if _model is None or mtime != _model_mtime:
        from anomalib.models import Patchcore
        _model = Patchcore.load_from_checkpoint(
            str(CKPT), map_location="cpu", weights_only=False).eval()
        _model_mtime = mtime
    return _model


def model_fallback_threshold(model):
    """The adaptive threshold the post-processor learned from the normal val split."""
    pp = getattr(model, "post_processor", None)
    for name in ("image_threshold", "_image_threshold"):
        value = getattr(pp, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def read_threshold():
    if CALIB.exists():
        data = json.loads(CALIB.read_text())
        if "threshold" in data:
            return float(data["threshold"]), "calibration.json"
    fallback = model_fallback_threshold(get_model())
    if fallback is not None:
        return fallback, "model adaptive (UNCALIBRATED - run --calibrate)"
    raise RuntimeError("no threshold available; run `python inspection/serve.py --calibrate`")


def score_image(bgr):
    """Returns (raw_score, full_res_anomaly_map)."""
    import torch
    import torch.nn.functional as F
    model = get_model()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        processed = model.pre_processor(x)      # Resize 320 + ImageNet norm, from the ckpt
        raw = model.model(processed)            # InferenceBatch: raw score + raw map
    amap = raw.anomaly_map
    if amap.dim() == 3:
        amap = amap.unsqueeze(1)
    h, w = bgr.shape[:2]
    full = F.interpolate(amap, size=(h, w), mode="bilinear", align_corners=False)
    return float(raw.pred_score.item()), full[0, 0].cpu().numpy()


def extract_blobs(full_map, threshold, verdict):
    """Thresholded anomaly-map connected components, normalized 0..1 coords."""
    mask = (full_map >= threshold).astype(np.uint8)
    if verdict == "defect" and mask.sum() == 0:
        # image-level score tripped but no pixel cleared the bar: take the hottest tail
        mask = (full_map >= np.quantile(full_map, 0.998)).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = full_map.shape
    blobs = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:                            # speckle
            continue
        cx, cy = centroids[i]
        blobs.append({
            "cx": round(float(cx) / w, 4),
            "cy": round(float(cy) / h, 4),
            "r": round(float(np.sqrt(area / np.pi)) / max(w, h), 4),
            "score": round(float(full_map[labels == i].max()), 4),
        })
    blobs.sort(key=lambda b: -b["score"])
    if verdict == "defect" and not blobs:
        # verdict says defect but every component died to the speckle filter:
        # guarantee one blob at the hottest pixel so the UI never contradicts itself
        iy, ix = np.unravel_index(int(np.argmax(full_map)), full_map.shape)
        blobs = [{"cx": round(ix / w, 4), "cy": round(iy / h, 4), "r": 0.02,
                  "score": round(float(full_map.max()), 4)}]
    return blobs[:5]


def dashed_circle(img, center, radius, color):
    for start in range(0, 360, 24):
        cv2.ellipse(img, center, (radius, radius), 0, start, start + 12, color, 2, cv2.LINE_AA)


def pixel_stats(image_threshold):
    """Anomalib's dataset-anchored pixel stats from the trained post-processor:
    (pixel_threshold, pixel_min, pixel_max), with graceful fallbacks when a
    buffer never fitted (normal-only training leaves _pixel_threshold NaN)."""
    import math
    pp = None
    try:
        pp = getattr(get_model(), "post_processor", None)
    except RuntimeError:
        pass

    def finite(name):
        value = getattr(pp, name, None)
        try:
            f = float(value)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    thr = finite("_pixel_threshold") or image_threshold
    pmin, pmax = finite("pixel_min"), finite("pixel_max")
    if pmin is None or pmax is None or pmax <= pmin:
        pmin, pmax = 0.0, max(2.0 * image_threshold, 1e-6)
    return thr, pmin, pmax


def save_overlay(bgr, full_map, blobs, threshold):
    """Plain image + red contours only (no heatmap, per operator preference).
    Contours are drawn ONLY around thresholded components that contain one of
    the surviving (non-cosmetic) blobs - the actual anomalous shapes, using the
    anomalib dataset-anchored pixel threshold."""
    thr_px, _, _ = pixel_stats(threshold)
    blend = bgr.copy()
    mask = (full_map >= thr_px).astype(np.uint8)
    h, w = mask.shape
    num, labels = cv2.connectedComponents(mask, 8)
    keep_ids = set()
    for b in blobs:
        cy, cx = min(int(b["cy"] * h), h - 1), min(int(b["cx"] * w), w - 1)
        if labels[cy, cx] > 0:
            keep_ids.add(labels[cy, cx])
    if keep_ids:
        keep_mask = np.isin(labels, list(keep_ids)).astype(np.uint8)
        contours, _ = cv2.findContours(keep_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, [c for c in contours if cv2.contourArea(c) >= 40.0],
                         -1, (61, 61, 255), 3)
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"overlay_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000) % 1000:03d}.png"
    cv2.imwrite(str(path), blend)
    return path


_board_seq_lock = None


def next_dish_id():
    global _board_seq_lock
    import threading
    if _board_seq_lock is None:
        _board_seq_lock = threading.Lock()
    with _board_seq_lock:                       # threadpool endpoints can race the file
        RUNS.mkdir(exist_ok=True)
        seq = int(DISH_SEQ.read_text()) if DISH_SEQ.exists() else 4   # seeds use -002/-003
        DISH_SEQ.write_text(str(seq + 1))
    return f"DISH-{seq:03d}"


def grab_live_frame():
    cap = open_camera(0, 1920, 1080)
    lock_settings(cap)
    for _ in range(5):                           # let the pipeline settle
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("camera grab failed")
    return frame


def build_findings(blobs, frame_w, frame_h):
    """One finding per anomalous blob. The description here is the fallback;
    the comparative vision pass replaces it (and adds the fix suggestion).
    Plates have no component map: WHERE comes from the anomaly-map geometry
    (label_locations), WHAT comes from vision."""
    findings = []
    for b in blobs:
        findings.append({
            "designator": None,
            "component": None,
            "description": f"Anomalous region at ({b['cx']:.2f}, {b['cy']:.2f})",
            "fix": None,
            "score": b["score"],
            "source": "anomaly map",
        })
    return findings


def run_inspection(bgr, dish_id=None):
    bgr = ingest(bgr)                     # fixed crop; no-op on already-cropped input
    score, full_map = score_image(bgr)
    threshold, threshold_source = read_threshold()
    verdict = "defect" if score >= threshold else "pass"
    # localization uses the anomalib pixel threshold (dataset-anchored), which
    # keeps regions tight instead of swallowing the frame on high-score images
    thr_px, _, _ = pixel_stats(threshold)
    blobs = extract_blobs(full_map, thr_px, verdict)
    coverage = round(float((full_map >= thr_px).mean()), 3)
    h, w = bgr.shape[:2]
    findings = build_findings(blobs, w, h) if verdict == "defect" else []
    cosmetic_regions = []
    if findings:
        from explain import explain_findings
        explain_findings(bgr, reference_frame()[0], blobs, findings)
        if os.environ.get("SEEFU_VISION_GATE", "1") != "0":
            # PatchCore only knows "different"; comparative vision decides
            # "matters". Cosmetic regions (stickers/labels/reflections/things
            # also on the reference) are suppressed from findings + contours.
            kept = [(b, f) for b, f in zip(blobs, findings) if not f.get("cosmetic")]
            cosmetic_regions = [f["description"] for f in findings if f.get("cosmetic")]
            blobs = [b for b, _ in kept]
            findings = [f for _, f in kept]
            if not findings and verdict == "defect":
                verdict = "pass"
    note = None
    if verdict == "pass" and cosmetic_regions:
        note = (f"{len(cosmetic_regions)} anomalous region(s) judged cosmetic "
                "(labels/stickers/lighting also present on the reference) - "
                "no physical damage found.")
    elif verdict == "defect" and coverage > 0.15:
        note = (f"{coverage:.0%} of the frame is anomalous - this looks like a "
                "different dish, a moved plate, or a recipe the memory bank was "
                "never trained on, not a local plating miss. Re-center the "
                "plate, or retrain with normals for this dish.")
    overlay_path = save_overlay(bgr, full_map, blobs, threshold)
    return {
        "dish_id": dish_id or next_dish_id(),
        "score": round(score, 4),
        "threshold": round(threshold, 4),
        "threshold_source": threshold_source,
        "verdict": verdict,
        "anomaly_coverage": coverage,
        "note": note,
        "cosmetic_regions": cosmetic_regions,
        "blobs": blobs,
        "findings": findings,
        "overlay_path": str(overlay_path.relative_to(ROOT)),
        "overlay_url": f"/overlays/{overlay_path.name}",
    }


_reference = None


def reference_frame():
    """The golden plate (data/reference.png, falling back to the first normal)
    + its bowl bbox: the canvas uploads are composited onto. The bowl is found
    by color distance from the border background, which survives any backdrop
    (checkerboard cutout, counter, tray). Cached after first use."""
    global _reference
    if _reference is None:
        ref_path = ROOT / "data/reference.png"
        if not ref_path.exists():
            frames = sorted((ROOT / "data/normal").glob("*.png"))
            if not frames:
                raise RuntimeError("no data/reference.png and data/normal is empty; "
                                   "add the normal set first")
            ref_path = frames[0]
        img = cv2.imread(str(ref_path))
        h, w = img.shape[:2]
        border = np.concatenate([img[:15].reshape(-1, 3), img[-15:].reshape(-1, 3),
                                 img[:, :15].reshape(-1, 3), img[:, -15:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(img.astype(np.int16) - bg.astype(np.int16), axis=2)
        m = (dist > 45).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bbox = tuple(int(stats[i, k]) for k in
                         (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                          cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        else:
            bbox = (0, 0, w, h)
        _reference = (img, bbox)
    return _reference


_mat_color = None


def training_mat_color():
    """Median border color of the normal set - the rig mat the bank knows."""
    global _mat_color
    if _mat_color is None:
        samples = []
        for p in sorted((ROOT / "data/normal").glob("*.png"))[:6]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            border = np.concatenate([img[:20].reshape(-1, 3), img[-20:].reshape(-1, 3),
                                     img[:, :20].reshape(-1, 3), img[:, -20:].reshape(-1, 3)])
            samples.append(np.median(border, axis=0))
        _mat_color = (np.median(samples, axis=0) if samples
                      else np.array([80.0, 77.0, 83.0]))
    return _mat_color


def register_to_reference(frame):
    """Golden-dish registration: ORB feature match + RANSAC homography warps
    the uploaded photo into pixel alignment with the reference frame (the dish
    is planar, so a homography is exact - handles any rotation, tilt, scale).
    Returns (aligned, H) or None when registration fails."""
    ref, _ = reference_frame()
    scale = min(1.0, 1600.0 / max(frame.shape[:2]))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    if len(matches) < 40:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches]) / scale
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inliers is None or int(inliers.sum()) < 30:
        return None
    h_ref, w_ref = ref.shape[:2]
    aligned = cv2.warpPerspective(frame, H, (w_ref, h_ref))
    # outside the warped photo there is no data - fill from the reference so
    # black borders never score as anomalies
    valid = cv2.warpPerspective(
        np.full(frame.shape[:2], 255, np.uint8), H, (w_ref, h_ref)) > 127
    out = ref.copy()
    out[valid] = aligned[valid]
    return out, H


def normalize_upload(frame):
    """Off-pass photos (foreign background, any orientation/scale) are recomposed
    to training conditions before inspection: register to the golden plate, or
    segment the dish from the background sampled at the borders and composite it
    onto the reference canvas. Without this, a counter or table background IS
    the anomaly and drowns the plate (the bank has only ever seen the training
    backdrop). Frames already at training geometry skip this.
    Returns (frame, was_normalized, transform)."""
    h, w = frame.shape[:2]
    ref, _ = reference_frame()
    ref_h, ref_w = ref.shape[:2]
    if (w, h) == (ref_w, ref_h):
        return frame, False, None

    # first choice: exact golden-dish registration; the composite path below
    # is the fallback when feature matching fails (blurry/occluded photos)
    registered = register_to_reference(frame)
    if registered is not None:
        aligned, H = registered
        return aligned, True, {"homography": H, "original": frame}

    border = np.concatenate([frame[:15].reshape(-1, 3), frame[-15:].reshape(-1, 3),
                             frame[:, :15].reshape(-1, 3), frame[:, -15:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(frame.astype(np.int16) - bg.astype(np.int16), axis=2)
    fg = (dist > 45).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if num < 2:
        return frame, False, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.15 * w * h:
        return frame, False, None                 # no dominant object found
    keep = (labels == i).astype(np.uint8)
    x, y, bw, bh = (int(stats[i, k]) for k in
                    (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    m = int(0.02 * max(bw, bh))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
    dish, mask = frame[y0:y1, x0:x1], keep[y0:y1, x0:x1]

    # the reference plate is square; only rotate when the crop's aspect clearly
    # disagrees with the reference (a plate shot is orientation-free otherwise)
    _, (_, _, rw0, rh0) = reference_frame()
    ref_portrait = rh0 > rw0 * 1.1
    was_portrait = dish.shape[0] > dish.shape[1] * 1.1 and not ref_portrait
    if was_portrait:
        dish = cv2.rotate(dish, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    # composite onto a REAL training frame, dish scaled into the exact bbox
    # where the trained dish sits - real mat texture, training geometry
    # (a flat synthetic mat and off-position dish both read as anomalies)
    canvas, (rx, ry, rw, rh) = reference_frame()
    canvas = canvas.copy()
    # exact-fill both axes: same physical dish, <3% aspect distortion, and the
    # reference dish underneath is fully covered (no ghost edges = no false heat)
    dish = cv2.resize(dish, (rw, rh))
    mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    region = canvas[ry:ry + rh, rx:rx + rw]
    region[mask > 0] = dish[mask > 0]

    # PatchCore has no rotation invariance and an upload carries no orientation
    # hint - score both landscape orientations and keep the one the memory bank
    # recognizes (lower raw score). Costs one extra inference, uploads only.
    rot180 = False
    flipped = cv2.rotate(canvas, cv2.ROTATE_180)
    try:
        score_a, _ = score_image(canvas)
        score_b, _ = score_image(flipped)
        if score_b < score_a:
            canvas = flipped
            rot180 = True
    except RuntimeError:
        pass                                      # no model yet: keep as composed
    return canvas, True, {"rot90cw": was_portrait, "rot180": rot180}


def overlay_file(name):
    """Sanitized lookup for /overlays/{name}: overlay PNGs in runs/ only."""
    if not name.startswith("overlay_") or not name.endswith(".png") or "/" in name \
            or ".." in name:
        return None
    path = RUNS / name
    return path if path.exists() else None


def image_b64(bgr, max_w=960):
    """Downscaled JPEG base64 of the (cropped) frame for the dashboard viewport."""
    import base64
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


def context_message():
    return {
        "type": "context",
        "dish": DISH_NAME,
        "batch": BATCH,
        "quadrant": QUADRANT,
        "bank": "Ready" if CKPT.exists() else "Not built",
        "calibration": "Locked" if CALIB.exists() else "Not set",
        "memory": "Connected" if os.environ.get("XTRACE_API_KEY") else "Off",
    }


_retrain = {"state": "idle", "detail": "", "started": None}

_sink = None


def sink():
    global _sink
    if _sink is None:
        _sink = get_sink()
    return _sink


def ws_score(raw, threshold):
    """The PROVIDED dashboard renders score/threshold on a 0..1 meter (its demo
    uses .21/.86 vs .55); raw patch distances (~30-60) would push the threshold
    tick to 4400% and peg the fill. Map raw -> raw/(2*T) so the tick sits at 0.5,
    matching the overlay's display convention. Raw values stay in the HTTP
    response and case records."""
    if threshold <= 0:
        return 0.0
    return round(min(1.0, max(0.0, raw / (2.0 * threshold))), 3)


def region_label(cx, cy):
    """Blob centre -> the words an operator would use for that spot on the dish.
    Coordinates are normalized to the frame the OPERATOR is looking at, so this
    must run after undo_display_transform has mapped uploads back."""
    col = "left" if cx < 0.34 else ("right" if cx > 0.66 else "centre")
    row = "upper" if cy < 0.34 else ("lower" if cy > 0.66 else "middle")
    if row == "middle" and col == "centre":
        return "centre of dish"
    if row == "middle":
        return f"{col} edge"
    if col == "centre":
        return f"{row} centre"
    return f"{row} {col}"


def label_locations(result):
    """Attach the geometric location to each finding. PatchCore's anomaly map is
    the only thing that actually knows WHERE the defect is - the vision model
    sees a context-free crop and its guesses were landing on the wrong side of
    the dish after upload normalization rotated the photo."""
    for blob, finding in zip(result["blobs"], result["findings"]):
        finding["location"] = region_label(blob["cx"], blob["cy"])
    return result


def undo_display_transform(result, transform):
    """Uploads are internally aligned to the memory bank's geometry; everything
    the OPERATOR sees (overlay image, blob coords) is mapped back to the
    photo's original orientation so the defect appears where they shot it."""
    path = ROOT / result["overlay_path"]
    img = cv2.imread(str(path))
    blobs = result["blobs"]
    if img is None:
        return

    if "homography" in transform:
        # draw the contours found in reference space back onto the ORIGINAL
        # photo by warping the annotation pixels through the inverse homography
        H = transform["homography"]
        original = transform["original"]
        oh, ow = original.shape[:2]
        rh, rw = img.shape[:2]
        red = ((img[:, :, 2].astype(np.int16) - img[:, :, 1] > 120) &
               (img[:, :, 2].astype(np.int16) - img[:, :, 0] > 120)).astype(np.uint8) * 255
        red_back = cv2.warpPerspective(red, H, (ow, oh),
                                       flags=cv2.WARP_INVERSE_MAP) > 127
        red_back = cv2.dilate(red_back.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        display = original.copy()
        display[red_back] = (61, 61, 255)
        cv2.imwrite(str(path), display)
        if blobs:
            pts = np.float32([[[b["cx"] * rw, b["cy"] * rh]] for b in blobs])
            back = cv2.perspectiveTransform(pts, np.linalg.inv(H))
            result["blobs"] = [
                {**b, "cx": round(float(p[0][0]) / ow, 4),
                 "cy": round(float(p[0][1]) / oh, 4)}
                for b, p in zip(blobs, back)]
        return

    if transform.get("rot180"):
        img = cv2.rotate(img, cv2.ROTATE_180)
        blobs = [{**b, "cx": round(1 - b["cx"], 4), "cy": round(1 - b["cy"], 4)}
                 for b in blobs]
    if transform.get("rot90cw"):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        blobs = [{**b, "cx": b["cy"], "cy": round(1 - b["cx"], 4)} for b in blobs]
    cv2.imwrite(str(path), img)
    result["blobs"] = blobs


def process_inspection(frame, dish_id=None, transform=None):
    """Full pipeline for one frame: inspect, emit ws messages, file the case."""
    frame = ingest(frame)
    dish_id = dish_id or next_dish_id()
    hub.emit({"type": "inspection_start", "dish_id": dish_id})

    result = run_inspection(frame, dish_id)
    if transform:
        undo_display_transform(result, transform)
    label_locations(result)               # after the undo: operator-space coords
    ws_findings = [{**{k: v for k, v in f.items() if k != "cosmetic"},
                    "score": ws_score(f["score"], result["threshold"])}
                   for f in result["findings"]]
    # defect verdicts show the rendered heatmap overlay in the viewport, so the
    # projector gets the "where is it broken" image, not just the raw photo
    viewport = frame
    if result["verdict"] == "defect":
        rendered = cv2.imread(str(ROOT / result["overlay_path"]))
        if rendered is not None:
            viewport = rendered
    hub.emit({"type": "inspection_result", "dish_id": dish_id,
              "verdict": result["verdict"],
              "score": ws_score(result["score"], result["threshold"]),
              "threshold": 0.5, "image_b64": image_b64(viewport),
              "blobs": result["blobs"], "findings": ws_findings})

    primary = result["findings"][0] if result["findings"] else {}
    status = "autonomous_resolution" if result["verdict"] == "pass" else "escalated"
    history.log_inspection(dish_id, result["verdict"], primary.get("designator"),
                           primary.get("description"), result["score"], status)
    case_id = sink().create_case(dish_id, result["verdict"], result["findings"],
                                 result["overlay_path"])
    result["case_id"] = case_id
    verify_and_mark(case_id)                     # M6; no-op fallback if unavailable
    # kitchen memory: ask XTrace whether this failure is a repeat BEFORE this
    # plate is ingested, then file the plate as a new memory (async server-side)
    result["memory"] = memory.recall_insight(result)
    memory.record_inspection(result)
    hub.emit({"type": "shift", **history.shift_summary()})
    if result["verdict"] == "defect":
        ring_operator(result)
    return result


def ring_operator(result):
    """Defect -> Twilio rings the operator and describes it. Fire-and-forget:
    a slow or failing Twilio API must never stall or fail the inspection."""
    import threading

    def _call():
        try:
            from core.phone import place_defect_alert
            place_defect_alert(result)
        except Exception as e:
            hub.emit({"type": "event", "kind": "phone", "source": "voice.twilio",
                      "message": f"Defect call failed: {e}"[:120],
                      "dish_id": result["dish_id"]})
            hub.emit({"type": "phone", "status": "failed",
                      "dish_id": result["dish_id"], "message": str(e)[:120]})

    threading.Thread(target=_call, daemon=True).start()


def verify_and_mark(case_id):
    """M6 hook: GPT consistency review -> set_verified. Filled in by verify.py."""
    try:
        from verify import verify_case
    except ImportError:
        return
    verified, note = verify_case(case_id)
    sink().set_verified(case_id, verified, note)


# ---------------- FastAPI ----------------

def build_app():
    import asyncio

    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

    app = FastAPI(title="Seefu plate check")

    @app.on_event("startup")
    async def warm():
        hub.register_loop(asyncio.get_running_loop())
        hub.emit(context_message())              # spec: context once on startup
        try:
            get_model()
            read_threshold()
        except RuntimeError as e:
            print(f"WARNING: {e}")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await hub.connect(websocket)
        import json as _json
        await websocket.send_text(_json.dumps(context_message()))
        try:
            while True:
                await websocket.receive_text()   # dashboard never sends; detects close
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    @app.get("/")
    def minimal():
        from fastapi.responses import FileResponse
        return FileResponse(ROOT / "dashboard/minimal.html")

    @app.api_route("/phone/twiml", methods=["GET", "POST"])
    def phone_twiml(case_id: str = ""):
        # Twilio trial accounts reject inline Twiml; they fetch TwiML from this
        # public (tunneled) URL. Plain Twilio voice reads the case summary -
        # deliberately no media bridge on the phone leg (reliability first);
        # the browser agent at /agent stays the interactive surface.
        from fastapi.responses import Response
        from core.phone import (case_alert_text, public_host,
                                synthesize_alert_audio, tts_status)
        from xml.sax.saxutils import escape
        raw_text = case_alert_text(case_id)
        audio = synthesize_alert_audio(raw_text, case_id or "test")
        host = public_host()
        status = tts_status()
        if audio is not None and host:
            announce = f'<Play>https://{host}/phone/audio/{audio.name}</Play>'
            voice, why = "elevenlabs", f"{status['model']} / {status['voice_id']}"
        else:
            announce = f'<Say>{escape(raw_text)}</Say>'
            voice = "twilio-fallback"
            why = (status["reason"] if audio is None else
                   "no public host to serve the mp3 from")
        # this is the moment of truth - Twilio fetches TwiML when the operator's
        # phone connects, so this line says what they are ABOUT to hear. Emitted
        # as an "event" (not a {"type":"phone"} message) on purpose: the minimal
        # dashboard cancels its call-status poll on any phone message.
        print(f"PHONE TwiML: case={case_id or '-'} voice={voice} ({why})", flush=True)
        hub.emit({"type": "event", "kind": "phone", "source": f"voice.{voice}",
                  "message": f"Operator is hearing the "
                             f"{'ElevenLabs' if voice == 'elevenlabs' else 'Twilio fallback'}"
                             f" voice ({why})",
                  "dish_id": ""})
        # announce-only: the summary plays twice so a manager reaching for a pen
        # hears everything; the case with the fix stays on the dashboard
        tail = f'<Pause length="1"/><Say>Repeating.</Say>{announce}'
        twiml = f'<Response><Pause length="1"/>{announce}{tail}</Response>'
        return Response(content=twiml, media_type="text/xml")

    @app.get("/phone/audio/{name}")
    def phone_audio(name: str):
        from fastapi.responses import FileResponse
        from core.phone import alert_audio_file
        path = alert_audio_file(name)
        if path is None:
            raise HTTPException(404, "no such alert audio")
        return FileResponse(path, media_type="audio/mpeg")

    @app.get("/phone/status/{sid}")
    def phone_status(sid: str):
        from core.phone import call_status
        try:
            return {"sid": sid, "status": call_status(sid)}
        except Exception as e:
            raise HTTPException(502, f"status lookup failed: {e}")

    @app.get("/overlays/{name}")
    def overlay(name: str):
        from fastapi.responses import FileResponse
        path = overlay_file(name)
        if path is None:
            raise HTTPException(404, "no such overlay")
        return FileResponse(path)

    @app.post("/phone/call")
    def phone_call():
        from core.phone import place_call
        try:
            return {"call_sid": place_call(), "status": "ringing"}
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    # ---------------- kitchen memory (XTrace) ----------------

    @app.get("/memory/status")
    def memory_status():
        return memory.status()

    @app.get("/memory/summary")
    def memory_summary():
        # compose mode: XTrace returns a display-ready shift briefing
        return memory.shift_briefing()

    @app.get("/memory/recent")
    def memory_recent():
        return memory.recent_memories()

    @app.post("/memory/correction")
    def memory_correction(body: dict):
        text = (body or {}).get("text", "").strip()
        if not text:
            raise HTTPException(400, "body must be {\"text\": \"...\"}")
        resp = memory.record_correction(text)
        if resp is None and not memory.enabled():
            raise HTTPException(503, "XTRACE_API_KEY not set")
        return {"recorded": resp is not None,
                "note": "correction stored as a directive memory; it will surface "
                        "on future plates of this dish via trigger recall"}

    # ---------------- retraining ----------------

    @app.post("/train/upload")
    def train_upload(files: list[UploadFile] = File(...)):
        """Stage new known-good plate photos into the normal set. The model
        does not change until /train/rebuild runs."""
        added = []
        normal_dir = ROOT / "data/normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for i, f in enumerate(files):
            data = f.file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, f"could not decode {f.filename}")
            name = f"dish_new_{stamp}_{i:02d}.png"
            cv2.imwrite(str(normal_dir / name), img)
            added.append(name)
        return {"staged": added, "normal_count": len(list(normal_dir.glob('*.png'))),
                "next": "POST /train/rebuild to retrain on the updated set"}

    @app.post("/train/rebuild")
    def train_rebuild():
        """Retrain the memory bank on data/normal in the background, then
        recalibrate the threshold. The serving model hot-reloads on the new
        checkpoint's mtime; the server never goes down."""
        if _retrain["state"] == "running":
            raise HTTPException(409, "a rebuild is already running")
        import subprocess
        import threading

        def _run():
            _retrain.update(state="running", started=time.strftime("%H:%M:%S"),
                            detail="training PatchCore bank")
            log = (RUNS / "retrain.log").open("w")
            try:
                for args, detail in (
                        ([sys.executable, "inspection/train_bank.py"], "training"),
                        ([sys.executable, "inspection/serve.py", "--calibrate"],
                         "recalibrating threshold")):
                    _retrain["detail"] = detail
                    proc = subprocess.run(args, cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=1800)
                    if proc.returncode != 0:
                        _retrain.update(state="failed",
                                        detail=f"{detail} exited {proc.returncode} "
                                               "(see runs/retrain.log)")
                        return
                _retrain.update(state="done", detail="new bank live; model "
                                "hot-reloads on next inspection")
                hub.emit({"type": "event", "kind": "train", "source": "retrain",
                          "message": "Memory bank rebuilt and threshold "
                                     "recalibrated on the updated normal set",
                          "dish_id": ""})
            except Exception as e:
                _retrain.update(state="failed", detail=str(e)[:200])
            finally:
                log.close()

        threading.Thread(target=_run, daemon=True).start()
        return {"started": True, "watch": "/train/status"}

    @app.get("/train/status")
    def train_status():
        out = dict(_retrain)
        out["ckpt_exists"] = CKPT.exists()
        if CKPT.exists():
            out["ckpt_mtime"] = time.strftime("%H:%M:%S",
                                              time.localtime(CKPT.stat().st_mtime))
        out["normal_count"] = len(list((ROOT / "data/normal").glob("*.png")))
        return out

    @app.get("/health")
    def health():
        try:
            threshold, source = read_threshold()
        except RuntimeError:
            threshold, source = None, "missing"
        return {"status": "ok", "ckpt": CKPT.exists(), "model_loaded": _model is not None,
                "threshold": threshold, "threshold_source": source}

    @app.post("/inspect")
    def inspect(file: UploadFile | None = File(None), live: int = 0,
                dish_id: str | None = None):
        # sync def on purpose: FastAPI runs it in the threadpool, keeping the
        # event loop free so ws messages stream out DURING the inspection
        # (async def here made start/result/case arrive in one burst at the end)
        transform = None
        if live:
            try:
                frame = grab_live_frame()
            except (RuntimeError, SystemExit) as e:
                raise HTTPException(503, f"live capture failed: {e}")
        elif file is not None:
            data = file.file.read()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(400, "could not decode image")
            RUNS.mkdir(exist_ok=True)
            cv2.imwrite(str(RUNS / f"upload_{time.strftime('%H%M%S')}.png"), frame)
            frame, was_normalized, transform = normalize_upload(frame)
        else:
            raise HTTPException(400, "upload a file or pass ?live=1")
        try:
            result = process_inspection(frame, dish_id,
                                        transform if file is not None else None)
            if file is not None and was_normalized:
                result["normalized_upload"] = True
                result["note"] = ((result.get("note") or "") +
                                  " Off-pass photo auto-normalized (background/orientation/"
                                  "position). The calibrated threshold is approximate here - "
                                  "trust the localized contours; tall garnishes may show "
                                  "parallax warmth.").strip()
            return result
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    return app


# ---------------- threshold calibration ----------------

def calibrate():
    def scores_for(folder):
        out = []
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = ingest(cv2.imread(str(p)))
            s, _ = score_image(img)
            out.append((s, p.name))
        return sorted(out)

    normals = scores_for(ROOT / "data/normal")
    defect_dir = ROOT / "data/test_defect"
    defects = scores_for(defect_dir) if defect_dir.exists() else []

    print(f"\nNORMAL scores ({len(normals)}):")
    for s, name in normals:
        print(f"  {s:9.4f}  {name}")
    print(f"\nDEFECT scores ({len(defects)}):")
    for s, name in defects:
        print(f"  {s:9.4f}  {name}")

    max_normal = normals[-1][0]
    if defects:
        min_defect = defects[0][0]
        proposed = (max_normal + min_defect) / 2.0
        print(f"\nmax normal = {max_normal:.4f}   min defect = {min_defect:.4f}")
        if min_defect <= max_normal:
            print("WARNING: distributions OVERLAP - defects are not separating. "
                  "Re-shoot or reframe before trusting this threshold.")
    else:
        proposed = max_normal * 1.15
        print(f"\nmax normal = {max_normal:.4f}   (no data/test_defect images yet; "
              "proposing max normal + 15% margin)")
    print(f"proposed threshold: {proposed:.4f}")

    existing = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    existing.update({
        "threshold": round(proposed, 4),
        "normal_max": round(max_normal, 4),
        "defect_min": round(defects[0][0], 4) if defects else None,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    CALIB.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"written to {CALIB} (human blesses this number at checkpoint 2)")


def main():
    ap = argparse.ArgumentParser(description="Seefu inspection service")
    ap.add_argument("--calibrate", action="store_true",
                    help="score normal + test_defect sets, propose threshold, exit")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
