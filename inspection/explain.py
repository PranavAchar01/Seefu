"""M4: GPT-4o vision description of anomalous crops.

For each blob on a defect verdict: crop a padded square around the blob from
the full-resolution (cropped) frame, send to gpt-4o, attach one technician
sentence as finding.description.

Verified against installed openai SDK 2.47.0 (Responses API):
- input_image takes image_url as a PLAIN data-URL string (chat.completions
  uses a nested object - do not mix the shapes)
- hard 8s budget needs with_options(timeout=8.0, max_retries=0): the SDK
  retries timeouts by default, which would triple the wall clock
Cache: keyed by sha256 of the crop JPEG in runs/explain_cache.json.
Fallback on timeout/any API error: "Anomalous region at <designator or coords>".
"""

import base64
import hashlib
import json
import os
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "runs/explain_cache.json"

PROMPT = ("Two crops of the SAME region of a plated dish, a sesame beef bowl: "
          "white rice base, glazed sesame beef with chili flecks, steamed "
          "broccoli, scallion garnish, black bowl. Image 1: the KNOWN-GOOD "
          "reference plate. Image 2: the plate under inspection, flagged as "
          "anomalous here. Compare image 2 against image 1 and reply with "
          "EXACTLY one line:\n"
          "FAIL: <what is wrong> | FIX: <the one action the kitchen takes> - if "
          "image 2 shows a missing or short ingredient, a wrong or misplaced "
          "element, sauce or food where it should not be, a foreign object, "
          "signs of contamination, or possible allergen cross-contact that is "
          "not in image 1. The FIX is imperative and specific, for example "
          "'re-garnish with scallion and send', 'add the broccoli portion', "
          "'wipe the rim and send', 'remake the plate: possible foreign object'.\n"
          "COSMETIC: <what it is> - if the differences are only lighting, steam, "
          "sauce sheen, slight position shifts within normal plating variance, "
          "or anything also present in image 1. Only say FAIL when you are "
          "CONFIDENT, with ONE exception: anything that could be a foreign "
          "object, hair, plastic, glass, or contamination is ALWAYS FAIL even "
          "if you are unsure, because a doubtful plate never leaves the pass.\n"
          "These are CLOSE-UP CROPS, not the whole plate, so you cannot tell "
          "where on the plate this region sits. NEVER name a plate position "
          "('the upper right', 'the left side of the bowl') - the station "
          "computes the true location from the anomaly map. Describe only WHAT "
          "is wrong with the food you can see.")
PROMPT_VERSION = b"v4-dish-fix"

# The model sees a crop of the frame AFTER upload normalization has rotated or
# warped the photo into the memory bank's geometry, so any dish-position claim
# it makes is both a guess and, on a rotated upload, usually the wrong one (the
# operator's upper-left arrives as the model's upper-right). Positions come from
# the blob geometry instead, which serve.py maps back to the original photo.
_IMAGE_PHRASING = (
    (r"\s+in image 2\b", ""),
    (r"\s+in image 1\b", " on the reference"),
    (r"\bimage 2\b", "the dish"),
    (r"\bimage 1\b", "the reference dish"),
)
_REGION_NOUN = r"(?:corner|side|edge|section|area|region|portion|part|quadrant|end)"
_DIRECTION = r"(?:upper|lower|top|bottom|left|right|upper[\s-]left|centre|center)"
_POSITION_PHRASING = (
    # "...damage in the upper right corner of the plate" -> "...damage"
    (r"[,\s]+(?:in|at|on|near|towards?|along)\s+the\s+"
     r"(?:" + _DIRECTION + r")[\s-]*(?:left|right)?[\s-]*(?:hand[\s-]*)?"
     + _REGION_NOUN + r"(?:\s+of\s+the\s+(?:dish|plate|bowl|image|photo|frame))?", ""),
    # "The upper right corner of the plate is damaged" -> "The corner of the plate is damaged"
    (r"\b(?:" + _DIRECTION + r")[\s-]+(?:left|right)?[\s-]*(?:hand[\s-]*)?"
     r"(?=" + _REGION_NOUN + r"\b)", ""),
)


def _scrub(text):
    """Strip the model's image-1/image-2 framing and any dish-position claim."""
    for pattern, repl in _IMAGE_PHRASING + _POSITION_PHRASING:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,;])", r"\1", text)
    return text

_client = None
_cache = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        _client = OpenAI()
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache, indent=2))


def crop_blob(frame, blob, pad=2.2, min_px=160):
    """Padded square around the blob center from the full-res frame."""
    h, w = frame.shape[:2]
    cx, cy = int(blob["cx"] * w), int(blob["cy"] * h)
    half = max(int(blob["r"] * max(w, h) * pad), min_px // 2)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    return frame[y0:y1, x0:x1]


_SAFETY_WORDS = ("foreign", "contamin", "hair", "plastic", "glass", "metal",
                 "insect", "mold", "allergen", "cross-contact")


def _split_fix(text):
    """'<what is wrong> | FIX: <action>' -> (description, fix or None)."""
    m = re.split(r"\|?\s*FIX\s*:", text, maxsplit=1, flags=re.IGNORECASE)
    if len(m) == 2:
        return m[0].strip(" |,;-"), m[1].strip() or None
    return text.strip(), None


def _parse_verdict(text):
    """FAIL/COSMETIC prefix -> (description, fix, cosmetic). Unknown format
    falls back to keyword heuristics, keeping the finding when genuinely unsure."""
    stripped = text.strip()
    upper = stripped.upper()
    # the model compares "image 1/2" and guesses plate positions; the kitchen
    # should hear neither. Scrub AFTER reading the FAIL/COSMETIC prefix off the
    # raw text so the verdict never depends on what the scrub rewrote.
    stripped = _scrub(stripped)
    if upper.startswith(("FAIL", "DAMAGE")):
        body = stripped.split(":", 1)[-1].strip() or stripped
        desc, fix = _split_fix(body)
        lowered = desc.lower()
        # food safety inverts the PCB rule: hedged wording downgrades a finding
        # UNLESS it touches contamination, where doubt itself is the failure
        hedged = any(h in lowered for h in ("possibl", "may be", "might be", "perhaps"))
        safety = any(s in lowered for s in _SAFETY_WORDS)
        if hedged and not safety:
            return desc, fix, True
        if safety and not fix:
            fix = "hold the plate and remake: do not send a doubtful plate"
        return desc, fix, False
    if upper.startswith("COSMETIC"):
        return stripped.split(":", 1)[-1].strip() or stripped, None, True
    lowered = stripped.lower()
    if any(kw in lowered for kw in ("nothing looks wrong", "no defect", "unable to",
                                    "appears normal", "looks normal")):
        return stripped, None, True
    desc, fix = _split_fix(stripped)
    return desc, fix, False


def describe_crop(crop_bgr, ref_crop_bgr, fallback):
    """Comparative call: the flagged crop NEXT TO the same region of a known-good
    frame, so the model judges what CHANGED, not what merely looks unusual.
    Returns (description, fix, cosmetic, how); cached; fallback on any failure."""
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ok_ref, buf_ref = cv2.imencode(".jpg", ref_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok or not ok_ref:
        return fallback, None, False, "fallback"
    jpeg, jpeg_ref = buf.tobytes(), buf_ref.tobytes()
    key = hashlib.sha256(jpeg + jpeg_ref + PROMPT_VERSION).hexdigest()
    cache = _load_cache()
    if key in cache:
        desc, fix, cosmetic = _parse_verdict(cache[key])
        return desc, fix, cosmetic, "vision (cached)"

    client = _get_client()
    if client is None:
        return fallback, None, False, "fallback (no OPENAI_API_KEY)"

    from openai import APIConnectionError, APIStatusError, APITimeoutError
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    url_ref = "data:image/jpeg;base64," + base64.b64encode(jpeg_ref).decode("ascii")
    try:
        resp = client.with_options(timeout=10.0, max_retries=0).responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": url_ref, "detail": "high"},
                    {"type": "input_image", "image_url": url, "detail": "high"},
                ],
            }],
            max_output_tokens=90,
        )
        text = resp.output_text.strip()
        if not text:
            return fallback, None, False, "fallback (empty response)"
        cache[key] = text
        _save_cache()
        desc, fix, cosmetic = _parse_verdict(text)
        return desc, fix, cosmetic, "vision"
    except (APITimeoutError, APIConnectionError, APIStatusError) as e:
        return fallback, None, False, f"fallback ({type(e).__name__})"


def explain_findings(frame, ref_frame, blobs, findings):
    """Attach comparative vision descriptions, the fix suggestion, and the
    cosmetic classification in place. Each finding gains "fix" and "cosmetic"."""
    for blob, finding in zip(blobs, findings):
        fallback = f"Anomalous region at ({blob['cx']:.2f}, {blob['cy']:.2f})"
        crop = crop_blob(frame, blob)
        ref_crop = crop_blob(ref_frame, blob)
        description, fix, cosmetic, how = describe_crop(crop, ref_crop, fallback)
        finding["description"] = description
        finding["fix"] = fix
        finding["cosmetic"] = cosmetic
        if how.startswith("vision"):
            finding["source"] = "vision"
    return findings
