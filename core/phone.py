"""Twilio phone alerts: a failed plate rings the kitchen manager.

When a dish fails plate check, Seefu places an outbound call via Twilio REST.
Twilio fetches TwiML from our public host (scripts/start_tunnel.py writes
runs/.tunnel_host, or set PUBLIC_HOST) and plays an ElevenLabs-synthesized
summary of the case: dish id, what is wrong, and the fix. Falls back to
Twilio's own voice when synthesis is unavailable. Announce-only by design:
the manager hears the alert and the case stays on the dashboard.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNNEL_FILE = ROOT / "runs/.tunnel_host"

from core import hub  # noqa: E402


def public_host():
    if os.environ.get("PUBLIC_HOST"):
        return os.environ["PUBLIC_HOST"].replace("https://", "").strip("/")
    if TUNNEL_FILE.exists():
        host = TUNNEL_FILE.read_text().strip()
        if host:
            return host
    return None


def place_call():
    """Create the outbound call via Twilio REST. Returns the call SID."""
    import httpx
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_ = os.environ.get("TWILIO_FROM")
    to = os.environ.get("TWILIO_TO")
    if not all((sid, token, from_, to)):
        raise RuntimeError("TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM/TO missing from .env")
    host = public_host()
    if host is None:
        raise RuntimeError("no public tunnel: run `make tunnel` in another terminal first")
    r = httpx.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
        data={"To": to, "From": from_, "Url": f"https://{host}/phone/twiml"},
        auth=(sid, token), timeout=15.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Twilio {r.status_code}: {r.text[:180]}")
    call_sid = r.json()["sid"]
    hub.emit({"type": "event", "kind": "phone", "source": "voice.twilio",
              "message": f"Calling the operator at {to}", "dish_id": ""})
    return call_sid


_last_alert_ts = 0.0

# ---------------- ElevenLabs voice ----------------
#
# Verified against https://elevenlabs.io/docs (models overview + the
# POST /v1/text-to-speech/{voice_id} reference) on 2026-07-24:
#   * eleven_turbo_v2_5 (what this used to pin) is documented as superseded by
#     the Flash line; the Flash models only buy latency, and this mp3 is
#     PRE-GENERATED at call placement and cached on disk, so latency buys us
#     nothing. eleven_multilingual_v2 is the model ElevenLabs recommends for
#     narration, which is exactly what an alert read is.
#   * output_format is a QUERY parameter on that endpoint, not a body field -
#     the old code put it in the JSON body, where it was silently ignored.
#   * voice_settings takes stability / similarity_boost / style /
#     use_speaker_boost / speed. Nothing here is invented.
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
# Twilio downsamples whatever we <Play> to 8 kHz u-law, so bitrate above this is
# bytes Twilio has to pull back through the demo tunnel for nothing. The lowest
# documented mp3 format is also the one available on every subscription tier.
ELEVEN_OUTPUT_FORMAT = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_22050_32")
ELEVEN_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"     # Rachel: calm, even, unhurried
# Tuned for an industrial alert read: steady rather than dramatic, no stylistic
# flourish, and a touch under normal pace because this is phone audio carrying
# part designators the operator has to write down.
ELEVEN_VOICE_SETTINGS = {
    "stability": 0.62,
    "similarity_boost": 0.80,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 0.94,
}

# What the operator actually heard on the most recent synthesis attempt. The
# fallback to Twilio's robotic built-in voice used to be completely silent -
# nobody could tell whether ElevenLabs had ever run. Now every attempt lands
# here, in the log, and on the dashboard.
_last_tts = {"voice": "twilio-fallback", "reason": "no synthesis attempted yet",
             "model": None, "voice_id": None, "bytes": 0, "cached": False}


def tts_status():
    """Snapshot of the most recent ElevenLabs attempt."""
    return dict(_last_tts)


def eleven_model():
    """Resolved per call, like the voice id: .env is loaded by the server after
    this module could already have been imported, so a module-level read alone
    would silently ignore an ELEVENLABS_MODEL_ID override."""
    return os.environ.get("ELEVENLABS_MODEL_ID") or ELEVEN_MODEL


def eleven_output_format():
    return os.environ.get("ELEVENLABS_OUTPUT_FORMAT") or ELEVEN_OUTPUT_FORMAT


def _tts_ok(path, voice_id, cached, size):
    _last_tts.update(voice="elevenlabs", reason="", model=eleven_model(),
                     voice_id=voice_id, bytes=size, cached=cached)
    how = "cached" if cached else "synthesized"
    print(f"ELEVENLABS: {how} {size} bytes  model={eleven_model()} "
          f"voice={voice_id} format={eleven_output_format()} -> {path.name}",
          flush=True)
    hub.emit({"type": "event", "kind": "phone", "source": "voice.elevenlabs",
              "message": f"ElevenLabs voice ready ({eleven_model()}, "
                         f"{max(1, size // 1024)} KB, {how})",
              "dish_id": ""})
    return path


def _tts_failed(reason, voice_id=None):
    _last_tts.update(voice="twilio-fallback", reason=reason, model=eleven_model(),
                     voice_id=voice_id, bytes=0, cached=False)
    print(f"ELEVENLABS FAILED - falling back to Twilio's built-in voice: {reason}",
          flush=True)
    hub.emit({"type": "event", "kind": "phone", "source": "voice.elevenlabs",
              "message": f"ElevenLabs unavailable, using Twilio voice: {reason}",
              "dish_id": ""})
    return None


def _tts_http_reason(r):
    """Turn an ElevenLabs error response into one line an operator can act on."""
    detail = ""
    try:
        payload = r.json()
        d = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(d, dict):
            detail = str(d.get("message") or d.get("status") or "")
        elif d:
            detail = str(d)
    except Exception:
        detail = ""
    if not detail:
        try:
            detail = str(r.text or "")[:160]
        except Exception:
            detail = ""
    code = r.status_code
    if code == 401:
        return f"HTTP 401, ELEVENLABS_API_KEY rejected. {detail}".strip()
    if code == 402:
        return f"HTTP 402, ElevenLabs quota exhausted. {detail}".strip()
    if code == 422:
        return f"HTTP 422, request rejected. {detail}".strip()
    if code == 429:
        return f"HTTP 429, ElevenLabs rate limit. {detail}".strip()
    return f"HTTP {code}. {detail}".strip()


def _safe_case(case_id):
    """case_id is used to build filenames; keep it to filename-safe characters."""
    keep = "".join(c for c in str(case_id or "") if c.isalnum() or c in "-_")
    return keep or "test"


def synthesize_alert_audio(text, case_id):
    """ElevenLabs TTS for the alert call: writes runs/alerts/<case>_<hash>.mp3
    and returns the path, or None when ElevenLabs could not be used.

    None still means "the call goes ahead on Twilio's built-in voice" - this can
    never break a call - but it is no longer silent: the reason is printed, kept
    in tts_status() and broadcast to the dashboard."""
    import httpx
    key = os.environ.get("ELEVENLABS_API_KEY")
    voice = os.environ.get("ELEVENLABS_VOICE_ID") or ELEVEN_DEFAULT_VOICE
    if not key:
        return _tts_failed("ELEVENLABS_API_KEY is not set (add it to .env)", voice)
    if not (text or "").strip():
        return _tts_failed("nothing to synthesize: the alert text was empty", voice)
    import hashlib
    stamp = hashlib.sha1(text.encode()).hexdigest()[:8]
    out = ROOT / f"runs/alerts/{_safe_case(case_id)}_{stamp}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:   # pre-generated at call placement
        return _tts_ok(out, voice, True, out.stat().st_size)
    try:
        r = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={"xi-api-key": key, "content-type": "application/json"},
            params={"output_format": eleven_output_format()},
            json={"text": text, "model_id": eleven_model(),
                  "voice_settings": dict(ELEVEN_VOICE_SETTINGS)},
            timeout=30.0)
    except Exception as e:
        return _tts_failed(f"{type(e).__name__}: {e}"[:200], voice)
    if r.status_code != 200:
        return _tts_failed(_tts_http_reason(r), voice)
    if not r.content:
        return _tts_failed("ElevenLabs returned 200 with an empty body", voice)
    out.write_bytes(r.content)
    return _tts_ok(out, voice, False, len(r.content))


def alert_audio_file(name):
    """Sanitized lookup for /phone/audio/{name}: alert MP3s only."""
    if not name.endswith(".mp3") or "/" in name or ".." in name:
        return None
    path = ROOT / "runs/alerts" / name
    return path if path.exists() else None


# ---------------- the spoken diagnosis ----------------

_COORDS = re.compile(r"\(?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)?")
_MARKDOWN = re.compile(r"[*_`#>\[\]{}|]+")
_COUNTS = ("no", "one", "two", "three", "four", "five",
           "six", "seven", "eight", "nine", "ten")
_ORDINALS = ("First", "Second", "Third")


def _count_word(n):
    return _COUNTS[n] if 0 <= n < len(_COUNTS) else str(n)


def speakable(text):
    """A finding description, trimmed to something that sounds right out loud:
    no coordinates (meaningless on a phone), no markdown, no line breaks."""
    s = str(text or "").strip()
    s = _COORDS.sub("", s)
    s = _MARKDOWN.sub("", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+(at|near|around|in)$", "", s, flags=re.I)   # left by the coord strip
    s = s.strip(" ,;:-.")
    if s.lower().startswith("anomalous region"):
        # the pre-vision fallback description; the sentence already says where
        return "an anomalous region"
    if len(s) > 200:
        s = s[:200].rsplit(" ", 1)[0].strip(" ,;:-") + ", see the dashboard for the rest"
    return s


def spoken_place(location):
    """The geometry-derived finding["location"] as a phrase, not a label.
    region_label() emits: 'upper left', 'lower right', 'left edge',
    'upper centre', 'centre of dish'."""
    loc = str(location or "").strip()
    if not loc:
        return ""
    if loc == "centre of dish":
        return "in the centre of the dish"
    if loc.endswith("edge"):
        return f"on the {loc}"
    return f"in the {loc}"


def finding_sentence(index, finding):
    """One finding, spoken: where it is, what part it is, what is wrong.
    Never says "unmapped" - an operator cannot walk to "unmapped", but they can
    walk to "the upper left"."""
    lead = _ORDINALS[index] if index < len(_ORDINALS) else "Also"
    place = spoken_place(finding.get("location"))
    designator = str(finding.get("designator") or "").strip()
    component = speakable(finding.get("component"))
    desc = speakable(finding.get("description")) or "an anomalous region"

    head = [lead]
    if place:
        head.append(place)
    if designator and component and not component.startswith("an anomalous"):
        head.append(f"at {designator}, the {component}")
    elif designator:
        head.append(f"at {designator}")
    fix = speakable(finding.get("fix"))
    tail = f" The fix, {fix}." if fix else ""
    if desc.startswith("an anomalous"):
        # nothing specific to add - keep it as one clause rather than a
        # sentence fragment hanging off a full stop
        return ", ".join(head + [desc]) + "." + tail
    return ", ".join(head) + ". " + desc[0].upper() + desc[1:] + "." + tail


def diagnosis_text(dish_id, verdict, score, threshold, findings):
    """The spoken diagnosis, shared by the live-result and filed-case paths.

    Written to be HEARD, not read: short sentences, commas where a person would
    breathe, no markdown, no coordinates, no JSON."""
    dish = str(dish_id or "unknown")
    findings = list(findings or [])
    parts = ["This is Seefu, the plate check station on the pass, with a hold alert."]
    if str(verdict or "defect").lower() == "pass":
        parts.append(f"Plate {dish} passed the check and was released.")
    else:
        parts.append(f"Plate {dish} failed the check and is being held at the pass.")

    if isinstance(score, (int, float)) and isinstance(threshold, (int, float)) \
            and not isinstance(score, bool) and threshold > 0:
        over = (float(score) - float(threshold)) / float(threshold) * 100.0
        side = "above" if over >= 0 else "below"
        parts.append(f"The anomaly score is {float(score):.1f}, against a pass "
                     f"limit of {float(threshold):.1f}. That is about "
                     f"{abs(over):.0f} percent {side} the line.")

    shown = findings[:3]
    if shown:
        n = len(findings)
        plural = "s were" if n != 1 else " was"
        tail = ", and here are the first three." if n > 3 else "."
        parts.append(f"{_count_word(n).capitalize()} problem region{plural} found{tail}")
        for i, f in enumerate(shown):
            parts.append(finding_sentence(i, f))
    else:
        parts.append("No single region stood out, but the overall anomaly score "
                     "exceeded the threshold, so this plate needs a human look "
                     "before it goes out.")

    parts.append("The marked plate image and the fix are on the dashboard.")
    return " ".join(parts)


def defect_alert_text(result):
    """The spoken diagnosis for a live inspection result."""
    return diagnosis_text(result.get("dish_id"), result.get("verdict", "defect"),
                          result.get("score"), result.get("threshold"),
                          result.get("findings"))


def _alert_text_file(case_id):
    return ROOT / f"runs/alerts/{_safe_case(case_id)}.txt"


def stash_alert_text(case_id, text):
    """Park the exact text the alert was placed with next to its mp3.

    /phone/twiml has only the case_id, and a filed case record carries no score
    or threshold - so without this the TwiML leg would build DIFFERENT words,
    miss the pre-generated mp3 (the cache key is a hash of the text) and make
    Twilio wait on a fresh synthesis mid-call."""
    try:
        path = _alert_text_file(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except Exception as e:
        print(f"PHONE: could not stash alert text: {type(e).__name__}: {e}", flush=True)


def stashed_alert_text(case_id):
    try:
        path = _alert_text_file(case_id)
        return path.read_text().strip() if path.exists() else ""
    except Exception:
        return ""


def case_alert_text(case_id):
    """Spoken summary for the TwiML endpoint: the exact text the alert was
    placed with when we have it, rebuilt from the filed case record otherwise."""
    stashed = stashed_alert_text(case_id)
    path = ROOT / f"runs/cases/{case_id}.json"
    case = None
    if case_id and path.exists():
        try:
            case = json.loads(path.read_text())
        except Exception as e:
            print(f"PHONE: unreadable case {case_id}: {type(e).__name__}: {e}",
                  flush=True)
    if stashed:
        # scripts/reset_demo.py restarts case ids at 1041, so a stash left over
        # from an earlier rehearsal could otherwise put the WRONG dish's
        # diagnosis on the call. If the filed case disagrees, the case wins.
        dish = str((case or {}).get("dish_id") or "")
        if case is None or (dish and dish in stashed):
            return stashed
        print(f"PHONE: ignoring stale alert text for case {case_id} "
              f"(expected dish {dish})", flush=True)
    if case is None:
        return ("This is Seefu, the automated inspection station. Test call "
                "successful. Stay on the line to talk to the inspector.")
    return diagnosis_text(case.get("dish_id"), case.get("verdict", "defect"),
                          case.get("score"), case.get("threshold"),
                          case.get("findings"))


def place_defect_alert(result):
    """Ring the operator and DESCRIBE the defect - a notification, not a
    conversation. Inline TwiML <Say> means no tunnel and no media bridge:
    Twilio synthesizes the speech itself (Polly neural voice). Returns the
    call SID, or None when skipped (cooldown / disabled / missing creds)."""
    global _last_alert_ts
    import time as _time

    import httpx
    if os.environ.get("SEEFU_CALL_ON_DEFECT", "1") == "0":
        hub.emit({"type": "phone", "status": "disabled",
                  "dish_id": result["dish_id"],
                  "message": "auto-call disabled (SEEFU_CALL_ON_DEFECT=0)"})
        return None
    cooldown = float(os.environ.get("SEEFU_CALL_COOLDOWN", "60"))
    if _time.time() - _last_alert_ts < cooldown:
        hub.emit({"type": "event", "kind": "phone", "source": "voice.twilio",
                  "message": f"Defect call skipped (cooldown {cooldown:.0f}s)",
                  "dish_id": result["dish_id"]})
        hub.emit({"type": "phone", "status": "skipped",
                  "dish_id": result["dish_id"],
                  "message": f"cooldown {cooldown:.0f}s"})
        return None
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_ = os.environ.get("TWILIO_FROM")
    to = os.environ.get("TWILIO_TO")
    if not all((sid, token, from_, to)):
        raise RuntimeError("TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM/TO missing from .env")

    host = public_host()
    if host is None:
        raise RuntimeError("Twilio trial accounts need a public TwiML URL - "
                           "run `make tunnel` first")
    # Build the spoken diagnosis ONCE, stash it, and pre-generate its ElevenLabs
    # mp3 so it is ready the instant the operator answers. /phone/twiml then
    # reads the same words back out of the stash and hits the audio cache
    # instead of making Twilio wait on a synthesis mid-call.
    case_key = result.get("case_id") or "test"
    text = defect_alert_text(result)
    stash_alert_text(case_key, text)
    audio = synthesize_alert_audio(text, case_key)
    tts = tts_status()

    from urllib.parse import quote
    url = f"https://{host}/phone/twiml?case_id={quote(str(result.get('case_id', '')))}"
    if audio is not None:
        mode = f"ElevenLabs voice ({tts['model']})"
    else:
        mode = f"Twilio fallback voice: {tts['reason']}"
    r = httpx.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
        data={"To": to, "From": from_, "Url": url},
        auth=(sid, token), timeout=15.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Twilio {r.status_code}: {r.text[:180]}")
    _last_alert_ts = _time.time()
    call_sid = r.json()["sid"]
    hub.emit({"type": "event", "kind": "phone", "source": "voice.twilio",
              "message": f"Calling operator: defect on {result['dish_id']} ({mode})",
              "dish_id": result["dish_id"]})
    # message is what the minimal dashboard prints verbatim, so the voice that
    # will actually speak is visible there; voice/reason are the structured form
    placed = {"type": "phone", "status": "placed", "sid": call_sid,
              "dish_id": result["dish_id"], "message": mode,
              "voice": tts["voice"]}
    if audio is None:
        placed["reason"] = tts["reason"]
    else:
        placed.update(model=tts["model"], voice_id=tts["voice_id"],
                      bytes=tts["bytes"], cached=tts["cached"])
    hub.emit(placed)
    return call_sid


def call_status(call_sid):
    """Current Twilio status of a call: queued/ringing/in-progress/completed/..."""
    import httpx
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not (sid and token) or not call_sid.startswith("CA"):
        raise RuntimeError("bad call sid or missing Twilio creds")
    r = httpx.get(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{call_sid}.json",
                  auth=(sid, token), timeout=10.0)
    r.raise_for_status()
    return r.json()["status"]
