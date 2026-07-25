"""XTrace memory: the kitchen's long-term memory across plates, shifts, stations.

PatchCore sees one plate at a time and forgets it the moment the verdict lands.
XTrace (docs.xtrace.ai) is the layer that remembers: every inspection is
ingested as a natural-language event, XTrace's extraction pipeline turns it
into searchable facts with belief-revision chains, and Seefu queries those
facts to answer the questions a head chef would ask:

  * "Is this the first missing garnish today, or the fifth?"  -> recall_insight
    escalates a one-off miss into a systemic alert (portioner empty, station
    misloaded) by counting semantically similar past failures.
  * "What did the chef say about this dish last time?"        -> corrections are
    ingested as directive memories (outcome=resolved) and surfaced through the
    unmetered /trigger endpoint before every verdict.
  * "What should the next shift watch?"                       -> shift_summary
    uses compose mode, which returns a ready-to-display markdown briefing
    built from the shift's episodes and facts.

Wire contract (verified against docs.xtrace.ai API reference):
  POST /v1/memories         ingest {messages, user_id, conv_id, ...} (async job)
  POST /v1/memories/search  {query, user_id|group_ids, mode: retrieve|compose}
  POST /v1/memories/trigger {action|entities, ...}  (unmetered)
Auth: Bearer XTRACE_API_KEY. Everything here is graceful: no key or a slow
network degrades to "no memory", never to a failed inspection.
"""

import os
import threading
import time
from datetime import datetime, timezone

BASE = "https://api.production.xtrace.ai"
AGENT_ID = "seefu-inspector"
STATION = os.environ.get("SEEFU_STATION", "pass-1")

_last_error = None


def _key():
    return os.environ.get("XTRACE_API_KEY")


def enabled():
    return bool(_key())


def _headers():
    return {"Authorization": f"Bearer {_key()}"}


def shift_id(now=None):
    """conv_id keys memories to a shift, so XTrace's episode extraction
    naturally produces per-shift summaries."""
    now = now or datetime.now()
    half = "am" if now.hour < 15 else "pm"
    return f"shift_{now.strftime('%Y_%m_%d')}_{half}"


def station_user():
    return f"station:{STATION}"


def _get(path, params, timeout=12.0):
    global _last_error
    import httpx
    try:
        r = httpx.get(f"{BASE}{path}", headers=_headers(), params=params,
                      timeout=timeout)
        if r.status_code >= 400:
            _last_error = f"{path} -> {r.status_code}: {r.text[:160]}"
            return None
        _last_error = None
        return r.json()
    except Exception as e:
        _last_error = f"{path} -> {type(e).__name__}: {e}"
        return None


def usage():
    """XTrace's own meter for this key: proof the memory calls land."""
    if not enabled():
        return None
    return _get("/v1/usage", {})


def list_all(limit=100):
    """Every memory XTrace holds for this station, newest first, with the pool
    (shift, chef-corrections, verdict-feedback) it was filed under."""
    if not enabled():
        return []
    out = []
    for mem_type in ("fact", "episode", "lesson", "procedure"):
        resp = _get("/v1/memories", {"user_id": station_user(), "type": mem_type,
                                     "limit": limit})
        for m in (resp or {}).get("data", []):
            out.append({"id": m.get("id"), "type": m.get("type"),
                        "text": m.get("text"), "created_at": m.get("created_at"),
                        "conv_id": m.get("conv_id"),
                        "updated_at": m.get("updated_at")})
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out[:limit]


def _post(path, payload, timeout, params=None):
    global _last_error
    import httpx
    try:
        r = httpx.post(f"{BASE}{path}", headers=_headers(), json=payload,
                       params=params, timeout=timeout)
        if r.status_code >= 400:
            _last_error = f"{path} -> {r.status_code}: {r.text[:160]}"
            print(f"XTRACE: {_last_error}", flush=True)
            return None
        _last_error = None
        return r.json()
    except Exception as e:
        _last_error = f"{path} -> {type(e).__name__}: {e}"
        print(f"XTRACE: {_last_error}", flush=True)
        return None


def status():
    return {"enabled": enabled(), "station": station_user(),
            "shift": shift_id(), "last_error": _last_error}


def inspection_text(result):
    """Serialize one inspection into the natural-language event XTrace ingests.
    XTrace extracts facts from conversational text, not structured rows, so
    this sentence IS the schema."""
    dish = result.get("dish_id", "unknown")
    verdict = result.get("verdict", "?")
    parts = [f"Plate check for dish {dish} (sesame beef bowl) at {station_user()}: "
             f"verdict {verdict.upper()}, anomaly score "
             f"{result.get('score')} against threshold {result.get('threshold')}."]
    for f in result.get("findings", []):
        where = f.get("location") or "unmapped region"
        line = f"Failure at the {where}: {f.get('description', 'anomaly')}"
        if f.get("fix"):
            line += f" Fix: {f['fix']}"
        parts.append(line)
    if not result.get("findings") and verdict == "pass":
        parts.append("Plate matched the golden reference and was released.")
    if result.get("note"):
        parts.append(f"Station note: {result['note']}")
    return " ".join(parts)


def record_inspection(result):
    """Fire-and-forget ingest; ingestion takes 3-10s server-side and must never
    sit on the inspection hot path."""
    if not enabled():
        return

    def _ingest():
        _post("/v1/memories", {
            "messages": [{"role": "user", "content": inspection_text(result),
                          "date": datetime.now(timezone.utc).isoformat()}],
            "user_id": station_user(),
            "conv_id": shift_id(),
            "agent_id": AGENT_ID,
        }, timeout=20.0)

    threading.Thread(target=_ingest, daemon=True).start()


CORRECTION_MARK = "Chef standing instruction"


def record_correction(text):
    """A chef's correction becomes a persistent instruction. XTrace's extractor
    decides the memory type itself and files short declaratives as facts, so
    the text carries a distinctive marker that recall can key on; outcome and
    agentic are still sent so the extractor MAY promote it to a directive
    lesson, in which case the unmetered trigger endpoint also surfaces it."""
    if not enabled():
        return None
    content = f"{CORRECTION_MARK} for the sesame beef bowl at {station_user()}: {text}"
    return _post("/v1/memories", {
        "messages": [{"role": "user", "content": content,
                      "date": datetime.now(timezone.utc).isoformat()}],
        "user_id": station_user(),
        "conv_id": "chef-corrections",
        "agent_id": AGENT_ID,
        "outcome": "resolved",
        "agentic": True,
    }, timeout=20.0, params={"wait": "true"})


FEEDBACK_MARK = "Operator verdict feedback"


def record_feedback(case, correct, note=""):
    """The right/wrong buttons: an operator's judgement of a verdict, filed so
    future inspections can consult it. A wrong HOLD teaches the station what
    the kitchen considers fine; a wrong PASS becomes a watch item."""
    if not enabled():
        return None
    verdict = str(case.get("verdict", "?")).upper()
    call = "RIGHT" if correct else "WRONG"
    findings = "; ".join(f.get("description", "") for f in case.get("findings", [])) \
        or "no findings"
    content = (f"{FEEDBACK_MARK}: the {verdict} verdict on dish "
               f"{case.get('dish_id', '?')} was {call}. Findings were: {findings}.")
    if note:
        content += f" Operator note: {note}"
    return _post("/v1/memories", {
        "messages": [{"role": "user", "content": content,
                      "date": datetime.now(timezone.utc).isoformat()}],
        "user_id": station_user(),
        "conv_id": "verdict-feedback",
        "agent_id": AGENT_ID,
        "outcome": "resolved",
    }, timeout=20.0, params={"wait": "true"})


def count_overrules(finding_desc):
    """How many times operators called a HOLD with a similar finding WRONG.
    Two or more concordant overrules teach the station to stop holding plates
    for this; a single one is treated as a possible misclick."""
    if not enabled():
        return 0
    resp = search(f"{FEEDBACK_MARK}: hold verdict was wrong for: {finding_desc}",
                  mode="retrieve", limit=8, include=["fact"])
    if not resp:
        return 0
    hits = 0
    for m in resp.get("data") or []:
        text = (m.get("text") or "").lower()
        if (m.get("score") or 0) < 0.4:
            continue
        if ("feedback" in text or "operator" in text) and "wrong" in text \
                and ("defect" in text or "hold" in text):
            hits += 1
    return hits


def watch_items():
    """Findings operators said a PASS wrongly missed: surfaced as warnings on
    future passes so a learned blind spot cannot stay silent."""
    if not enabled():
        return []
    resp = search(f"{FEEDBACK_MARK}: a pass verdict was wrong, the plate had a miss",
                  mode="retrieve", limit=6, include=["fact"])
    if not resp:
        return []
    items = []
    for m in resp.get("data") or []:
        text = (m.get("text") or "")
        low = text.lower()
        if (m.get("score") or 0) >= 0.4 and "wrong" in low and "pass" in low \
                and ("feedback" in low or "operator" in low):
            items.append(text)
    return items[:3]


def search(query, mode="retrieve", limit=8, include=None):
    if not enabled():
        return None
    payload = {"query": query[:4000], "user_id": station_user(),
               "agent_id": AGENT_ID, "mode": mode, "limit": limit}
    if include:
        payload["include"] = include
    return _post("/v1/memories/search", payload, timeout=15.0)


def trigger_lessons(dish_id):
    """Pre-verdict recall of chef corrections for this dish/station.

    Two paths, cheap first: the unmetered /trigger endpoint returns directive
    lessons when XTrace's extractor promoted a correction to one; the reliable
    path is a semantic search keyed on the correction marker, because short
    declarative corrections are filed as plain facts."""
    if not enabled():
        return []
    lessons = []
    resp = _post("/v1/memories/trigger", {
        "action": {"tool": "inspect_dish",
                   "args": {"dish": "sesame beef bowl", "station": STATION,
                            "dish_id": str(dish_id)}},
        "user_id": station_user(),
        "agent_id": AGENT_ID,
    }, timeout=8.0)
    if resp:
        data = resp.get("data") or resp.get("memories") or []
        lessons = [m.get("text", "") for m in data if m.get("text")]
    if not lessons:
        resp = search(f"{CORRECTION_MARK} for this dish", mode="retrieve",
                      limit=5, include=["fact"])
        if resp:
            mark = CORRECTION_MARK.lower()
            lessons = [m.get("text", "") for m in (resp.get("data") or [])
                       if mark in (m.get("text", "").lower())
                       or "instruction" in (m.get("text", "").lower())]
    return lessons[:3]


def recall_insight(result):
    """The memory beat on a defect verdict: how often has THIS failure happened
    before? A semantic search over past inspection facts, counted and phrased
    for the pass display. Returns a dict for result['memory'] or None."""
    if not enabled():
        return None
    findings = result.get("findings") or []
    if result.get("verdict") != "defect" or not findings:
        lessons = trigger_lessons(result.get("dish_id"))
        watch = watch_items()
        if lessons or watch:
            return {"recurrences": 0, "insight": None, "lessons": lessons,
                    "watch": watch}
        return None
    primary = findings[0].get("description", "plate failure")
    resp = search(f"past plate failures like: {primary}", mode="retrieve", limit=8,
                  include=["fact"])
    matches = []
    if resp:
        # facts describing a problem (not the dish-identity or fix-only facts),
        # kept when semantically close to the current failure
        problem_words = ("missing", "fail", "defect", "contamin", "foreign",
                         "wrong", "short", "smear", "absent", "held", "issue")
        matches = [m for m in (resp.get("data") or [])
                   if (m.get("score") or 0) >= 0.4
                   and any(w in (m.get("text", "").lower()) for w in problem_words)]
    lessons = trigger_lessons(result.get("dish_id"))
    n = len(matches)
    if n >= 2:
        insight = (f"This failure matches {n} earlier plates in kitchen memory. "
                   "A repeating miss on the same element usually means the robot's "
                   "ingredient station is empty or misloaded, not a one-off slip. "
                   "Check the line before the next plate fires.")
    elif n == 1:
        insight = ("One similar failure is already in kitchen memory this shift. "
                   "A second will flag this as a line problem.")
    else:
        insight = None
    return {"recurrences": n, "insight": insight, "lessons": lessons,
            "matched": [m.get("text", "")[:140] for m in matches[:3]]}


def shift_briefing():
    """Compose-mode summary of the shift: XTrace selects the relevant episodes
    and facts and returns display-ready markdown in `context`."""
    if not enabled():
        return {"enabled": False, "context": None}
    resp = search("what went wrong on the pass this shift, which dishes failed, "
                  "what should the next shift watch for",
                  mode="compose", limit=12)
    if resp is None:
        return {"enabled": True, "context": None, "error": _last_error}
    return {"enabled": True,
            "context": resp.get("context"),
            "facts": [m.get("text", "") for m in (resp.get("data") or [])[:8]]}


def recent_memories(limit=10):
    if not enabled():
        return {"enabled": False, "memories": []}
    resp = search("plate check failures and fixes on this station",
                  mode="retrieve", limit=limit)
    mems = [{"text": m.get("text", ""), "type": m.get("type"),
             "score": m.get("score"), "created_at": m.get("created_at")}
            for m in ((resp or {}).get("data") or [])]
    return {"enabled": True, "memories": mems}


def wait_for_ingest(seconds=12.0):
    """Test helper: ingestion is async server-side; give extraction a moment."""
    time.sleep(seconds)
