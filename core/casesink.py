"""CaseSink interface + LocalSink.

Every inspection result flows through a CaseSink, so tomorrow's ZendeskSink is
a second implementation behind SEEFU_SINK=zendesk|local - not a rewire.
LocalSink: one JSON per case in runs/cases/, and every mutation broadcast to
the dashboard as a verbatim {type:"case", ...} message.
Statuses: autonomous_resolution, escalated, quarantined, resolved.
"""

import json
import threading
import time
from pathlib import Path

from core import history, hub

_seq_lock = threading.Lock()

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "runs/cases"
CASE_SEQ = ROOT / "runs/.case_seq"

STATUSES = ("autonomous_resolution", "escalated", "quarantined", "resolved")


class CaseSink:
    """Interface. Implementations: LocalSink (tonight), ZendeskSink (tomorrow)."""

    def create_case(self, dish_id, verdict, findings, overlay_path) -> str:
        raise NotImplementedError

    def add_event(self, case_id, text):
        raise NotImplementedError

    def set_status(self, case_id, status):
        raise NotImplementedError

    def set_verified(self, case_id, verified, note=""):
        raise NotImplementedError


class LocalSink(CaseSink):

    def _path(self, case_id):
        return CASES / f"{case_id}.json"

    def _load(self, case_id):
        return json.loads(self._path(case_id).read_text())

    def _save(self, case):
        CASES.mkdir(parents=True, exist_ok=True)
        self._path(case["case_id"]).write_text(json.dumps(case, indent=2) + "\n")

    def _emit(self, case):
        # the dashboard counts every case message as a transition and fills the
        # Case note panel (with its hardcoded VERIFIED badge) whenever a note is
        # present - so emit deliberately: notes only ride along when the case is
        # actually verified or carries an operator-facing disposition note
        msg = {"type": "case", "case_id": case["case_id"], "status": case["status"],
               "verified": case["verified"]}
        if case.get("zendesk_id"):
            msg["zendesk_id"] = case["zendesk_id"]
        if case.get("note") and (case["verified"] or
                                 case["status"] in ("escalated", "quarantined")):
            msg["note"] = case["note"]
        hub.emit(msg)

    def _next_id(self):
        with _seq_lock:                          # threadpool endpoints can race the file
            CASES.mkdir(parents=True, exist_ok=True)
            seq = int(CASE_SEQ.read_text()) if CASE_SEQ.exists() else 1041
            CASE_SEQ.write_text(str(seq + 1))
        return str(seq)

    def create_case(self, dish_id, verdict, findings, overlay_path):
        case = {
            "case_id": self._next_id(),
            "dish_id": dish_id,
            "verdict": verdict,
            "status": "autonomous_resolution" if verdict == "pass" else "escalated",
            "verified": False,
            "note": "",
            "findings": findings,
            "overlay_path": str(overlay_path),
            "events": [{"ts": time.strftime("%H:%M:%S"), "text": "case created"}],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save(case)
        history.update_status(dish_id, case["status"])
        self._emit(case)
        return case["case_id"]

    def add_event(self, case_id, text):
        case = self._load(case_id)
        case["events"].append({"ts": time.strftime("%H:%M:%S"), "text": text})
        self._save(case)

    def set_status(self, case_id, status, note=None):
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}; expected one of {STATUSES}")
        case = self._load(case_id)
        unchanged = case["status"] == status
        case["status"] = status
        if note:
            case["note"] = note                  # operator-facing disposition note
        case["events"].append({"ts": time.strftime("%H:%M:%S"), "text": f"status -> {status}"})
        self._save(case)
        if unchanged and not note:
            return                               # no transition: don't re-count on the dashboard
        history.update_status(case["dish_id"], status)
        self._emit(case)
        hub.emit({"type": "shift", **history.shift_summary()})

    def set_verified(self, case_id, verified, note=""):
        case = self._load(case_id)
        case["verified"] = bool(verified)
        if note and verified:
            case["note"] = note                  # failure notes stay in events only
        case["events"].append({"ts": time.strftime("%H:%M:%S"),
                               "text": f"verified={verified} {note}".strip()})
        self._save(case)
        history.update_verified(case["dish_id"], verified)
        if verified:
            self._emit(case)                     # dashboard needs the VERIFIED beat
        # unverified: no emit - create_case already showed the status, and the
        # dashboard would double-count the same status + show a false VERIFIED badge


def get_sink():
    """SEEFU_SINK env var picks the implementation; local is the fallback."""
    import os
    if os.environ.get("SEEFU_SINK", "local") == "zendesk":
        from core.zendesk_sink import ZendeskSink   # built tomorrow morning
        return ZendeskSink()
    return LocalSink()
