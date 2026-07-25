"""The manager voice agent: token mint + the HTTP endpoints its tools call.

The browser page (agent/index.html) fetches POST /agent/token for an ephemeral
client secret, connects to OpenAI's Realtime API over WebRTC, and dispatches
the model's function calls back to these endpoints, so every claim the voice
makes is grounded in the same record and memory the rest of Seefu uses.

Realtime shapes verified against the installed openai SDK (client_secrets
mint with session.type="realtime", voice under audio.output, FLAT tool schema).
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import history, memory

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter()

REALTIME_MODEL = "gpt-realtime-2.1"
VOICE = "marin"

TOOLS = [
    {"type": "function", "name": "get_latest_case",
     "description": "The most recent plate check: verdict, findings, fix, status.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "get_shift_summary",
     "description": "This shift's counts: plates checked, released, held, and "
                    "the zones failing most.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "get_defect_history",
     "description": "Past failures recorded for one zone of the plate, newest "
                    "first. Zones are spoken locations like 'centre' or "
                    "'upper right'.",
     "parameters": {"type": "object",
                    "properties": {"zone": {"type": "string"}},
                    "required": ["zone"]}},
    {"type": "function", "name": "memory_trends",
     "description": "The analyst's read over the kitchen's whole memory: the "
                    "long-running issues, their evidence, root causes and fixes. "
                    "Use this to open the briefing.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "memory_recall",
     "description": "Semantic search over everything the kitchen remembers: "
                    "past failures, fixes, corrections, feedback.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"type": "function", "name": "teach_correction",
     "description": "Store a standing instruction the manager just gave, so it "
                    "applies to future plates of this dish.",
     "parameters": {"type": "object",
                    "properties": {"instruction": {"type": "string"}},
                    "required": ["instruction"]}},
]


@router.get("/agent")
def agent_page():
    return FileResponse(ROOT / "agent/index.html")


@router.post("/agent/token")
def mint_token():
    """Ephemeral client secret for the browser WebRTC session."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not set")
    instructions = (ROOT / "agent/prompt.md").read_text()
    try:
        from openai import OpenAI
        secret = OpenAI().realtime.client_secrets.create(
            expires_after={"anchor": "created_at", "seconds": 600},
            session={
                "type": "realtime",
                "model": REALTIME_MODEL,
                "instructions": instructions,
                "audio": {"output": {"voice": VOICE}},
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        )
    except Exception as e:
        raise HTTPException(503, f"token mint failed: {type(e).__name__}: {e}")
    return {"value": secret.value, "expires_at": secret.expires_at,
            "model": REALTIME_MODEL}


def _latest_case():
    cases = sorted((ROOT / "runs/cases").glob("*.json"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not cases:
        return None
    return json.loads(cases[-1].read_text())


@router.post("/agent/tools/get_latest_case")
def tool_latest_case():
    case = _latest_case()
    if case is None:
        return {"case": None, "message": "no plates checked yet"}
    keys = ("case_id", "dish_id", "verdict", "status", "findings", "note",
            "verified", "created_at")
    return {"case": {k: case[k] for k in keys if k in case}}


@router.post("/agent/tools/get_shift_summary")
def tool_shift_summary():
    return history.shift_summary()


class DefectHistoryArgs(BaseModel):
    zone: str


@router.post("/agent/tools/get_defect_history")
def tool_defect_history(args: DefectHistoryArgs):
    rows = history.defect_history(args.zone)
    return {"zone": args.zone, "count": len(rows), "defects": rows}


@router.post("/agent/tools/memory_trends")
def tool_memory_trends():
    import sys
    sys.path.insert(0, str(ROOT / "inspection"))
    from trends import analyze_trends
    return analyze_trends()


class RecallArgs(BaseModel):
    query: str


@router.post("/agent/tools/memory_recall")
def tool_memory_recall(args: RecallArgs):
    resp = memory.search(args.query, mode="retrieve", limit=8)
    if resp is None:
        return {"memories": [], "note": "kitchen memory unavailable"}
    return {"memories": [m.get("text", "") for m in resp.get("data", [])
                         if m.get("text")]}


class TeachArgs(BaseModel):
    instruction: str


@router.post("/agent/tools/teach_correction")
def tool_teach_correction(args: TeachArgs):
    text = args.instruction.strip()
    if not text:
        raise HTTPException(400, "empty instruction")
    resp = memory.record_correction(text)
    return {"stored": resp is not None,
            "note": ("standing instruction stored; it surfaces on future plates"
                     if resp is not None else "kitchen memory unavailable")}
