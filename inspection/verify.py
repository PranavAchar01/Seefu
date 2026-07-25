"""M6: verified resolutions - one GPT call reviews each case for consistency.

Mirrors Zendesk's Relate 2026 "verified resolutions": after a case is created,
the model checks findings vs blobs vs verdict vs disposition for internal
consistency and completeness, returning verified true/false plus one line.
On API failure the case stays verified=false with an honest note - a case is
never marked verified without the check actually running.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "runs/cases"

PROMPT = """You are the verification step of a plate inspection system. Review this
case record for internal consistency and completeness:
- does the verdict match the findings (defect verdicts need findings, pass verdicts need none)?
- do finding locations correspond to blob locations?
- is the disposition (status) appropriate for the verdict?
Reply with STRICT JSON only: {"verified": true|false, "note": "<one line a
technician would read>"}"""


def verify_case(case_id):
    """Returns (verified: bool, note: str). Never raises."""
    path = CASES / f"{case_id}.json"
    if not path.exists():
        return False, "verification skipped: case record missing"
    try:
        case = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, "verification skipped: unreadable case record"
    record = {k: case.get(k) for k in
              ("dish_id", "verdict", "status", "findings", "overlay_path")}

    if not os.environ.get("OPENAI_API_KEY"):
        return False, "verification unavailable: no API key"
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.with_options(timeout=10.0, max_retries=0).responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": [
                {"type": "input_text",
                 "text": PROMPT + "\n\nCASE RECORD:\n" + json.dumps(record, indent=2)}]}],
            max_output_tokens=120,
        )
        text = resp.output_text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        v = data.get("verified")
        verified = v is True or (isinstance(v, str) and v.strip().lower() == "true")
        return verified, str(data.get("note", ""))[:200]
    except Exception as e:                       # any API/parse failure -> honest unverified
        return False, f"verification unavailable: {type(e).__name__}"
