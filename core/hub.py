"""Websocket hub: fan-out server events to connected dashboard clients.

emit() is safe to call from sync code anywhere in the request path; it schedules
the broadcast onto the server's event loop. Messages emitted before the loop
exists (startup) are queued and flushed on the first client connect.
"""

import asyncio
import json

_clients = set()
_loop = None
_pending = []


def register_loop(loop):
    global _loop
    _loop = loop


async def connect(websocket):
    _clients.add(websocket)
    for msg in list(_pending):
        try:
            await websocket.send_text(msg)
        except Exception:
            break
    _pending.clear()


def disconnect(websocket):
    _clients.discard(websocket)


async def _broadcast(text):
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def emit(msg: dict):
    """Fire-and-forget broadcast of one dashboard message (verbatim shape)."""
    text = json.dumps(msg)
    if _loop is None or not _clients:
        _pending.append(text)
        del _pending[:-50]                     # keep the queue bounded
        return
    asyncio.run_coroutine_threadsafe(_broadcast(text), _loop)
