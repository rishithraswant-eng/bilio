"""Evaluation console routes (development only): replay bundled scenarios through the harness + scorer.

Mounted by ui/api.py when TRIAGELINE_ENV != production and TRIAGELINE_CONSOLE != 0.
UI: /console  (ui/static/index.html).  API: GET /api/scenarios, POST /api/run.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
AGENTS = {"bilio": "agent.agent:ParticipantAgent", "baseline": "agent.agent:BaselineAgent"}
OFFICIAL_TAIL_MS = 6000.0          # identical to run_local.py / runner default
MAX_EVENTS = 60
MAX_SCENARIO_MS = 60_000
_RUN_SEM = threading.BoundedSemaphore(1)   # one scored run at a time keeps virtual timing honest
_QUEUE = threading.BoundedSemaphore(4)     # bounded waiting room; beyond this -> 429


class BadRequest(ValueError):
    pass


def list_scenarios():
    out = []
    for p in sorted(glob.glob(str(ROOT / "scenarios/*.json"))) + sorted(glob.glob(str(ROOT / "scenarios_extra/*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        rel = os.path.relpath(p, ROOT)
        md = d.get("metadata", {})
        refs = [(e.get("payload") or {}).get("audio_ref") or (e.get("payload") or {}).get("image_ref")
                for e in d.get("events", [])]
        out.append({"path": rel, "id": d.get("scenario_id"), "modality": md.get("modality"),
                    "difficulty": md.get("difficulty"), "description": md.get("description", ""),
                    "missing_media": [r for r in refs if r and not (ROOT / r).exists()],
                    "events": [{"t": e.get("timestamp_ms"), "type": e.get("event_type"),
                                "text": (e.get("payload") or {}).get("text")
                                or (e.get("payload") or {}).get("audio_ref")
                                or (e.get("payload") or {}).get("image_ref")} for e in d.get("events", [])]})
    return out


def _num(v, name, default):
    try:
        f = float(default if v is None else v)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a number")
    if f != f or f in (float("inf"), float("-inf")):
        raise BadRequest(f"{name} must be finite")
    return f


def validate_scenario(sc):
    if not isinstance(sc, dict) or not isinstance(sc.get("events"), list):
        raise BadRequest("scenario must be an object with an events list")
    if len(sc["events"]) > MAX_EVENTS:
        raise BadRequest(f"at most {MAX_EVENTS} events")
    for e in sc["events"]:
        if not isinstance(e, dict) or e.get("event_type") not in ("user_speech_chunk", "interruption",
                                                                 "user_audio_chunk", "video_frame"):
            raise BadRequest("unsupported event type")
        if not (0 <= _num(e.get("timestamp_ms"), "timestamp_ms", 0) <= MAX_SCENARIO_MS):
            raise BadRequest("timestamp out of range")
        p = e.get("payload") or {}
        for k in ("audio_ref", "image_ref"):
            if p.get(k) and not re.fullmatch(r"(audio|frames)/[\w.\-]+", str(p[k])):
                raise BadRequest(f"{k} must reference a bundled asset")
        if len(str(p.get("text", ""))) > 500:
            raise BadRequest("text too long")


def run(req: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking scored replay (call from a worker thread)."""
    from harness.runner import run_scenario
    from harness.scorer import score_scenario
    from run_local import load_agent_factory

    if req.get("scenario"):
        sc = req["scenario"]
        validate_scenario(sc)
    else:
        path = str(req.get("path", "")).replace("\\", "/")
        path = os.path.normpath(path).replace("\\", "/")
        if not (re.fullmatch(r"scenarios(_extra)?/[\w\-]+\.json", path) and (ROOT / path).exists()):
            raise BadRequest("bad path")
        with open(ROOT / path) as fh:
            sc = json.load(fh)
    agent = req.get("agent") if req.get("agent") in AGENTS else "bilio"
    factory = load_agent_factory(AGENTS[agent])
    ts = min(max(_num(req.get("time_scale"), "time_scale", 1), 1.0), 8.0)
    if not _QUEUE.acquire(blocking=False):
        raise OverflowError("run queue full — try again in a moment")
    try:
        with _RUN_SEM:
            cwd = os.getcwd()
            os.chdir(ROOT)           # harness resolves audio/ and frames/ relative to the repo
            try:
                trace = run_scenario(sc, factory, time_scale=ts, verbose=False, tail_ms=OFFICIAL_TAIL_MS)
            finally:
                os.chdir(cwd)
    finally:
        _QUEUE.release()
    score = score_scenario(sc, trace) if sc.get("ground_truth") else None
    refs = [(e.get("payload") or {}).get("audio_ref") or (e.get("payload") or {}).get("image_ref")
            for e in sc.get("events", [])]
    return {"trace": trace, "score": score, "scenario_id": sc.get("scenario_id"),
            "config": {"agent": agent, "time_scale": ts, "tail_ms": OFFICIAL_TAIL_MS,
                       "official": ts == 1.0},
            "status": {"runner_stopped_with_pending_calls": [e["call_id"] for e in trace if e.get("kind") == "tool_abandoned"],
                       "agent_crash": [e.get("error") for e in trace if e.get("kind") == "agent_crash"],
                       "missing_media": [r for r in refs if r and not (ROOT / r).exists()]}}


def register(app: FastAPI, limiter) -> None:
    @app.get("/console")
    async def console_page():
        return FileResponse(ROOT / "ui/static/index.html")

    @app.get("/api/scenarios")
    async def scenarios():
        return await asyncio.to_thread(list_scenarios)

    @app.post("/api/run")
    async def run_route(request: Request):
        limiter.check(("run", request.client.host if request.client else "unknown"), 60)
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "invalid JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "JSON body must be an object")
        try:
            return await asyncio.to_thread(run, body)
        except BadRequest as exc:
            return JSONResponse({"error": str(exc), "code": "bad_request"}, 400)
        except OverflowError as exc:
            return JSONResponse({"error": str(exc), "code": "busy"}, 429)
