"""Regression tests for the audit report (B02–B27). Drive the agent directly through
its two queues, injecting tool results by hand so races are deterministic.

    python -m pytest -q tests/
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import nlu  # noqa: E402
from agent import perception as P  # noqa: E402
from agent.agent import ParticipantAgent  # noqa: E402
from harness.mock_env import TOOL_REGISTRY  # noqa: E402

RENTAL = {"reserve_rental_car": {"kind": "state_modifying", "description": "Reserve a rental car at a city's airport.",
                                 "args": {"city": {"type": "string", "required": True},
                                          "car_class": {"type": "string", "required": True,
                                                        "enum": ["compact", "suv", "luxury"]}}}}


class Rig:
    def __init__(self, tools=None, live=False):
        self.inq, self.outq = asyncio.Queue(), asyncio.Queue()
        self.agent = ParticipantAgent(self.inq, self.outq, live=live)
        self.out = []
        self.tools = tools or dict(TOOL_REGISTRY)

    async def __aenter__(self):
        self.task = asyncio.create_task(self.agent.run())
        await self.send("tool_manifest", tools=self.tools)
        return self

    async def __aexit__(self, *a):
        self.task.cancel()

    async def send(self, et, **payload):
        await self.inq.put({"event_type": et, "payload": payload})
        await self.settle()

    async def say(self, text):
        await self.send("user_speech_chunk", text=text, end_of_turn=True)

    async def settle(self, t=0.05):
        await asyncio.sleep(t)
        while not self.outq.empty():
            self.out.append(self.outq.get_nowait())

    def calls(self, api=None):
        return [a["payload"] for a in self.out if a["action"] == "tool_call"
                and (api is None or a["payload"]["api_name"] == api)]

    def spoken(self, kind=None):
        return [a["payload"]["text"] for a in self.out if a["action"] in
                ("filler_speech", "clarification_request", "final_response") and (kind is None or a["action"] == kind)]

    def cancels(self):
        return [a["payload"]["call_id"] for a in self.out if a["action"] == "cancel_tool"]

    async def result(self, call, **res):
        status = res.pop("status", "success")
        await self.send("tool_result", call_id=call["call_id"], api_name=call["api_name"], status=status,
                        result={"status": status, **res})


def flights(code):
    return {"flights": [{"flight_id": f"FL-{code}-8AM", "depart": "08:00", "price_usd": 129},
                        {"flight_id": f"FL-{code}-2PM", "depart": "14:00", "price_usd": 99}]}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- B02 slow perception
def test_b02_interruption_processed_while_asr_running(monkeypatch):
    def slow(ref, prompt=""):
        time.sleep(0.6)
        return {"text": "find flights to Boston", "words": [("Boston", 0.99)], "ok": True, "alt_text": ""}
    monkeypatch.setattr(P, "transcribe", slow)

    async def go():
        async with Rig() as r:
            await r.send("user_audio_chunk", audio_ref="x.mp3", end_of_turn=True)
            t0 = time.monotonic()
            await r.send("interruption", text="Never mind.")
            assert time.monotonic() - t0 < 0.3             # consumer not blocked by ASR
            assert any("stopped" in s for s in r.spoken("final_response"))
            await r.settle(0.9)
            assert not r.calls("flight_search")            # superseded ASR never becomes a call
    run(go())


def test_b02_manual_lookup_does_not_block_on_vision(monkeypatch):
    def slow(ref):
        time.sleep(0.6)
        return {"label": "HDMI port", "confidence": 0.95, "embedding": [0.1] * 4, "source": "ocr+clip"}
    monkeypatch.setattr(P, "analyze_frame", slow)

    async def go():
        async with Rig() as r:
            await r.send("video_frame", image_ref="f.png", frame_id="f1")
            await r.say("What is this port used for?")
            t0 = time.monotonic()
            await r.send("interruption", text="Never mind.")
            assert time.monotonic() - t0 < 0.3
            await r.settle(0.9)
            assert not r.calls("lookup_manual")
    run(go())


# ---------------------------------------------------------------- B03/B04 stale results
def test_b03_b04_old_result_not_relabelled():
    async def go():
        async with Rig() as r:
            await r.say("Find flights to Boston")
            bos = r.calls("flight_search")[0]
            await r.say("Find flights to Seattle")        # normal new turn, not an interruption
            assert bos["call_id"] in r.cancels()
            await r.result(bos, **flights("BOS"))
            assert not any("FL-BOS" in s and "Seattle" in s for s in r.spoken())
            sea = r.calls("flight_search")[-1]
            assert sea["args"]["destination"] == "Seattle"
            await r.result(sea, **flights("SEA"))
            assert "Seattle" in r.spoken("final_response")[-1] and "FL-SEA" in r.spoken("final_response")[-1]
    run(go())


def test_b03_late_invalidated_result_ignored_even_if_not_cancelled():
    async def go():
        async with Rig() as r:
            await r.say("Find flights to Boston")
            bos = r.calls("flight_search")[0]
            r.agent.inflight[bos["call_id"]]["deps"] = {"destination": "Boston"}
            r.agent.state["slots"]["destination"] = "Seattle"   # state moved on
            await r.result(bos, **flights("BOS"))
            assert not r.spoken("final_response")
    run(go())


# ---------------------------------------------------------------- B05 negation
def test_b05_do_not_book_never_books():
    async def go():
        for text in ("Do not book a flight to Boston for Priya.", "Only search flights to Boston, don't book.",
                     "Show me the options to Boston without booking."):
            async with Rig() as r:
                await r.say(text)
                assert r.agent.plan == []
                s = r.calls("flight_search")
                if s:
                    await r.result(s[0], **flights("BOS"))
                assert not r.calls("book_flight"), text
    run(go())


# ---------------------------------------------------------------- B06/B07 retraction
def test_b06_retraction_clears_partial_buffer():
    async def go():
        async with Rig() as r:
            await r.send("user_speech_chunk", text="Book a flight to Boston", end_of_turn=False)
            await r.send("interruption", text="Never mind.")
            await r.send("user_speech_chunk", text="for Alice.", end_of_turn=True)
            assert not r.calls("flight_search") and not r.calls("book_flight")
    run(go())


def test_b07_spoken_never_mind_cancels_inflight():
    async def go():
        async with Rig() as r:
            await r.say("Find flights to Miami")
            c = r.calls("flight_search")[0]
            await r.say("Never mind.")
            assert c["call_id"] in r.cancels()
            assert r.agent.state["intent"] == "cancelled"
    run(go())


# ---------------------------------------------------------------- B08 correction mid-booking
def test_b08_correction_during_booking_replans():
    async def go():
        async with Rig() as r:
            await r.say("Find a flight to Denver and book the 8 AM one for Alice.")
            await r.result(r.calls("flight_search")[0], **flights("DEN"))
            book = r.calls("book_flight")[0]
            await r.send("interruption", text="Wait, actually make it Seattle.")
            assert book["call_id"] in r.cancels()
            sea = r.calls("flight_search")[-1]
            assert sea["args"]["destination"] == "Seattle"
            await r.result(sea, **flights("SEA"))
            b2 = r.calls("book_flight")[-1]
            assert b2["args"]["flight_id"] == "FL-SEA-8AM"
            await r.result(b2, booking_id="BK-0002", flight_id="FL-SEA-8AM")
            assert "BK-0002" in r.spoken("final_response")[-1]
    run(go())


# ---------------------------------------------------------------- B09 idempotency ledger
def test_b09_failed_op_can_be_retried_and_success_is_reported():
    async def go():
        async with Rig(tools={**TOOL_REGISTRY, **RENTAL}) as r:
            await r.say("Reserve a rental car in Denver, compact class.")
            c1 = r.calls("reserve_rental_car")[0]
            await r.result(c1, status="error", error="unavailable")
            await r.say("Reserve a rental car in Denver, compact class.")
            c2 = r.calls("reserve_rental_car")
            assert len(c2) == 2                               # failed op not permanently blocked
            await r.result(c2[-1], reservation_id="RC-77")
            n = len(r.out)
            await r.say("Reserve a rental car in Denver, compact class.")
            assert len(r.calls("reserve_rental_car")) == 2    # no duplicate side effect
            assert len(r.out) > n and "already" in r.spoken()[-1].lower()  # …but not silent
    run(go())


# ---------------------------------------------------------------- B10 intent switch w/ city
def test_b10_weather_switch_with_city():
    weather = {"weather_lookup": {"kind": "read_only", "description": "Current weather for a city.",
                                  "args": {"city": {"type": "string", "required": True}}}}
    async def go():
        async with Rig(tools={**TOOL_REGISTRY, **weather}) as r:
            await r.say("Find flights to Boston")
            await r.send("interruption", text="What is the weather in Seattle?")
            assert r.calls("weather_lookup") and r.calls("weather_lookup")[-1]["args"]["city"] == "Seattle"
            assert len(r.calls("flight_search")) == 1
    run(go())


# ---------------------------------------------------------------- B11 targeted cancel
def test_b11_destination_change_does_not_cancel_manual_lookup():
    async def go():
        async with Rig() as r:
            await r.say("Tell me about USB ports in the laptop manual.")
            man = r.calls("lookup_manual")[0]
            await r.say("Find flights to Boston")
            fl = r.calls("flight_search")[0]
            await r.send("interruption", text="Actually make it Seattle.")
            assert fl["call_id"] in r.cancels()
            assert man["call_id"] not in r.cancels()
    run(go())


# ---------------------------------------------------------------- B12/B13 clarification
def test_b12_enum_clarification_resumes():
    async def go():
        async with Rig(tools={**TOOL_REGISTRY, **RENTAL}) as r:
            await r.say("Reserve a rental car in Denver.")
            assert r.spoken("clarification_request")
            await r.say("SUV.")
            c = r.calls("reserve_rental_car")[0]
            assert c["args"] == {"city": "Denver", "car_class": "suv"}
    run(go())


def test_b13_yes_does_not_pick_between_two():
    async def go():
        async with Rig() as r:
            r.agent.pending_clarify = {"field": "destination", "candidates": ["Austin", "Boston"],
                                       "text": "book a flight to", "api": None, "args": {}}
            await r.say("yes")
            assert not r.calls()
            assert "Austin or Boston" in r.spoken("clarification_request")[-1]
            await r.say("Boston")
            assert r.calls("flight_search")[0]["args"]["destination"] == "Boston"
    run(go())


# ---------------------------------------------------------------- B14–B18 NLU
def test_b14_names():
    assert nlu.parse_name_answer("priya sharma") == "Priya Sharma"
    assert nlu.extract_name("for Alice, actually for Priya") == "Priya"

    async def go():
        async with Rig() as r:
            await r.say("Book a flight to Denver")
            await r.result(r.calls("flight_search")[0], **flights("DEN"))
            await r.say("priya sharma")
            assert r.calls("book_flight")[0]["args"]["passenger_name"] == "Priya Sharma"
    run(go())


def test_b15_greeting_does_not_override_request():
    async def go():
        async with Rig() as r:
            await r.say("Hello, find flights to Boston.")
            assert r.calls("flight_search")
    run(go())


def test_b16_city_roles():
    assert nlu.extract_city("find flights to kochi") == "Kochi"
    assert nlu.extract_city("find flights to Kochi for Priya") == "Kochi"
    assert nlu.extract_city("flights to Boston from Seattle") == "Boston"
    assert nlu.extract_origin("flights to Boston from Seattle") == "Seattle"
    spec = {"args": {"origin": {"type": "string", "required": True},
                     "destination": {"type": "string", "required": True}}}
    args, _ = nlu.build_args(spec, "flights to Boston from Seattle", {})
    assert args == {"origin": "Seattle", "destination": "Boston"}


def test_b17_time_minutes_and_no_silent_substitution():
    assert nlu.extract_time("book the 2:45 PM one") == "14:45"

    async def go():
        async with Rig() as r:
            await r.say("Find a flight to Denver and book the 2:45 PM one for Alice.")
            await r.result(r.calls("flight_search")[0], **flights("DEN"))
            assert not r.calls("book_flight")
            assert r.spoken("clarification_request")
    run(go())


def test_b18_explicit_booking_id():
    async def go():
        async with Rig() as r:
            await r.say("Cancel booking BK-0001.")
            c = r.calls("cancel_booking")
            assert c and c[0]["args"]["booking_id"] == "BK-0001"
    run(go())


# ---------------------------------------------------------------- B19 schema types
def test_b19_schema_bool_array_numbers_nested():
    spec = {"args": {"nights": {"type": "integer", "required": True, "description": "number of nights"},
                     "guests": {"type": "integer", "required": True, "description": "number of guests"},
                     "breakfast": {"type": "boolean", "required": True},
                     "amenities": {"type": "array", "items": {"enum": ["wifi", "pool", "gym"]}},
                     "contact": {"type": "object", "required": True,
                                 "properties": {"city": {"type": "string", "required": True}}}}}
    args, missing = nlu.build_args(spec, "Hotel in Seattle for 3 nights, 2 guests, no breakfast, with wifi and pool", {})
    assert args["nights"] == 3 and args["guests"] == 2
    assert args["breakfast"] is False
    assert args["amenities"] == ["wifi", "pool"]
    assert args["contact"] == {"city": "Seattle"} and not missing


# ---------------------------------------------------------------- B20 manifest guard
def test_b20_no_undeclared_tool_calls():
    async def go():
        tools = {"flight_search": TOOL_REGISTRY["flight_search"]}
        async with Rig(tools=tools) as r:
            await r.say("My TV is broken, please open a support ticket.")
            await r.say("What does the LED on this manual page mean?")
            assert all(c["api_name"] in tools for c in r.calls())
    run(go())


# ---------------------------------------------------------------- B21 fillers
def test_b21_interruption_ack_after_many_fillers():
    async def go():
        async with Rig() as r:
            await r.say("Find flights to Boston")
            await r.send("interruption", text="Actually make it Seattle.")
            await r.send("interruption", text="Actually make it Denver.")
            await r.send("interruption", text="Actually make it Miami.")
            fill = r.spoken("filler_speech")
            assert len(fill) <= 3 and any("Denver" in f or "Miami" in f for f in fill)
    run(go())


# ---------------------------------------------------------------- B23/B24 vision
def test_b23_low_confidence_label_not_used(monkeypatch):
    monkeypatch.setattr(P, "analyze_frame", lambda ref: {"label": "HDMI port", "confidence": 0.01,
                                                         "embedding": None, "source": "clip"})
    async def go():
        async with Rig() as r:
            await r.send("video_frame", image_ref="f.png")
            await r.settle(0.1)
            await r.say("What is this port used for?")
            await r.settle(0.1)
            calls = r.calls("lookup_manual")
            assert len(calls) == 0
    run(go())


def test_b24_text_manual_question_needs_no_camera():
    async def go():
        async with Rig() as r:
            await r.say("Tell me about USB ports in the laptop manual.")
            c = r.calls("lookup_manual")[0]
            await r.result(c, pages=[{"doc": "GENERIC-laptop-manual", "page": 23, "title": "USB Ports"},
                                     {"doc": "GENERIC-laptop-manual", "page": 27, "title": "HDMI Output"}])
            fin = r.spoken()[-1]
            assert "camera" not in fin.lower() and "23" in fin
    run(go())


# ---------------------------------------------------------------- B27 ASR clarification relevance
def test_b27_unclear_support_audio_asks_to_repeat(monkeypatch):
    monkeypatch.setattr(P, "transcribe", lambda ref, prompt="": {"text": "", "words": [], "ok": False})
    async def go():
        async with Rig() as r:
            await r.say("My TV is blinking red, please open a support ticket.")
            await r.send("user_audio_chunk", audio_ref="x.mp3", end_of_turn=True)
            await r.settle(0.2)
            q = r.spoken("clarification_request")[-1].lower()
            assert "city" not in q and "again" in q
    run(go())
