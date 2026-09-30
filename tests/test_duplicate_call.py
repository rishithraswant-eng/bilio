import pytest
import asyncio
from agent.agent import ParticipantAgent

@pytest.mark.asyncio
async def test_duplicate_call_invariant():
    in_q, out_q = asyncio.Queue(), asyncio.Queue()
    agent = ParticipantAgent(in_q, out_q)
    agent.tools = {
        "cancel_booking": {"kind": "state_modifying", "args": {"booking_id": {"type": "string"}}},
        "book_flight": {"kind": "state_modifying", "args": {"flight_id": {"type": "string"}}},
    }
    # No kind specified -> should default to state modifying for these!
    # Wait, if we call them twice, they should NOT hit cache.
    agent.state["intent"] = "cancel_booking"
    agent.read_keys.add(agent.op_key("cancel_booking", {"booking_id": "B1"}))
    await agent.start_task("cancel_booking", "", {"booking_id": "B1"})
    # wait, start_task adds it to inflight!
    assert len(agent.inflight) == 1
