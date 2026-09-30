import pytest
import asyncio
from agent.agent import ParticipantAgent

@pytest.mark.asyncio
async def test_localized_correction_invariant():
    in_q, out_q = asyncio.Queue(), asyncio.Queue()
    agent = ParticipantAgent(in_q, out_q)
    agent.tools = {"book_flight": {"args": {"date": {"type": "string"}, "destination": {"type": "string"}}}}
    agent.state["intent"] = "book_flight"
    agent.state["slots"] = {"destination": "Seattle", "date": "Sunday", "passenger_name": "Alice", "flight_id": "FL-SEA-1"}
    await agent.revise("change departure to Monday", announce=False)
    assert agent.state["slots"]["date"] == "Monday"
    assert agent.state["slots"]["destination"] == "Seattle"
    assert agent.state["slots"]["passenger_name"] == "Alice"
    assert agent.state["slots"]["flight_id"] == "FL-SEA-1"  # INVARIANT 1 says ALL other slots remain unchanged

