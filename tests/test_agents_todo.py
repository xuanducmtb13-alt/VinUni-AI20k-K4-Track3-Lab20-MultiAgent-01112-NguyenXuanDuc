"""Skeleton guard test.

NOTE(student): Test này chỉ xác nhận skeleton còn nguyên TODO. Sau khi bạn implement
SupervisorAgent, test này SẼ FAIL - đó là điều bình thường. Hãy xóa hoặc thay thế nó
bằng unit test thật cho routing policy của bạn.
"""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "researcher"

def test_supervisor_routes_to_analyst() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), research_notes="Done")
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "analyst"

def test_supervisor_routes_to_writer() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Done",
        analysis_notes="Done"
    )
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "writer"

def test_supervisor_finishes() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Done",
        analysis_notes="Done",
        final_answer="Done"
    )
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "FINISH"
