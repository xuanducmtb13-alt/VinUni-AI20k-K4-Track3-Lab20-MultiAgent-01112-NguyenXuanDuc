"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        import os
        
        max_iterations = int(os.getenv("MAX_ITERATIONS", "6"))
        
        if state.iteration >= max_iterations:
            state.record_route("FINISH")
            state.errors.append("Max iterations reached")
            return state
            
        if not state.research_notes:
            state.record_route("researcher")
        elif not state.analysis_notes:
            state.record_route("analyst")
        elif not state.final_answer:
            state.record_route("writer")
        else:
            state.record_route("FINISH")
            
        return state
