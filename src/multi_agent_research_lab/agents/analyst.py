"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.core.schemas import AgentResult

        llm_client = LLMClient()
        
        system_prompt = "You are a critical analyst. Your job is to extract key claims, compare viewpoints, and flag weak evidence from the provided research notes."
        user_prompt = f"Original Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}\n\nPlease analyze the research and provide structured analysis notes."
        
        response = llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content
        
        state.agent_results.append(AgentResult(
            agent=self.name,
            content=response.content,
            metadata={"cost_usd": response.cost_usd}
        ))
        
        return state
