"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.core.schemas import AgentResult

        llm_client = LLMClient()
        
        system_prompt = f"You are an expert technical writer. Write a comprehensive response for '{state.request.audience}'. Include inline citations referencing the sources."
        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes}\n\n"
            f"Analysis Notes:\n{state.analysis_notes}\n\n"
            "Please write the final answer synthesizing all the information above."
        )
        
        response = llm_client.complete(system_prompt, user_prompt)
        state.final_answer = response.content
        
        state.agent_results.append(AgentResult(
            agent=self.name,
            content=response.content,
            metadata={"cost_usd": response.cost_usd}
        ))
        
        return state
