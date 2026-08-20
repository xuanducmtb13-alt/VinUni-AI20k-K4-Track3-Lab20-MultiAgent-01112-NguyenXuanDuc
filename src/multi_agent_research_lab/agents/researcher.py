"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        from multi_agent_research_lab.services.search_client import SearchClient
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.core.schemas import AgentResult

        search_client = SearchClient()
        llm_client = LLMClient()
        
        # 1. Search for sources
        docs = search_client.search(state.request.query, max_results=state.request.max_sources)
        state.sources.extend(docs)
        
        # 2. Summarize sources
        context = "\n\n".join([f"[{i+1}] {doc.title} ({doc.url}):\n{doc.snippet}" for i, doc in enumerate(docs)])
        
        system_prompt = "You are an expert researcher. Synthesize the provided search results into concise research notes."
        user_prompt = f"Query: {state.request.query}\n\nSources:\n{context}\n\nPlease write research notes extracting the most relevant facts."
        
        response = llm_client.complete(system_prompt, user_prompt)
        state.research_notes = response.content
        
        # 3. Add to agent results
        state.agent_results.append(AgentResult(
            agent=self.name,
            content=response.content,
            metadata={"sources_count": len(docs), "cost_usd": response.cost_usd}
        ))
        
        return state
