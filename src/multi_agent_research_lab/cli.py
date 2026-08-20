"""Command-line entrypoint for the lab starter."""

# Load .env into OS environment BEFORE importing LangChain/LangGraph
# so that LANGCHAIN_TRACING_V2 is visible to the tracing SDK.
from dotenv import load_dotenv
load_dotenv(override=True)

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""
    _init()
    request = _parse_query(query)
    
    def baseline_runner(q: str) -> ResearchState:
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.services.search_client import SearchClient
        from multi_agent_research_lab.core.schemas import AgentResult
        
        st = ResearchState(request=request)
        llm = LLMClient()
        searcher = SearchClient()
        
        # 1. Search
        st.sources = searcher.search(q, max_results=5)
        
        # 2. Write
        context = "\n".join([f"- {s.title}: {s.snippet} ({s.url})" for s in st.sources])
        sys_prompt = "You are a single helpful assistant. Use the provided context to answer the user's research query comprehensively."
        user_prompt = f"Context:\n{context}\n\nQuery: {q}"
        
        res = llm.complete(sys_prompt, user_prompt)
        st.final_answer = res.content
        st.agent_results.append(AgentResult(
            agent="writer",
            content=res.content,
            metadata={"cost_usd": res.cost_usd}
        ))
        st.iteration = 1
        return st

    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    
    console.print("[yellow]Running Single-Agent Baseline...[/yellow]")
    state, metrics = run_benchmark("Single-Agent Baseline", query, baseline_runner)
    
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Answer"))
    console.print(render_markdown_report([metrics]))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
