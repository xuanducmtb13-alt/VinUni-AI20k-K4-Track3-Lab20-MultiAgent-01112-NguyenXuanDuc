"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a metric object."""
    from multi_agent_research_lab.services.llm_client import LLMClient
    
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # 1. Total Token Cost
    total_cost = sum([res.metadata.get("cost_usd", 0.0) for res in state.agent_results if res.metadata.get("cost_usd") is not None])
    
    # 2. Failure Rate
    failure_rate = 1.0 if len(state.errors) > 0 else 0.0
    
    # 3. Citation Coverage
    citation_cov = 0.0
    if state.sources and state.final_answer:
        cited = 0
        for src in state.sources:
            if (src.url and src.url in state.final_answer) or (src.title and src.title in state.final_answer):
                cited += 1
        citation_cov = cited / len(state.sources)
        
    # 4. Quality Scoring
    quality_score = 0.0
    if state.final_answer:
        llm = LLMClient()
        prompt = (
            f"Query: {query}\nAnswer: {state.final_answer}\n\n"
            "Score this answer from 0 to 10 on accuracy and completeness. Output ONLY the integer."
        )
        try:
            res = llm.complete("You are an evaluator. Output a number only.", prompt)
            score_text = "".join(filter(str.isdigit, res.content))
            quality_score = min(max(float(score_text), 0.0), 10.0) if score_text else 0.0
        except Exception:
            quality_score = 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=f"Iter: {state.iteration}. Errors: {len(state.errors)}"
    )
    return state, metrics
