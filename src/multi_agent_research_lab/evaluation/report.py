"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""
    
    lines = [
        "# Benchmark Report",
        "",
        "This report compares the performance of different agent architectures.",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f} | {cost} | {quality}/10 "
            f"| {citation} | {failure} | {item.notes} |"
        )
        
    lines.extend([
        "",
        "## Analysis",
        "- **Quality**: Evaluated by LLM based on correctness and detail.",
        "- **Citation Coverage**: % of retrieved sources cited in the final answer.",
        "- **Failure Rate**: 100% if the run exceeded max iterations or threw unhandled errors."
    ])
        
    return "\n".join(lines) + "\n"
