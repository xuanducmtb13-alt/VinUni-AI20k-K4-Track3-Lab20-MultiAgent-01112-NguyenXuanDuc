"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self):
        from multi_agent_research_lab.core.config import get_settings
        self.api_key = get_settings().tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if not self.api_key:
            return [
                SourceDocument(
                    title=f"[Mock] Search result for '{query}'",
                    url="https://example.com/mock",
                    content="This is a mocked search result because TAVILY_API_KEY is not set. In a real scenario, this would contain information about the query.",
                    snippet="This is a mocked search result..."
                )
            ]
            
        import urllib.request
        import urllib.parse
        import json
        
        url = "https://api.tavily.com/search"
        data = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                docs = []
                for res in result.get('results', []):
                    docs.append(SourceDocument(
                        title=res.get('title', ''),
                        url=res.get('url', ''),
                        content=res.get('content', ''),
                        snippet=res.get('content', '')[:200]
                    ))
                return docs
        except Exception as e:
            return [
                SourceDocument(
                    title=f"Error searching for '{query}'",
                    url="https://example.com/error",
                    content=str(e),
                    snippet=f"Failed with error: {str(e)}"
                )
            ]
