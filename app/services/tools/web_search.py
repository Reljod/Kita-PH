from pydantic_ai import FunctionToolset
from pydantic import Field
from typing import Annotated, Optional
from pydantic_ai import RunContext
from app.services.tool_service import ToolService
from app.services.web_search_service import SerperSearchService

web_search_toolset = FunctionToolset()

@web_search_toolset.tool
async def web_search(
    ctx: RunContext[dict],
    query: Annotated[str, Field(description="The search query to find information on the web.")],
    country: Annotated[str, Field(description="Country code for the search (e.g., 'us', 'ph').")] = "us",
    language: Annotated[str, Field(description="Language code for the search (e.g., 'en').")] = "en",
    autocorrect: Annotated[bool, Field(description="Whether to autocorrect the search query.")] = True,
    page: Annotated[int, Field(description="The page number of the search results.")] = 1,
    search_type: Annotated[str, Field(description="The type of search (e.g., 'search', 'images', 'news').")] = "search",
    date_range: Annotated[Optional[str], Field(description="Additional search parameters like date range (e.g., 'past_hour', 'past_24_hours', or 'qdr:h').")] = None
) -> str:
    """
    Performs a web search to find relevant and up-to-date information.
    Use this tool when you need information from the internet that might not be in your training data.
    """
    web_search_service = SerperSearchService()
    service = ToolService(web_search_service)
    try:
        results = await service.web_search(
            query=query,
            country=country,
            language=language,
            autocorrect=autocorrect,
            page=page,
            search_type=search_type,
            date_range=date_range
        )
    except Exception as e:
        return f"Error performing web search: {str(e)}"
    
    if not results or "organic" not in results or not results["organic"]:
        return "No relevant information found on the web."
    
    formatted_results = []
    # Limit to top 8 results to keep the context size manageable
    for i, res in enumerate(results["organic"][:8], 1):
        title = res.get("title", "No Title")
        link = res.get("link", "No Link")
        snippet = res.get("snippet", "No Snippet")
        formatted_results.append(f"{i}. {title}\nLink: {link}\nSnippet: {snippet}")
    
    return "\n\n".join(formatted_results)
