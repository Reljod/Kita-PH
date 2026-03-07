from pydantic_ai import FunctionToolset
from pydantic import Field
from typing import Annotated, List
from app.services.rag_service import MongoVectorDbRagService
from pydantic_ai import RunContext
from app.db import db, TenantCollection

memory_toolset = FunctionToolset()

@memory_toolset.tool
async def search_memory(
    ctx: RunContext[dict],
    query: Annotated[str, Field(description="The search query to find relevant information in the memory.")],
    limit: Annotated[int, Field(description="The maximum number of results to return.")] = 5
) -> str:
    """
    Searches the agent's memory (RAG) for relevant information based on the query.
    This uses MongoDB embeddings for vector search.
    """
    org_id = ctx.deps["org_id"]
    agent_id = ctx.deps.get("agent_id")
    collection = TenantCollection(db.get_rag_collection(), org_id)
    service = MongoVectorDbRagService(collection, agent_id=agent_id)
    results = await service.search(query, limit=limit)
    
    if not results:
        return "No relevant information found in memory."
    
    formatted_results = []
    for i, res in enumerate(results, 1):
        formatted_results.append(f"{i}. Title: {res.title}\nContent: {res.content}")
    
    return "\n\n".join(formatted_results)
