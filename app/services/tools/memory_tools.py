from pydantic_ai import FunctionToolset
from pydantic import Field
from typing import Annotated, List, Optional, Dict, Any
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
    # Update status key if present in deps
    status_key = ctx.deps.get("status_key") if ctx.deps else None
    if status_key:
        try:
            from app.dependencies.services import get_services
            services = get_services(ctx.deps.get("org_id"))
            await services.agent_status_service.update_step(status_key, "retrieve_facts_vector", ctx.deps.get("agent_id"))
        except Exception:
            pass

    service = ctx.deps.get("rag_service")
    if not service:
        # Fallback to manual instantiation
        org_id = ctx.deps["org_id"]
        collection = TenantCollection(db.get_rag_collection(), org_id)
        service = MongoVectorDbRagService(collection)

    agent_id = ctx.deps.get("agent_id")
    results = await service.search(query, limit=limit, agent_id=agent_id)
    
    if not results:
        return "No relevant information found in memory."
    
    formatted_results = []
    for i, res in enumerate(results, 1):
        if res.question and res.answer:
            formatted_results.append(f"{i}. Question: {res.question}\nAnswer: {res.answer}")
        else:
            formatted_results.append(f"{i}. Title: {res.title}\nContent: {res.content}")
    
    return "\n\n".join(formatted_results)

@memory_toolset.tool
async def rag_search(
    ctx: RunContext[dict],
    query: Annotated[str, Field(description="The search query to find relevant information in the memory using hybrid vector and keyword text search.")],
    limit: Annotated[int, Field(description="The maximum number of results to return.")] = 5
) -> str:
    """
    Performs a hybrid search combining vector search and full text search, followed by reranking.
    Use this to retrieve factual context from uploaded documents to answer queries.
    """
    # Update status key if present in deps
    status_key = ctx.deps.get("status_key") if ctx.deps else None
    if status_key:
        try:
            from app.dependencies.services import get_services
            services = get_services(ctx.deps.get("org_id"))
            await services.agent_status_service.update_step(status_key, "retrieve_facts_vector", ctx.deps.get("agent_id"))
        except Exception:
            pass

    from app.services.rag.retrieval_service import IRetrievalService
    
    retrieval_service: IRetrievalService = ctx.deps.get("retrieval_service")
    if not retrieval_service:
        # Fallback to standard vector RAG search if retrieval service is not available
        return await search_memory(ctx, query, limit)
    
    results = await retrieval_service.search(query, limit=limit)
    
    if not results:
        return "No relevant information found in memory."
    
    formatted_results = []
    for i, res in enumerate(results, 1):
        formatted_results.append(f"Result {i} (Title: {res.title}):\n{res.content}")
        
    return "\n\n---\n\n".join(formatted_results)
