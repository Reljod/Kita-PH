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
async def search_memory_v2(
    ctx: RunContext[dict],
    query: Annotated[str, Field(description="The search query to find relevant information in the memory using hybrid search.")],
    limit: Annotated[int, Field(description="The maximum number of results to return.")] = 5
) -> str:
    """
    Advanced memory search using Graph RAG. 
    Retrieves relevant text chunks and associated entities/relationships.
    Use this for complex queries that require connecting different pieces of information.
    """
    from app.services.graph_rag_service import GraphRagService
    
    graph_service: GraphRagService = ctx.deps.get("graph_rag_service")
    if not graph_service:
        # Fallback to standard search if graph service is not available
        return await search_memory(ctx, query, limit)
    
    agent_id = ctx.deps.get("agent_id")
    # Apply agent scoping if available
    filters = {"agent_id": agent_id} if agent_id else None
    
    results = await graph_service.query(query, limit=limit, filters=filters)
    
    if not results:
        return "No relevant information found in memory."
    
    formatted_results = []
    for i, res in enumerate(results, 1):
        chunk_text = f"RESULT {i} [Relevance: {res.score:.2f}]\n"
        chunk_text += f"CONTENT: {res.content}\n"
        
        # Source metadata
        filename = res.metadata.get("filename")
        if filename:
            chunk_text += f"SOURCE: {filename}\n"
            
        # Graph Context: Entities and Relationships
        if res.nodes:
            entities = []
            for n in res.nodes:
                if n.label == "Entity":
                    name = n.properties.get("name", "Unknown")
                    etype = n.properties.get("type", "Entity")
                    entities.append(f"{name} ({etype})")
            
            if entities:
                # Deduplicate and sort
                unique_entities = sorted(list(set(entities)))
                chunk_text += f"MENTIONS: {', '.join(unique_entities)}\n"
        
        formatted_results.append(chunk_text.strip())
    
    return "\n\n---\n\n".join(formatted_results)
