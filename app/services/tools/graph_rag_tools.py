import uuid
import logfire
from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated, Dict, Any, List, Optional
from datetime import datetime

from app.models.graph_rag import GraphDocument, GraphChunk, GraphEntity, GraphRelationship

graph_rag_toolset = FunctionToolset()

@graph_rag_toolset.tool
async def ingest_into_graph(
    ctx: RunContext[dict],
    file_id: Annotated[str, Field(description="The unique file ID")],
    filename: Annotated[str, Field(description="The original filename")],
    chunks: Annotated[List[Dict[str, Any]], Field(description="List of chunks with 'content', 'heading', and 'question'")],
    entities: Annotated[List[Dict[str, Any]], Field(description="List of entities with 'name', 'type', 'description'")],
    relationships: Annotated[List[Dict[str, Any]], Field(description="List of relationships with 'source', 'target', 'type'")]
) -> str:
    """
    Ingests a processed document into the Graph RAG system.
    This includes creating a document node, chunk nodes with 'question' metadata,
    and associated entities and relationships.
    """
    from app.services.graph_rag_service import GraphRagService
    
    graph_service: GraphRagService = ctx.deps.get("graph_rag_service")
    if not graph_service:
        return "Error: Graph RAG service not found in dependencies."
    
    try:
        # 1. Prepare Document
        doc = GraphDocument(
            id=file_id,
            title=filename,
            metadata={"source": "rag_manager_agent", "ingested_at": datetime.utcnow().isoformat()}
        )
        
        # 2. Prepare Chunks
        graph_chunks = []
        chunk_ids = []
        for c in chunks:
            cid = str(uuid.uuid4())
            chunk_ids.append(cid)
            graph_chunks.append(GraphChunk(
                id=cid,
                document_id=file_id,
                content=c.get("content", ""),
                metadata={
                    "heading": c.get("heading", ""),
                    "question": c.get("question", ""),
                    "source_file": filename
                }
            ))
            
        await graph_service.ingest_document(doc, graph_chunks)
        
        # 3. Prepare Entities
        graph_entities = []
        entity_map = {} # Maps name to ID for relationship lookup
        for e in entities:
            eid = str(uuid.uuid4())
            name = e.get("name", "")
            entity_map[name] = eid
            graph_entities.append(GraphEntity(
                id=eid,
                name=name,
                type=e.get("type", "General"),
                description=e.get("description", ""),
                properties=e.get("properties", {})
            ))
            
        # 4. Prepare Relationships
        graph_relationships = []
        for r in relationships:
            src_name = r.get("source")
            tgt_name = r.get("target")
            
            src_id = entity_map.get(src_name)
            tgt_id = entity_map.get(tgt_name)
            
            if src_id and tgt_id:
                graph_relationships.append(GraphRelationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=r.get("type", "RELATED_TO"),
                    properties=r.get("properties", {})
                ))
                
        await graph_service.add_entities_and_relationships(
            graph_entities, 
            graph_relationships, 
            chunk_ids=chunk_ids
        )
        
        return f"Successfully ingested {len(graph_chunks)} chunks, {len(graph_entities)} entities, and {len(graph_relationships)} relationships."
    except Exception as e:
        error_msg = str(e)
        logfire.error("Error during graph ingestion: {error}", error=error_msg)
        # Check if error contains sensitivity (common for Neo4j context errors)
        if "bolt://" in error_msg or "neo4j" in error_msg.lower():
            return "Error during ingestion: Authentication or connection failure. Please check Graph RAG configuration."
        return f"Error during ingestion: {error_msg}"
