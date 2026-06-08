import uuid
import logfire
from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field, BaseModel
from typing import Annotated, Dict, Any, List, Optional
from datetime import datetime, timezone

from app.models.graph_rag import GraphDocument, GraphChunk, GraphEntity, GraphRelationship

graph_rag_toolset = FunctionToolset()

class ChunkInput(BaseModel):
    content: Annotated[str, Field(description="The main text content of the chunk")]
    heading: Annotated[str, Field(description="A descriptive heading for the chunk content")]
    question: Annotated[str, Field(description="A question that this specific chunk provides the answer for")]

class PropertyInput(BaseModel):
    key: Annotated[str, Field(description="The name of the metadata field")]
    value: Annotated[str, Field(description="The value of the metadata field")]

class EntityInput(BaseModel):
    name: Annotated[str, Field(description="The name of the entity")]
    type: Annotated[str, Field(description="The type of entity (e.g., Person, Organization, Location, Concept)")]
    description: Annotated[str, Field(description="A brief description of what this entity is or does")]
    properties: Annotated[List[PropertyInput], Field(default_factory=list, description="Additional metadata for the entity")]

class RelationshipInput(BaseModel):
    source: Annotated[str, Field(description="The name of the source entity")]
    target: Annotated[str, Field(description="The name of the target entity")]
    type: Annotated[str, Field(description="The relationship type (e.g., PLAYS_FOR, LOCATED_IN, WORKS_AT)")]
    properties: Annotated[List[PropertyInput], Field(default_factory=list, description="Additional metadata for the relationship")]

@graph_rag_toolset.tool
async def ingest_into_graph(
    ctx: RunContext[dict],
    file_id: Annotated[str, Field(description="The unique file ID")],
    filename: Annotated[str, Field(description="The original filename")],
    chunks: Annotated[List[ChunkInput], Field(description="List of chunks to ingest")],
    entities: Annotated[List[EntityInput], Field(description="List of entities to ingest")],
    relationships: Annotated[List[RelationshipInput], Field(description="List of relationships to ingest")]
) -> str:
    """
    Ingests a processed document into the Graph RAG system.
    This includes creating a document node, chunk nodes with 'question' metadata,
    and associated entities and relationships.
    """
    from app.services.graph_rag_service import GraphRagService
    from app.services.file_service import FileService
    
    graph_service: GraphRagService = ctx.deps.get("graph_rag_service")
    file_service: FileService = ctx.deps.get("file_service")
    
    if not graph_service:
        return "Error: Graph RAG service not found in dependencies."
    
    try:
        # 1. Determine Scope
        # If the file has an agent_id in its metadata, we use it for scoping.
        # This ensures RAG filtering works correctly for agent-specific vs org-wide files.
        agent_id_to_store = None
        if file_service:
            file_info = await file_service.get_file(file_id)
            if file_info:
                agent_id_to_store = file_info.agent_id

        # 2. Prepare Document
        doc = GraphDocument(
            id=file_id,
            title=filename,
            metadata={
                "source": "rag_manager_agent", 
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id_to_store,
                "filename": filename
            }
        )
        
        # 3. Prepare Chunks
        graph_chunks = []
        chunk_ids = []
        for c in chunks:
            cid = str(uuid.uuid4())
            chunk_ids.append(cid)
            graph_chunks.append(GraphChunk(
                id=cid,
                document_id=file_id,
                content=c.content,
                metadata={
                    "heading": c.heading,
                    "question": c.question,
                    "source_file": filename,
                    "agent_id": agent_id_to_store,
                    "filename": filename
                }
            ))
            
        await graph_service.ingest_document(doc, graph_chunks)
        
        # 4. Prepare Entities
        graph_entities = []
        entity_map = {} # Maps name to ID for relationship lookup
        for e in entities:
            # We still generate a UUID for the GraphEntity object, 
            # but the Service will MERGE on name.
            eid = str(uuid.uuid4())
            name = e.name
            entity_map[name] = eid
            # Convert List[PropertyInput] to dict
            props_dict = {p.key: p.value for p in e.properties}
            graph_entities.append(GraphEntity(
                id=eid,
                name=name,
                type=e.type,
                description=e.description,
                properties={**props_dict, "agent_id": agent_id_to_store}
            ))
            
        # 5. Prepare Relationships
        graph_relationships = []
        for r in relationships:
            src_name = r.source
            tgt_name = r.target
            
            src_id = entity_map.get(src_name)
            tgt_id = entity_map.get(tgt_name)
            
            if src_id and tgt_id:
                # Convert List[PropertyInput] to dict
                props_dict = {p.key: p.value for p in r.properties}
                graph_relationships.append(GraphRelationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=r.type,
                    properties={**props_dict, "agent_id": agent_id_to_store}
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
