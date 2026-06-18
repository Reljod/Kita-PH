import os
import json
import asyncio
import logging
from typing import List, Optional, Dict, Any, Protocol
from neo4j import GraphDatabase, AsyncGraphDatabase
from datetime import datetime
import logfire

logger = logging.getLogger(__name__)

from app.models.graph_rag import (
    GraphNode, GraphRelationship, GraphDocument, 
    GraphChunk, GraphEntity, GraphConcept, GraphRagSearchResult
)

class GraphRagService(Protocol):
    async def ingest_document(self, doc: GraphDocument, chunks: List[GraphChunk]) -> bool:
        ...
    
    async def add_entities_and_relationships(
        self, 
        entities: List[GraphEntity], 
        relationships: List[GraphRelationship],
        chunk_ids: Optional[List[str]] = None
    ) -> bool:
        ...

    async def query(
        self, 
        query_text: str, 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[GraphRagSearchResult]:
        ...

    async def upsert_node(self, label: str, properties: Dict[str, Any]) -> str:
        ...

    async def upsert_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        properties: Dict[str, Any]
    ) -> bool:
        ...

class Neo4JGraphRagService:
    def __init__(self, uri: str, user: str, password: str, org_id: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.org_id = org_id
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._model = None

    def _prepare_props(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Serializes complex types for Neo4j storage."""
        cleaned = {}
        for k, v in props.items():
            if isinstance(v, (dict, list)) and not all(isinstance(x, (int, float, str, bool)) for x in (v if isinstance(v, list) else [])):
                cleaned[k] = json.dumps(v)
            else:
                cleaned[k] = v
        return cleaned

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    async def close(self):
        await self.driver.close()

    async def initialize_schema(self):
        """Sets up constraints and indexes."""
        async with self.driver.session() as session:
            # Constraints for unique IDs per organization
            constraints = [
                "CREATE CONSTRAINT document_id_org IF NOT EXISTS FOR (d:Document) REQUIRE (d.id, d.org_id) IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_org IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.id, c.org_id) IS UNIQUE",
                "CREATE CONSTRAINT entity_id_org IF NOT EXISTS FOR (e:Entity) REQUIRE (e.id, e.org_id) IS UNIQUE",
                "CREATE CONSTRAINT concept_id_org IF NOT EXISTS FOR (co:Concept) REQUIRE (co.id, co.org_id) IS UNIQUE",
            ]
            for c in constraints:
                await session.run(c)
            
            # Indexes
            indexes = [
                "CREATE INDEX entity_name_org IF NOT EXISTS FOR (e:Entity) ON (e.name, e.org_id)",
                "CREATE INDEX chunk_org_id IF NOT EXISTS FOR (c:Chunk) ON (c.org_id)",
                "CREATE INDEX chunk_agent_id IF NOT EXISTS FOR (c:Chunk) ON (c.agent_id)",
                "CREATE INDEX chunk_filename IF NOT EXISTS FOR (c:Chunk) ON (c.filename)",
            ]
            for i in indexes:
                await session.run(i)

            # Vector Index (Requires Neo4j 5.11+)
            # Note: dimension should match the embedding model (e.g., 384 for all-MiniLM-L6-v2)
            # Defaulting to 384 but this can be parameterized
            try:
                await session.run("""
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: 384,
                      `vector.similarity_function`: 'cosine'
                    }}
                """)
            except Exception as e:
                logfire.error(f"Failed to create vector index: {e}")

    async def ingest_document(self, doc: GraphDocument, chunks: List[GraphChunk]) -> bool:
        logger.info(
            f"Ingesting document in Neo4j GraphRAG: doc_id={doc.id}, title={doc.title}, chunks={len(chunks)}",
            extra={"doc_id": doc.id, "title": doc.title, "chunks_count": len(chunks)}
        )
        async with self.driver.session() as session:
            # Create Document Node
            await session.run("""
                MERGE (d:Document {id: $doc_id, org_id: $org_id})
                SET d.title = $title, d.metadata = $metadata, d.updated_at = datetime()
            """, doc_id=doc.id, org_id=self.org_id, title=doc.title, metadata=json.dumps(doc.metadata))

            # Create Chunks and link to Document
            for chunk in chunks:
                # Extract agent_id and filename from metadata for indexing/filtering
                agent_id = chunk.metadata.get("agent_id")
                filename = chunk.metadata.get("filename")
                
                # Calculate embedding if missing
                if not chunk.embedding:
                    loop = asyncio.get_event_loop()
                    emb = await loop.run_in_executor(None, self.model.encode, chunk.content)
                    chunk.embedding = emb.tolist()
                
                await session.run("""
                    MERGE (c:Chunk {id: $chunk_id, org_id: $org_id})
                    SET c.content = $content, 
                        c.embedding = $embedding, 
                        c.metadata = $metadata,
                        c.agent_id = $agent_id,
                        c.filename = $filename,
                        c.updated_at = datetime()
                    WITH c
                    MATCH (d:Document {id: $doc_id, org_id: $org_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, 
                chunk_id=chunk.id, 
                org_id=self.org_id, 
                content=chunk.content, 
                embedding=chunk.embedding, 
                metadata=json.dumps(chunk.metadata),
                agent_id=agent_id,
                filename=filename,
                doc_id=doc.id)
            
            return True

    async def add_entities_and_relationships(
        self, 
        entities: List[GraphEntity], 
        relationships: List[GraphRelationship],
        chunk_ids: Optional[List[str]] = None
    ) -> bool:
        async with self.driver.session() as session:
            # Create Entities
            for ent in entities:
                await session.run("""
                    MERGE (e:Entity {name: $name, org_id: $org_id})
                    SET e.id = CASE WHEN e.id IS NULL THEN $entity_id ELSE e.id END,
                        e.type = $type, 
                        e.description = $description, 
                        e.properties = $properties,
                        e.updated_at = datetime()
                """, 
                entity_id=ent.id, 
                org_id=self.org_id, 
                name=ent.name, 
                type=ent.type, 
                description=ent.description,
                properties=json.dumps(ent.properties))

                # Link to chunks if provided
                if chunk_ids:
                    for cid in chunk_ids:
                        await session.run("""
                            MATCH (e:Entity {id: $entity_id, org_id: $org_id})
                            MATCH (c:Chunk {id: $chunk_id, org_id: $org_id})
                            MERGE (c)-[:MENTIONS]->(e)
                        """, entity_id=ent.id, chunk_id=cid, org_id=self.org_id)

            # Create Relationships
            for rel in relationships:
                # Use name-based lookup for source and target if possible, or ID
                # Since we MERGE entities by name, we can reliably match them by name here
                # However, the rel object passed currently has IDs. 
                # We need to make sure we are matching on properties that unique identify the node.
                # Given we MERGE on name, matching by ID (which we also SET) is safe but name is better for cross-window
                await session.run("""
                    MATCH (s:Entity {id: $source_id, org_id: $org_id})
                    MATCH (t:Entity {id: $target_id, org_id: $org_id})
                    MERGE (s)-[r:RELATED_TO {org_id: $org_id}]->(t)
                    SET r.type = $rel_type, r += $properties, r.updated_at = datetime()
                """, 
                source_id=rel.source_id, 
                target_id=rel.target_id, 
                rel_type=rel.rel_type, 
                org_id=self.org_id, 
                properties=self._prepare_props(rel.properties))

            return True

    async def query(
        self, 
        query_text: str, 
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[GraphRagSearchResult]:
        """
        Hybrid search: 
        1. Find relevant chunks via vector search.
        2. Traverse graph from chunks to entities and neighbors.
        3. Collect context and return.
        """
        import time
        start_time = time.perf_counter()
        truncated_query = query_text[:150] + "..." if len(query_text) > 150 else query_text

        # Calculate embedding
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, self.model.encode, query_text)
        query_embedding = query_embedding.tolist()

        results = []
        async with self.driver.session() as session:
            # Dynamically build WHERE clause for filters
            filter_clauses = ["chunk.org_id = $org_id"]
            params = {
                "embedding": query_embedding,
                "limit": limit,
                "org_id": self.org_id
            }

            if filters:
                for idx, (key, value) in enumerate(filters.items()):
                    if value is not None:
                        # Sanitize key name for Cypher (simple check)
                        if not key.isidentifier():
                            continue
                        param_name = f"filter_{idx}"
                        
                        # Special handling for agent_id scoping:
                        # Allow access to agent-specific memory OR org-wide memory (NULL)
                        if key == "agent_id":
                            filter_clauses.append(f"(chunk.{key} = ${param_name} OR chunk.{key} IS NULL)")
                        else:
                            filter_clauses.append(f"chunk.{key} = ${param_name}")
                            
                        params[param_name] = value

            where_clause = " WHERE " + " AND ".join(filter_clauses)

            # Step 1 & 2: Vector search + graph traversal
            cypher = f"""
            CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
            YIELD node AS chunk, score
            {where_clause}
            
            // Get mentioned entities
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
            
            // Get related entities (1-hop)
            OPTIONAL MATCH (entity)-[r:RELATED_TO]-(neighbor:Entity)
            
            RETURN 
                chunk.content AS content,
                score,
                chunk.metadata AS metadata,
                collect(DISTINCT {{id: entity.id, label: 'Entity', properties: entity {{.*}}}}) AS entities,
                collect(DISTINCT {{id: neighbor.id, label: 'Entity', properties: neighbor {{.*}}}}) AS neighbors
            """
            
            res = await session.run(cypher, **params)
            async for record in res:
                # Combine nodes for the result
                nodes = []
                for n in record["entities"] + record["neighbors"]:
                    if n.get("id"):
                        # Deserialize properties if they are JSON strings
                        raw_props = n["properties"] or {}
                        props = {}
                        for pk, pv in raw_props.items():
                            # Handle Neo4j specific types that aren't JSON serializable by default
                            if hasattr(pv, 'to_native'):
                                props[pk] = pv.to_native()
                            elif isinstance(pv, str) and (pv.startswith("{") or pv.startswith("[")):
                                try:
                                    props[pk] = json.loads(pv)
                                except:
                                    props[pk] = pv
                            else:
                                props[pk] = pv
                        
                        nodes.append(GraphNode(id=n["id"], label=n["label"], properties=props))
                
                # Deserialize metadata
                raw_metadata = record["metadata"]
                metadata = {}
                if isinstance(raw_metadata, str):
                    try:
                        metadata = json.loads(raw_metadata)
                    except:
                        metadata = {"raw": raw_metadata}
                elif isinstance(raw_metadata, dict):
                    metadata = raw_metadata

                results.append(GraphRagSearchResult(
                    content=record["content"],
                    score=record["score"],
                    metadata=metadata,
                    nodes=nodes
                ))
        
        duration = time.perf_counter() - start_time
        logger.info(
            f"Neo4j GraphRAG query completed: query={truncated_query}, results={len(results)}, duration={duration:.3f}s",
            extra={
                "query": truncated_query,
                "limit": limit,
                "org_id": self.org_id,
                "results_count": len(results),
                "duration": duration
            }
        )
        return results

    async def upsert_node(self, label: str, properties: Dict[str, Any]) -> str:
        if "id" not in properties:
            raise ValueError("Node properties must include 'id'")
        
        async with self.driver.session() as session:
            # Note: Using string formatting for labels as they can't be parameterized in Cypher MERGE
            # We trust the label source here but should normally validate against a whitelist
            query = f"MERGE (n:{label} {{id: $id, org_id: $org_id}}) SET n += $props RETURN n.id"
            res = await session.run(query, id=properties["id"], org_id=self.org_id, props=self._prepare_props(properties))
            record = await res.single()
            return record[0]

    async def upsert_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        properties: Dict[str, Any]
    ) -> bool:
        async with self.driver.session() as session:
            query = f"""
                MATCH (s {{id: $source_id, org_id: $org_id}})
                MATCH (t {{id: $target_id, org_id: $org_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $props
                RETURN count(r)
            """
            res = await session.run(query, source_id=source_id, target_id=target_id, org_id=self.org_id, props=self._prepare_props(properties))
            record = await res.single()
            return record[0] > 0
