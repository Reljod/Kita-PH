from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphDocument(BaseModel):
    id: str
    title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GraphChunk(BaseModel):
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphEntity(BaseModel):
    id: str
    name: str
    type: str # Person, Org, Location, etc.
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphConcept(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

class GraphRagSearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
