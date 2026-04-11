import json
from typing import Dict, Any, List, Annotated
from pydantic import BaseModel, Field

class PropertyInput(BaseModel):
    key: Annotated[str, Field(description="The name of the metadata field")]
    value: Annotated[str, Field(description="The value of the metadata field")]

class EntityInput(BaseModel):
    name: Annotated[str, Field(description="The name of the entity")]
    type: Annotated[str, Field(description="The type of entity (e.g., Person, Organization, Location, Concept)")]
    description: Annotated[str, Field(description="A brief description of what this entity is or does")]
    properties: Annotated[List[PropertyInput], Field(default_factory=list, description="Additional metadata for the entity")]

print("--- Updated EntityInput Schema ---")
print(json.dumps(EntityInput.model_json_schema(), indent=2))
