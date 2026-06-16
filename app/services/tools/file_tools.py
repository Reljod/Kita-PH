from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated, Optional
from app.utils.logger import log_tool_call

file_toolset = FunctionToolset()

@file_toolset.tool
@log_tool_call
async def resolve_file_id(
    ctx: RunContext[dict],
    file_path: Annotated[str, Field(description="The file path in format {id}.{extension}")]
) -> str:
    """
    Parses a file path to extract the unique file ID and verifies its existence.
    """
    from app.services.file_service import FileService
    
    file_service: FileService = ctx.deps.get("file_service")
    if not file_service:
        return "Error: File service not found in dependencies."
    
    # Extract ID from path (id.extension)
    file_id = file_path.split(".")[0]
    
    try:
        file_res = await file_service.get_file(file_id)
        if not file_res:
            return f"Error: File with ID '{file_id}' not found."
        
        return file_id
    except Exception as e:
        return f"Error resolving file ID: {str(e)}"
