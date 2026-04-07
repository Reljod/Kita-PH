from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated, Dict, Any, List, Optional

parse_toolset = FunctionToolset()

@parse_toolset.tool
async def fetch_latest_parse(
    ctx: RunContext[dict],
    file_id: Annotated[str, Field(description="The unique file ID")]
) -> Dict[str, Any]:
    """
    Retrieves the most recent parse result for a given file.
    Returns the full parse result, including markdown, text, and page-by-page items.
    """
    from app.services.parse_service import IParseService
    
    parse_service: IParseService = ctx.deps.get("parse_service")
    if not parse_service:
        return {"error": "Parse service not found in dependencies."}
    
    try:
        parse_record = await parse_service.get_latest_parse(file_id)
        if not parse_record:
            return {"error": f"No parse records found for file {file_id}."}
        
        return parse_record.result
    except Exception as e:
        return {"error": f"Error fetching latest parse: {str(e)}"}
