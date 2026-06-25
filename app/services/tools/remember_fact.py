from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated
from app.utils.logger import log_tool_call

context_toolset = FunctionToolset()


@context_toolset.tool
@log_tool_call
async def remember_fact(
    ctx: RunContext[dict],
    key: Annotated[str, Field(description="Short descriptive key for the fact, e.g. user_name, preferred_language")],
    value: Annotated[str, Field(description="The fact content to remember")],
) -> str:
    """Remember an important fact about the user, their preferences, or conversation context.
    Use this to persist key information that should be recalled in future conversations.
    """
    chat_id = ctx.deps.get("chat_id")
    if not chat_id:
        return "Error: chat_id not found"
    from app.dependencies.services import get_services
    services = get_services(ctx.deps.get("org_id"))
    await services.chat_context_service.store_fact(chat_id, key, value)
    return f"Fact remembered: {key} = {value}"
