def get_full_instructions(custom_prompt: str = None) -> str:
    base_instructions = custom_prompt or "You are a helpful and concise AI assistant."
    return (
        f"{base_instructions}\n\n"
        "You have access to a memory toolset (RAG). IMPORTANT: Always use the `search_memory` tool to search for relevant information "
        "in your knowledge base when you need facts, context, or specific details that you don't already have."
    )
