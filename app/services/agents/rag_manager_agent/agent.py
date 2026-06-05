from pydantic_ai import Agent

class RagManagerAgent(Agent):
    def __init__(self, model, instructions, toolsets):
        super().__init__(
            model=model,
            instructions=instructions,
            retries=3,
            toolsets=toolsets
        )
