from pydantic_ai import Agent

class CreatorAgent(Agent):
    def __init__(self, model, instructions, toolsets):
        super().__init__(
            model=model,
            instructions=instructions,
            toolsets=toolsets
        )
