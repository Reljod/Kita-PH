import os
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from app.db import TenantCollection
from bson import ObjectId

class EnrichmentResult(BaseModel):
    question: str = Field(description="A highly specific question derived from the title and content. It must incorporate the key entities and context of the memory, and serve as a retrieval query.")
    answer: str = Field(description="A concise and accurate answer that answers the question, derived from the content.")

class RagEnrichmentService:
    def __init__(self, collection: TenantCollection):
        self.collection = collection
        self._model = None

    def _get_llm_model(self):
        if self._model is None:
            # Using deepseek/deepseek-v4-flash as requested by the user
            model_name = "deepseek/deepseek-v4-flash"
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            self._model = OpenRouterModel(
                model_name,
                provider=OpenRouterProvider(api_key=api_key)
            )
        return self._model

    async def enrich_and_embed(self, doc_id: str, sentence_transformer_model) -> dict:
        try:
            obj_id = ObjectId(doc_id)
        except Exception:
            raise ValueError("Invalid RAG ID")

        doc = self.collection.find_one({"_id": obj_id})
        if not doc:
            return None

        # Determine date representation
        created_at = doc.get("created_at") or datetime.utcnow()
        date_str = created_at.strftime("%Y-%m-%d")

        raw_content = doc.get("original_content") or doc.get("content")

        try:
            agent = Agent(
                model=self._get_llm_model(),
                output_type=EnrichmentResult,
                system_prompt=(
                    "You are an expert knowledge engineer. Your task is to convert a given title and content "
                    "into a highly specific question and its corresponding answer. "
                    "The question must be extremely specific to the content provided so that it is suitable "
                    "for vector database retrieval. The answer must answer the question using the content provided."
                )
            )
            
            prompt = f"Title: {doc['title']}\nContent: {raw_content}"
            result = await agent.run(prompt)
            enrichment = result.output
            question = enrichment.question
            answer = enrichment.answer
        except Exception as llm_err:
            print(f"LLM enrichment failed for doc {doc_id}, falling back to defaults: {llm_err}")
            # Fallback values
            question = f"What is the information about {doc['title']}?"
            answer = raw_content

        # Add date to the text we embed
        embedding_text = f"Date: {date_str}. Question: {question}"

        # Generate embedding vector
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, sentence_transformer_model.encode, embedding_text)

        update_data = {
            "question": question,
            "content": answer,  # The content field answers the question
            "answer": answer,
            "embedding": embedding.tolist(),
            "status": "completed",
            "updated_at": datetime.utcnow()
        }

        self.collection.update_one(
            {"_id": obj_id},
            {"$set": update_data}
        )
        return update_data
