import os
import logging
from bson import ObjectId
from typing import List, Optional, Dict, Any, Protocol
from openai import AsyncOpenAI
import logfire

logger = logging.getLogger(__name__)

from app.models.llm import LlmCreateRequest, LlmResponse, LlmDocument
from app.db import TenantCollection


class ILlmService(Protocol):
    def add_llm(self, req: LlmCreateRequest) -> LlmResponse:
        ...
    def list_llms(self) -> List[LlmResponse]:
        ...
    def get_llm(self, llm_id: str) -> Optional[LlmResponse]:
        ...
    def delete_llm(self, llm_id: str) -> bool:
        ...
    async def run(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        status_key: Optional[str] = None,
        step: Optional[str] = None,
        agent_id: str = "KitaAgent",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        ...


def format_llm_response(doc: dict) -> LlmResponse:
    return LlmResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        model=doc["model"],
        provider=doc["provider"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"]
    )


class LlmService(ILlmService):
    def __init__(self, collection: TenantCollection):
        self.collection = collection
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1"
        )

    def add_llm(self, req: LlmCreateRequest) -> LlmResponse:
        new_llm = LlmDocument(
            name=req.name,
            model=req.model,
            provider=req.provider
        )
        doc = new_llm.model_dump()
        res = self.collection.insert_one(doc)
        doc["_id"] = res.inserted_id
        return format_llm_response(doc)

    def list_llms(self) -> List[LlmResponse]:
        llms = self.collection.find().sort("created_at", -1)
        return [format_llm_response(l) for l in llms]

    def get_llm(self, llm_id: str) -> Optional[LlmResponse]:
        try:
            obj_id = ObjectId(llm_id)
        except Exception:
            raise ValueError("Invalid LLM ID")
            
        doc = self.collection.find_one({"_id": obj_id})
        if not doc:
            return None
        return format_llm_response(doc)

    def delete_llm(self, llm_id: str) -> bool:
        try:
            obj_id = ObjectId(llm_id)
        except Exception:
            raise ValueError("Invalid LLM ID")
            
        res = self.collection.delete_one({"_id": obj_id})
        return res.deleted_count > 0

    async def run(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        status_key: Optional[str] = None,
        step: Optional[str] = None,
        agent_id: str = "KitaAgent",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        if status_key and step:
            try:
                from app.dependencies.services import get_services
                services = get_services(self.collection.org_id)
                await services.agent_status_service.update_step(status_key, step, agent_id)
            except Exception as e:
                logfire.error("Failed to update status step in LLM run: {error}", error=str(e))

        import time
        start_time = time.perf_counter()
        prompt_preview = str(messages[-1].get("content") if messages else "")
        truncated_prompt = prompt_preview[:150] + "..." if len(prompt_preview) > 150 else prompt_preview

        response_format = {"type": "json_object"} if json_mode else None
        try:
            chat_completion = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens
            )
            duration = time.perf_counter() - start_time
            usage = getattr(chat_completion, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            logger.info(
                f"LLM call completed: model={model_name}, duration={duration:.3f}s, tokens={total_tokens}",
                extra={
                    "model": model_name,
                    "prompt": truncated_prompt,
                    "duration": duration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "temperature": temperature,
                }
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"LLM call failed: model={model_name}, duration={duration:.3f}s: {e}",
                extra={
                    "model": model_name,
                    "prompt": truncated_prompt,
                    "duration": duration,
                    "error": str(e)
                },
                exc_info=True
            )
            raise e

