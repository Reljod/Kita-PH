import os
import json
import httpx
import logfire
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class AdaptiveRagService:
    def __init__(self, org_id: str, retrieval_service, web_search_service):
        self.org_id = org_id
        self.retrieval_service = retrieval_service
        self.web_search_service = web_search_service
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = os.getenv("ROUTER_LLM_MODEL", "openai/gpt-4o-mini")
        
        # Instantiate AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL
        )

    def _get_deps(self, agent_id: Optional[str]) -> Dict[str, Any]:
        from app.dependencies.services import get_services
        services = get_services(self.org_id)

        return {
            "org_id": self.org_id, 
            "agent_id": agent_id, 
            "agent_service": services.agent_service,
            "file_service": services.file_service,
            "parse_service": services.parse_service,
            "graph_rag_service": services.graph_rag_service,
            "rag_service": services.rag_service,
            "retrieval_service": services.retrieval_service
        }

    async def _call_cheap_llm(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """Helper to invoke the cheap router/grader LLM on OpenRouter."""
        if not self.api_key:
            logfire.warning("OPENROUTER_API_KEY not set. Falling back to default heuristics.")
            raise ValueError("OPENROUTER_API_KEY not configured.")

        response_format = {"type": "json_object"} if json_mode else None
        try:
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_format,
                temperature=0.0,
                max_tokens=150 if json_mode else 50
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logfire.error("Adaptive RAG cheap LLM call failed: {error}", error=str(e))
            raise e

    async def route_query(self, query: str) -> str:
        """Classifies the query into 'vector', 'web', or 'none'."""
        system_prompt = (
            "You are an expert query router. Your job is to analyze the user's intent and query and decide the best retrieval strategy.\n"
            "Output one of the following exact string values: 'vector', 'web', or 'none'.\n"
            "- Choose 'vector' if the query is a single-hop query requiring semantic similarity, best for finding specific facts or paragraphs within isolated, unstructured internal company text/documents (e.g. employee handbook, project specs).\n"
            "- Choose 'web' if the query is temporally sensitive, real-time, or highly transient data that internal files cannot possibly have (e.g. current events, major recent tech news, external tutorials).\n"
            "- Choose 'none' if the question relies on universal parametric knowledge or conversational pleasantries (e.g. 'What is TCP?', 'Hello, how are you?')."
        )
        user_prompt = f"User Query: {query}\nOutput (just the word 'vector', 'web', or 'none'):"
        try:
            res = await self._call_cheap_llm(system_prompt, user_prompt)
            cleaned = res.lower().strip().replace('"', '').replace("'", "")
            if cleaned in ["vector", "web", "none"]:
                return cleaned
            # Fallback regex
            for choice in ["vector", "web", "none"]:
                if choice in cleaned:
                    return choice
            return "vector"
        except Exception:
            return "vector" # Default fallback

    async def grade_relevance(self, query: str, doc_content: str) -> bool:
        """Returns True if the document content is relevant to the query, False otherwise."""
        system_prompt = (
            "You are a strict relevance grader assessing whether a retrieved document is relevant to a user query.\n"
            "Analyze if the retrieved document contains keywords or semantic meaning relevant to the user's query.\n"
            "Output YES or NO. Do not output anything else."
        )
        user_prompt = f"User Query: {query}\nRetrieved Document: {doc_content}\nOutput (YES/NO):"
        try:
            res = await self._call_cheap_llm(system_prompt, user_prompt)
            return "yes" in res.lower()
        except Exception:
            return True # Fallback: keep document on error

    async def grade_groundedness(self, facts: List[str], answer: str) -> bool:
        """Returns True if the answer is strictly grounded in the provided facts, False otherwise."""
        if not facts:
            return True # If no facts were retrieved, groundedness is vacuously true/none
        system_prompt = (
            "You are a strict logic grader assessing whether a generated answer is grounded in and supported by a set of facts.\n"
            "Analyze if the generated answer is strictly supported by the provided facts.\n"
            "Output YES or NO. Do not output anything else."
        )
        facts_block = "\n---\n".join(facts)
        user_prompt = f"Facts:\n{facts_block}\n\nGenerated Answer:\n{answer}\n\nOutput (YES/NO):"
        try:
            res = await self._call_cheap_llm(system_prompt, user_prompt)
            return "yes" in res.lower()
        except Exception:
            return True # Fallback

    async def grade_completeness(self, query: str, answer: str) -> bool:
        """Returns True if the answer is complete, False otherwise."""
        system_prompt = (
            "You are a strict logic grader assessing whether a generated answer fully resolves a user's question.\n"
            "You are evaluating based on COMPLETENESS.\n"
            "* If the answer only addresses part of the user's question, it is incomplete.\n"
            "* If the answer says 'The provided documents do not contain this information,' it is incomplete.\n"
            "* If the answer fully resolves the core intent of the prompt, it is complete.\n"
            "Respond ONLY with a valid JSON object: {\"is_complete\": true} or {\"is_complete\": false}. Do not include markdown code block formatting or any other text."
        )
        user_prompt = f"User Question: {query}\nDraft Answer: {answer}\nOutput JSON:"
        try:
            res = await self._call_cheap_llm(system_prompt, user_prompt, json_mode=True)
            # Parse JSON
            cleaned = res.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned.strip())
            return bool(data.get("is_complete", False))
        except Exception as e:
            logfire.warning("Completeness grading failed, assuming complete: {error}", error=str(e))
            return True # Fallback

    async def rewrite_query(self, query: str) -> str:
        """Optimizes the query for search engines or vector databases."""
        system_prompt = (
            "You are a query rewriter. Your job is to analyze the original query and rewrite it into an optimized search query for better document retrieval.\n"
            "Output ONLY the optimized query. Do not include quotes or conversational prefix."
        )
        user_prompt = f"Original Query: {query}\nOptimized Search Query:"
        try:
            res = await self._call_cheap_llm(system_prompt, user_prompt)
            return res.strip().replace('"', '').replace("'", "")
        except Exception:
            return query # Fallback to original query

    async def retrieve_facts(self, strategy: str, query: str) -> List[str]:
        """Runs the search for the specified strategy and filters for relevance."""
        facts = []
        if strategy == "vector":
            logfire.info("Adaptive RAG: retrieving from Vector/Hybrid search using query={query!r}", query=query)
            try:
                results = await self.retrieval_service.search(query)
                for res in results:
                    facts.append(res.content)
            except Exception as e:
                logfire.error("Adaptive RAG retrieval search error: {error}", error=str(e))
        elif strategy == "web":
            logfire.info("Adaptive RAG: retrieving from Web search using query={query!r}", query=query)
            try:
                results = await self.web_search_service.search(query)
                if results and "organic" in results:
                    for res in results["organic"][:8]:
                        title = res.get("title", "No Title")
                        link = res.get("link", "No Link")
                        snippet = res.get("snippet", "No Snippet")
                        facts.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
            except Exception as e:
                logfire.error("Adaptive RAG web search error: {error}", error=str(e))
        
        # Grade relevance and filter documents
        relevant_facts = []
        for doc_content in facts:
            is_relevant = await self.grade_relevance(query, doc_content)
            if is_relevant:
                relevant_facts.append(doc_content)
        
        logfire.info("Adaptive RAG: retrieved {total} documents, {relevant} are relevant.", total=len(facts), relevant=len(relevant_facts))
        return relevant_facts

    async def run_agentic_flow(self, query: str, agent, message_history, agent_id: Optional[str] = None) -> Any:
        """Executes the complete Adaptive RAG + Self-RAG loop."""
        # 1. Route query
        strategy = await self.route_query(query)
        logfire.info("Adaptive RAG: query={query!r} routed to strategy={strategy!r}", query=query, strategy=strategy)
        
        visited_strategies = {strategy}
        current_query = query
        relevant_facts = []
        
        if strategy != "none":
            relevant_facts = await self.retrieve_facts(strategy, current_query)
            
            # If no relevant facts found, failover to alternative strategy
            if not relevant_facts:
                alt_strategy = "web" if strategy == "vector" else "vector"
                if alt_strategy not in visited_strategies:
                    logfire.info("Adaptive RAG: No relevant facts found. Failing over to {alt_strategy}", alt_strategy=alt_strategy)
                    visited_strategies.add(alt_strategy)
                    strategy = alt_strategy
                    current_query = await self.rewrite_query(query)
                    relevant_facts = await self.retrieve_facts(strategy, current_query)

        # Loop for Generation & Evaluation (Self-RAG)
        max_attempts = 3
        result = None
        
        deps = self._get_deps(agent_id)
        
        for attempt in range(1, max_attempts + 1):
            logfire.info("Adaptive RAG Self-Evaluation: Attempt {attempt} of {max_attempts}", attempt=attempt, max_attempts=max_attempts)
            
            # Prepare generator prompt
            user_prompt = query
            if relevant_facts:
                facts_block = "\n\n".join([f"Fact {i}:\n{fact}" for i, fact in enumerate(relevant_facts, 1)])
                user_prompt = (
                    f"{query}\n\n"
                    f"[RELEVANT RETRIEVED CONTEXT]\n"
                    f"{facts_block}\n\n"
                    f"INSTRUCTION: Draft your response using ONLY the relevant retrieved facts above when applicable. "
                    f"Do not hallucinate or state details not supported by the facts."
                )
            
            # Execute agent (Generator)
            logfire.info("Adaptive RAG: Invoking generator agent.")
            result = await agent.run(user_prompt, message_history=message_history, deps=deps)
            draft_response = result.data if hasattr(result, "data") else getattr(result, "content", str(result))
            
            # If no retrieval happened, we are done
            if not relevant_facts:
                logfire.info("Adaptive RAG: No retrieval strategy was used. Returning agent response directly.")
                return result
                
            # Grade groundedness
            is_grounded = await self.grade_groundedness(relevant_facts, draft_response)
            if not is_grounded:
                logfire.warning("Adaptive RAG: Draft answer failed groundedness (hallucination check). Retrying.")
                # Rewrite query and search again to find better facts
                current_query = await self.rewrite_query(current_query)
                relevant_facts = await self.retrieve_facts(strategy, current_query)
                continue
                
            # Grade completeness
            is_complete = await self.grade_completeness(query, draft_response)
            if is_complete:
                logfire.info("Adaptive RAG: Answer is complete and grounded! Returning result.")
                return result
            else:
                logfire.warning("Adaptive RAG: Draft answer failed completeness check.")
                # Failover to the next strategy if not already visited
                alt_strategy = "web" if strategy == "vector" else "vector"
                if alt_strategy not in visited_strategies:
                    logfire.info("Adaptive RAG: Failing over from {strategy} to {alt_strategy}", strategy=strategy, alt_strategy=alt_strategy)
                    visited_strategies.add(alt_strategy)
                    strategy = alt_strategy
                    current_query = await self.rewrite_query(current_query)
                    relevant_facts = await self.retrieve_facts(strategy, current_query)
                else:
                    # If we already tried all strategies, try rewriting the query and searching again
                    logfire.info("Adaptive RAG: Both search strategies already tried. Rewriting query and retrying current strategy.")
                    current_query = await self.rewrite_query(current_query)
                    relevant_facts = await self.retrieve_facts(strategy, current_query)

        # If we exhausted max attempts, return the last draft
        logfire.warning("Adaptive RAG: Exhausted max attempts. Returning last generated result.")
        return result
