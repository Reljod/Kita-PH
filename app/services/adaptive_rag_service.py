import os
import json
import logfire
from typing import List, Dict, Any, Optional

class AdaptiveRagService:
    def __init__(self, org_id: str, retrieval_service, web_search_service):
        self.org_id = org_id
        self.retrieval_service = retrieval_service
        self.web_search_service = web_search_service
        self.model_name = os.getenv("ROUTER_LLM_MODEL", "deepseek/deepseek-v4-flash")

    def _get_deps(self, agent_id: Optional[str], status_key: Optional[str] = None) -> Dict[str, Any]:
        from app.dependencies.services import get_services
        services = get_services(self.org_id)

        return {
            "org_id": self.org_id, 
            "agent_id": agent_id, 
            "status_key": status_key,
            "agent_service": services.agent_service,
            "file_service": services.file_service,
            "parse_service": services.parse_service,
            "graph_rag_service": services.graph_rag_service,
            "rag_service": services.rag_service,
            "retrieval_service": services.retrieval_service
        }

    async def _call_cheap_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        status_key: Optional[str] = None,
        step: Optional[str] = None,
        agent_id: str = "KitaAgent",
        json_mode: bool = False
    ) -> str:
        """Helper to invoke the cheap router/grader LLM via LlmService."""
        from app.dependencies.services import get_services
        services = get_services(self.org_id)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await services.llm_service.run(
            model_name=self.model_name,
            messages=messages,
            status_key=status_key,
            step=step,
            agent_id=agent_id,
            json_mode=json_mode,
            temperature=0.0,
            max_tokens=150 if json_mode else 50
        )

    def _format_chat_history(self, message_history: Optional[List[Any]], limit: int = 5) -> str:
        """Formats the last N turns of Pydantic AI messages into a plain text transcript."""
        if not message_history:
            return ""
        
        transcript = []
        for msg in message_history[-limit:]:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                parts = msg.get("parts", [])
                role_label = "User" if role in ["user", "ModelRequest"] else "Agent"
                for part in parts:
                    if isinstance(part, dict):
                        if "content" in part:
                            transcript.append(f"{role_label}: {part['content']}")
            else:
                role = msg.__class__.__name__
                role_label = "User" if role == "ModelRequest" else "Agent"
                parts = getattr(msg, "parts", [])
                for part in parts:
                    part_name = part.__class__.__name__
                    if part_name in ["UserPrompt", "TextPart"]:
                        content = getattr(part, "content", "")
                        if content:
                            transcript.append(f"{role_label}: {content}")
                            
        return "\n".join(transcript)

    async def condense_query(self, query: str, message_history: Optional[List[Any]], status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> str:
        """Rewrites the user's query to incorporate context from message_history."""
        if not message_history:
            return query
            
        history_str = self._format_chat_history(message_history)
        if not history_str:
            return query
            
        system_prompt = (
            "You are an expert query condenser. Your job is to analyze the conversation history "
            "and the latest user query, then output a standalone query that can be searched "
            "without needing to read the chat history. Resolve any pronouns and add necessary context from the conversation.\n"
            "Output ONLY the standalone rewritten query, no conversational padding or quotes."
        )
        user_prompt = (
            f"Rules for rewriting the query:\n"
            f"1. Check the Chat History below to understand the current context.\n"
            f"2. Rewrite the Latest Query to be a standalone search query.\n"
            f"3. Resolve all ambiguous pronouns (like 'it', 'they', 'that') and add missing context/topics from the Chat History.\n"
            f"4. Output ONLY the rewritten standalone query, with no quotes or introduction.\n\n"
            f"Chat History:\n{history_str}\n\n"
            f"Latest Query: {query}\n\n"
            f"Standalone Query:"
        )
        
        try:
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="condense_query", 
                agent_id=agent_id
            )
            condensed = res.strip().replace('"', '').replace("'", "")
            logfire.info("Adaptive RAG query condensation: raw={query!r} -> condensed={condensed!r}", query=query, condensed=condensed)
            return condensed
        except Exception as e:
            logfire.warning("Adaptive RAG query condensation failed: {error}", error=str(e))
            return query

    async def route_query(self, query: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> str:
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
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="route_query", 
                agent_id=agent_id
            )
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

    async def grade_relevance(self, query: str, doc_content: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> bool:
        """Returns True if the document content is relevant to the query, False otherwise."""
        system_prompt = (
            "You are a strict relevance grader assessing whether a retrieved document is relevant to a user query.\n"
            "Analyze if the retrieved document contains keywords or semantic meaning relevant to the user's query.\n"
            "Output YES or NO. Do not output anything else."
        )
        user_prompt = f"User Query: {query}\nRetrieved Document: {doc_content}\nOutput (YES/NO):"
        try:
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="grade_relevance", 
                agent_id=agent_id
            )
            return "yes" in res.lower()
        except Exception:
            return True # Fallback: keep document on error

    async def grade_groundedness(self, facts: List[str], answer: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> bool:
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
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="grade_groundedness", 
                agent_id=agent_id
            )
            return "yes" in res.lower()
        except Exception:
            return True # Fallback

    async def grade_completeness(self, query: str, answer: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> bool:
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
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="grade_completeness", 
                agent_id=agent_id,
                json_mode=True
            )
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

    async def rewrite_query(self, query: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> str:
        """Optimizes the query for search engines or vector databases."""
        system_prompt = (
            "You are a query rewriter. Your job is to analyze the original query and rewrite it into an optimized search query for better document retrieval.\n"
            "Output ONLY the optimized query. Do not include quotes or conversational prefix."
        )
        user_prompt = f"Original Query: {query}\nOptimized Search Query:"
        try:
            res = await self._call_cheap_llm(
                system_prompt, 
                user_prompt, 
                status_key=status_key, 
                step="rewrite_query", 
                agent_id=agent_id
            )
            return res.strip().replace('"', '').replace("'", "")
        except Exception:
            return query # Fallback to original query

    async def retrieve_facts(self, strategy: str, query: str, status_key: Optional[str] = None, agent_id: str = "KitaAgent") -> List[str]:
        """Runs the search for the specified strategy and filters for relevance."""
        # Update status before searching
        if status_key:
            try:
                from app.dependencies.services import get_services
                services = get_services(self.org_id)
                step_name = f"retrieve_facts_{strategy}"
                await services.agent_status_service.update_step(status_key, step_name, agent_id)
            except Exception as e:
                logfire.error("Failed to update status step in retrieve_facts: {error}", error=str(e))

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
            is_relevant = await self.grade_relevance(query, doc_content, status_key=status_key, agent_id=agent_id)
            if is_relevant:
                relevant_facts.append(doc_content)
        
        logfire.info("Adaptive RAG: retrieved {total} documents, {relevant} are relevant.", total=len(facts), relevant=len(relevant_facts))
        return relevant_facts

    async def run_agentic_flow(self, query: str, agent, message_history, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> Any:
        """Executes the complete Adaptive RAG + Self-RAG loop."""
        active_agent_id = agent_id or "KitaAgent"

        # 0. Condense query if history exists
        standalone_query = await self.condense_query(query, message_history, status_key=status_key, agent_id=active_agent_id)

        # 1. Route query
        strategy = await self.route_query(standalone_query, status_key=status_key, agent_id=active_agent_id)
        logfire.info("Adaptive RAG: query={query!r} (condensed={standalone_query!r}) routed to strategy={strategy!r}", query=query, standalone_query=standalone_query, strategy=strategy)
        
        visited_strategies = {strategy}
        current_query = standalone_query
        relevant_facts = []
        
        if strategy != "none":
            relevant_facts = await self.retrieve_facts(strategy, current_query, status_key=status_key, agent_id=active_agent_id)
            
            # If no relevant facts found, failover to alternative strategy
            if not relevant_facts:
                alt_strategy = "web" if strategy == "vector" else "vector"
                if alt_strategy not in visited_strategies:
                    logfire.info("Adaptive RAG: No relevant facts found. Failing over to {alt_strategy}", alt_strategy=alt_strategy)
                    visited_strategies.add(alt_strategy)
                    strategy = alt_strategy
                    current_query = await self.rewrite_query(standalone_query, status_key=status_key, agent_id=active_agent_id)
                    relevant_facts = await self.retrieve_facts(strategy, current_query, status_key=status_key, agent_id=active_agent_id)

        # Loop for Generation & Evaluation (Self-RAG)
        max_attempts = 3
        result = None
        
        deps = self._get_deps(agent_id, status_key=status_key)
        
        for attempt in range(1, max_attempts + 1):
            logfire.info("Adaptive RAG Self-Evaluation: Attempt {attempt} of {max_attempts}", attempt=attempt, max_attempts=max_attempts)
            
            # Prepare generator prompt instructions
            run_instructions = None
            if relevant_facts:
                facts_block = "\n\n".join([f"Fact {i}:\n{fact}" for i, fact in enumerate(relevant_facts, 1)])
                run_instructions = (
                    f"[RELEVANT RETRIEVED CONTEXT]\n"
                    f"{facts_block}\n\n"
                    f"INSTRUCTION: Draft your response using ONLY the relevant retrieved facts above when applicable. "
                    f"Do not hallucinate or state details not supported by the facts."
                )
            
            # Update status to generating response
            if status_key:
                try:
                    from app.dependencies.services import get_services
                    services = get_services(self.org_id)
                    await services.agent_status_service.update_step(status_key, "generate_response", active_agent_id)
                except Exception as e:
                    logfire.error("Failed to update status step before generator run: {error}", error=str(e))

            # Execute agent (Generator)
            logfire.info("Adaptive RAG: Invoking generator agent.")
            result = await agent.run(
                query,
                message_history=message_history,
                deps=deps,
                instructions=run_instructions
            )
            draft_response = result.data if hasattr(result, "data") else getattr(result, "content", str(result))
            
            # If no retrieval happened, we are done
            if not relevant_facts:
                logfire.info("Adaptive RAG: No retrieval strategy was used. Returning agent response directly.")
                return result
                
            # Grade groundedness
            is_grounded = await self.grade_groundedness(relevant_facts, draft_response, status_key=status_key, agent_id=active_agent_id)
            if not is_grounded:
                logfire.warning("Adaptive RAG: Draft answer failed groundedness (hallucination check). Retrying.")
                # Rewrite query and search again to find better facts
                current_query = await self.rewrite_query(current_query, status_key=status_key, agent_id=active_agent_id)
                relevant_facts = await self.retrieve_facts(strategy, current_query, status_key=status_key, agent_id=active_agent_id)
                continue
                
            # Grade completeness
            is_complete = await self.grade_completeness(standalone_query, draft_response, status_key=status_key, agent_id=active_agent_id)
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
                    current_query = await self.rewrite_query(standalone_query, status_key=status_key, agent_id=active_agent_id)
                    relevant_facts = await self.retrieve_facts(strategy, current_query, status_key=status_key, agent_id=active_agent_id)
                else:
                    # If we already tried all strategies, try rewriting the query and searching again
                    logfire.info("Adaptive RAG: Both search strategies already tried. Rewriting query and retrying current strategy.")
                    current_query = await self.rewrite_query(current_query, status_key=status_key, agent_id=active_agent_id)
                    relevant_facts = await self.retrieve_facts(strategy, current_query, status_key=status_key, agent_id=active_agent_id)

        # If we exhausted max attempts, return the last draft
        logfire.warning("Adaptive RAG: Exhausted max attempts. Returning last generated result.")
        return result
