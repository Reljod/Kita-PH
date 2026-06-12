import asyncio
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Protocol
from bson import ObjectId
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
import logfire

from app.models.rag import RagResponse
from app.db import TenantCollection
from app.services.rag.nested_data_enrichment_service import INestedDataEnrichmentService, get_nested_value
from app.services.rag.mongodb_vector_search_rag_service import MongoDBVectorSearchRagService
from app.services.rag.mongodb_text_search_rag_service import MongoDBTextSearchRagService
from app.services.rag.reranking_rag_service import RerankingRagService

class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(description="True if the context provides enough information to fully answer the query, False otherwise.")

class IRetrievalService(Protocol):
    async def search(self, query: str, limit: int = 5) -> List[RagResponse]:
        ...

class RetrievalService(IRetrievalService):
    def __init__(
        self,
        collection: TenantCollection, # file_parsed_flattened
        vector_service: MongoDBVectorSearchRagService,
        text_service: MongoDBTextSearchRagService,
        reranking_service: RerankingRagService,
        parse_collection: TenantCollection, # file_parse (contains the original unflattened result)
        nested_data_enrichment_service: INestedDataEnrichmentService,
        agent_id: Optional[str] = None
    ):
        self.collection = collection
        self.vector_service = vector_service
        self.text_service = text_service
        self.reranking_service = reranking_service
        self.parse_collection = parse_collection
        self.nested_data_enrichment_service = nested_data_enrichment_service
        self.agent_id = agent_id
        self._llm_model = None

    def _get_llm_model(self):
        if self._llm_model is None:
            model_name = "deepseek/deepseek-v4-flash"
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            self._llm_model = OpenRouterModel(
                model_name,
                provider=OpenRouterProvider(api_key=api_key)
            )
        return self._llm_model

    async def search(self, query: str, limit: int = 5) -> List[RagResponse]:
        """
        Retrieval Service orchestrates the Ultimate Structured RAG Pipeline.
        """
        logfire.info("Starting RAG search: query={query!r}, limit={limit}", query=query, limit=limit)

        # 1. Parallel search against the flattened leaves (Stage One)
        vector_task = self.vector_service.vector_search(query, limit=100, agent_id=self.agent_id)
        text_task = self.text_service.text_search(query, limit=100, agent_id=self.agent_id)

        vector_results, text_results = await asyncio.gather(vector_task, text_task)
        logfire.info("Stage 1 completed: vector_results={vector_count}, text_results={text_count}",
                     vector_count=len(vector_results), text_count=len(text_results))
        logfire.info("Stage 1 Vector Search texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("text") or doc.get("heading_to_text")} for doc in vector_results])
        logfire.info("Stage 1 Text Search texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("text") or doc.get("heading_to_text")} for doc in text_results])

        # 2. Reciprocal Rank Fusion Merger (Stage Two)
        rrf_candidates = self._apply_rrf(vector_results, text_results, limit=100)
        logfire.info("Stage 2 RRF fusion completed: rrf_candidates={count}", count=len(rrf_candidates))
        logfire.info("Stage 2 RRF candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("text") or doc.get("heading_to_text")} for doc in rrf_candidates])
        if not rrf_candidates:
            return []

        # 3. Parent Fetch & Auto-Merging Sibling Thresholds (Stage Three)
        # Fetch the parse documents to reconstruct trees in memory
        file_ids = {doc["file_id"] for doc in rrf_candidates if doc.get("file_id")}
        parse_docs = list(self.parse_collection.find({"file_id": {"$in": list(file_ids)}}))
        parse_doc_map = {doc["file_id"]: doc for doc in parse_docs}

        # Build document trees in memory for path lookup
        trees_map = {}
        for fid, doc in parse_doc_map.items():
            tree, _ = self.nested_data_enrichment_service.build_hierarchy_and_leaves(
                parse_result=doc.get("result", {}),
                file_id=fid,
                org_id=self.collection.org_id
            )
            trees_map[fid] = tree

        # Run Auto-Merging sibling threshold
        merged_candidates = self._auto_merge_siblings(rrf_candidates, trees_map, threshold_ratio=0.35)
        logfire.info("Stage 3 Auto-merging siblings completed: merged_candidates={count}", count=len(merged_candidates))
        logfire.info("Stage 3 merged candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("text") or doc.get("heading_to_text")} for doc in merged_candidates])

        # 4. Fetch sub-trees and serialize content
        resolved_candidates = []
        for cand in merged_candidates:
            fid = cand.get("file_id")
            path = cand.get("json_path")

            # If it's manual input, it has no parse tree
            if path == "manual_input" or not fid or fid not in trees_map:
                cand["context_text"] = cand.get("heading_to_text") or cand.get("text")
                cand["sub_tree_val"] = cand.get("content")
                resolved_candidates.append(cand)
                continue

            tree = trees_map[fid]
            sub_tree_val = get_nested_value(tree, path.split("."))
            if sub_tree_val is None:
                continue

            cand["sub_tree_val"] = sub_tree_val
            cand["context_text"] = self._serialize_sub_tree(cand.get("heading_text") or "", sub_tree_val)
            resolved_candidates.append(cand)

        logfire.info("Stage 3 resolved candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in resolved_candidates])

        # Deduplicate resolved candidates by (file_id, json_path)
        unique_resolved_candidates = []
        seen_resolved_keys = set()
        for cand in resolved_candidates:
            fid = cand.get("file_id")
            path = cand.get("json_path")
            if fid and path and path != "manual_input":
                key = (fid, path)
            else:
                key = str(cand.get("_id") or id(cand))

            if key not in seen_resolved_keys:
                seen_resolved_keys.add(key)
                unique_resolved_candidates.append(cand)

        logfire.info("Stage 3 unique resolved candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in unique_resolved_candidates])

        # 5. Stage Four: Cross-Encoder Reranking
        # Scores the sub-trees and filters out the bottom 90%
        reranked_candidates = await self.reranking_service.rerank(query, unique_resolved_candidates, limit=10)
        logfire.info("Stage 4 Cross-Encoder reranking completed: reranked_candidates={count}", count=len(reranked_candidates))
        logfire.info("Stage 4 reranked candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in reranked_candidates])

        # 6. Stateful Recursive Retrieval (Agentic Loop) on the top candidates
        final_candidates = []
        for cand in reranked_candidates:
            fid = cand.get("file_id")
            path = cand.get("json_path")

            # Skip recursion for manual inputs or missing trees
            if path == "manual_input" or not fid or fid not in trees_map:
                final_candidates.append(cand)
                continue

            # Perform stateful recursive evaluation loop
            updated_cand = await self._recursive_evaluation_loop(query, cand, trees_map[fid])
            final_candidates.append(updated_cand)

        logfire.info("Stage 6 final candidates (before dedup) texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in final_candidates])

        # Deduplicate final candidates after recursive evaluation
        unique_final_candidates = []
        seen_final_keys = set()
        for cand in final_candidates:
            fid = cand.get("file_id")
            path = cand.get("json_path")
            if fid and path and path != "manual_input":
                key = (fid, path)
            else:
                key = str(cand.get("_id") or id(cand))

            if key not in seen_final_keys:
                seen_final_keys.add(key)
                unique_final_candidates.append(cand)

        logfire.info("Stage 6 unique final candidates texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in unique_final_candidates])

        # 7. Final Rerank and format output (take top `limit`, default 5)
        final_reranked = await self.reranking_service.rerank(query, unique_final_candidates, limit=limit)
        logfire.info("RAG search finished. Returning {count} final responses.", count=len(final_reranked))
        logfire.info("Stage 7 final reranked results texts: {texts}", texts=[{"path": doc.get("json_path"), "text": doc.get("context_text")} for doc in final_reranked])

        # Format as RagResponse for compatibility
        return [self._format_response(cand, query) for cand in final_reranked]

    def _apply_rrf(self, list_a: List[dict], list_b: List[dict], limit: int = 100) -> List[dict]:
        """Reciprocal Rank Fusion merger."""
        scores = {}
        doc_map = {}

        def process_list(lst):
            for rank, doc in enumerate(lst):
                doc_id = str(doc["_id"])
                doc_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank + 1))

        process_list(list_a)
        process_list(list_b)

        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:limit]]

    def _auto_merge_siblings(self, candidates: List[dict], trees_map: Dict[str, dict], threshold_ratio: float) -> List[dict]:
        """Auto-merges sibling paths to their parents if enough sibling leaves are hit."""
        # Group hits by (file_id, parent_path)
        grouped_hits: Dict[tuple, List[dict]] = {}
        for cand in candidates:
            path = cand.get("json_path", "")
            fid = cand.get("file_id")
            if not fid or "." not in path:
                continue
            parts = path.split(".")
            parent_path = ".".join(parts[:-1])
            grouped_hits.setdefault((fid, parent_path), []).append(cand)

        merged_paths = []
        processed_candidates = []

        # Track which candidates have been merged
        merged_ids = set()

        for (fid, parent_path), hits in grouped_hits.items():
            if fid not in trees_map:
                continue
            tree = trees_map[fid]
            parent_val = get_nested_value(tree, parent_path.split("."))

            if isinstance(parent_val, dict):
                total_siblings = len(parent_val.keys())
                hits_count = len(hits)
                ratio = hits_count / total_siblings if total_siblings > 0 else 0.0

                if ratio >= threshold_ratio:
                    # Merge into a single parent candidate!
                    rep_hit = hits[0]
                    merged_cand = {
                        "_id": rep_hit["_id"],
                        "file_id": fid,
                        "org_id": rep_hit["org_id"],
                        "json_path": parent_path,
                        "heading_text": " > ".join(rep_hit.get("heading_text", "").split(" > ")[:-1]),
                        "page": rep_hit.get("page", 1),
                        "location": rep_hit.get("location"),
                        "agent_id": rep_hit.get("agent_id")
                    }
                    processed_candidates.append(merged_cand)
                    for h in hits:
                        merged_ids.add(str(h["_id"]))

        # Keep non-merged items as is
        for cand in candidates:
            if str(cand["_id"]) not in merged_ids:
                processed_candidates.append(cand)

        return processed_candidates

    def _serialize_sub_tree(self, heading: str, val: Any) -> str:
        """Helper to cleanly format a nested sub-tree for CrossEncoder or LLM consumption."""
        heading_prefix = f"Location: {heading}\n" if heading else ""
        if isinstance(val, dict):
            # Format nicely as lines
            lines = []
            for k, v in val.items():
                if isinstance(v, dict):
                    lines.append(f"{k}: {json.dumps(v)}")
                else:
                    lines.append(f"{k}: {v}")
            return heading_prefix + "\n".join(lines)
        return heading_prefix + str(val)

    async def _recursive_evaluation_loop(self, query: str, candidate: dict, tree: dict) -> dict:
        """Stateful agentic loop evaluating context sufficiency using deepseek-v4-flash."""
        path = candidate["json_path"]
        fid = candidate["file_id"]

        # Agent for lightweight evaluation
        agent = Agent(
            model=self._get_llm_model(),
            output_type=EvaluationResult,
            system_prompt=(
                "You are an expert evaluator. Given a user query and a retrieved text/JSON block, "
                "determine whether the block contains sufficient details to accurately and fully answer the query. "
                "Respond with is_sufficient = True or False."
            )
        )

        max_iterations = 3
        current_path = path

        for iteration in range(max_iterations):
            # Fetch sub-tree
            sub_tree_val = get_nested_value(tree, current_path.split("."))
            if sub_tree_val is None:
                break

            serialized_content = self._serialize_sub_tree(candidate.get("heading_text") or "", sub_tree_val)

            # Circuit Breaker 1: Token/Byte size limit (8,000 tokens ≈ 32,000 bytes)
            byte_size = len(serialized_content.encode('utf-8'))
            if byte_size > 32000:
                # Halt traversal: context size limit reached
                break

            # Call evaluator
            prompt = f"Query: {query}\nContext Content:\n{serialized_content}"
            try:
                eval_res = await agent.run(prompt)
                is_sufficient = eval_res.output.is_sufficient
            except Exception as e:
                logfire.error("Evaluator error on iteration {iteration}: {error}", iteration=iteration, error=str(e))
                is_sufficient = True # Fallback: stop loop to be safe

            logfire.info("Agentic evaluation: iteration={iteration}, current_path={current_path}, byte_size={byte_size}, is_sufficient={is_sufficient}",
                         iteration=iteration, current_path=current_path, byte_size=byte_size, is_sufficient=is_sufficient)
            if is_sufficient:
                # Context is sufficient! Return immediately.
                candidate["json_path"] = current_path
                candidate["sub_tree_val"] = sub_tree_val
                candidate["context_text"] = serialized_content
                return candidate

            # Move up one level: chop off the last key of the path
            if "." in current_path:
                current_path = ".".join(current_path.split(".")[:-1])
            else:
                # Reached root level, cannot traverse further
                break

        # Final fallback
        sub_tree_val = get_nested_value(tree, current_path.split("."))
        candidate["json_path"] = current_path
        candidate["sub_tree_val"] = sub_tree_val
        candidate["context_text"] = self._serialize_sub_tree(candidate.get("heading_text") or "", sub_tree_val)
        return candidate

    def _format_response(self, doc: dict, query: str) -> RagResponse:
        return RagResponse(
            id=str(doc["_id"]),
            title=doc.get("heading_text") or doc.get("json_path") or "RAG Block",
            content=doc.get("context_text") or doc.get("text") or "",
            status="completed",
            agent_id=doc.get("agent_id"),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            question=query,
            answer=doc.get("context_text") or doc.get("content") or "",
            original_content=doc.get("text")
        )
