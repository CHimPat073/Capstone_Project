"""
chat_service.py — Connects the API to the existing LangGraph reasoning loop.
Reuses existing app.reasoning.agent_graph and app.reasoning.llm_client modules.
"""

import re
from typing import Any, Dict, List, Optional

from app.reasoning.agent_graph import answer
from app.evaluation.ragas_evaluator import calculate_faithfulness, calculate_answer_relevancy
from backend.services.document_service import document_service
from backend.models.schemas import ChatResponse, Citation, EvalMetrics, LoopStatus, Source


# Standard clarification suggestions if the user asks a very vague question
DEFAULT_SUGGESTIONS = [
    "What are the payment terms?",
    "What are the termination conditions?",
    "Who are the parties to the agreement?",
    "What is the governing law?",
]

# Common out-of-domain patterns for quick classification
OUT_OF_DOMAIN_PATTERNS = [
    r"\bcapital of\b",
    r"\bworld cup\b",
    r"\bweather in\b",
    r"\bwho is the president\b",
    r"\bwrite a python script\b",
    r"\bwho won\b",
    r"\bhow to bake\b",
]


class ChatService:
    def process_chat(self, question: str) -> ChatResponse:
        """
        Processes a user question through the existing LangGraph RAG pipeline.
        Classifies status into:
        - "answered"
        - "needs_clarification"
        - "out_of_domain"
        - "retrieval_failed"
        - "grounding_failed"
        """
        index = document_service.current_index
        if not index:
            return ChatResponse(
                success=False,
                status="retrieval_failed",
                answer="No document is currently loaded. Please upload a PDF document first.",
                citations=[],
                sources=[],
                loop=LoopStatus(
                    query_relevant=False,
                    retrieval_attempts=0,
                    rewrites=0,
                    grounding_passed=False,
                ),
                suggestions=DEFAULT_SUGGESTIONS,
            )

        q_clean = question.strip()

        # PART 4: Check for extremely vague / unclear / isolated single-word queries
        vague_short_queries = {"what", "how", "why", "that", "it", "more", "tell me", "what about that", "details", "explain", "help"}
        words = q_clean.split()
        is_isolated_word = len(words) == 1 and not q_clean.endswith("?") and q_clean.lower() not in {"parties", "term", "termination", "payment", "indemnification", "confidentiality", "warranty", "liability", "governing", "jurisdiction", "arbitration", "assignment", "amendment", "severability"}

        if q_clean.lower() in vague_short_queries or is_isolated_word:
            return ChatResponse(
                success=False,
                status="needs_clarification",
                answer="I'm not sure what part of the document you're referring to. Could you please ask a specific question about the document?",
                citations=[],
                sources=[],
                loop=LoopStatus(
                    query_relevant=False,
                    retrieval_attempts=0,
                    rewrites=0,
                    grounding_passed=False,
                    original_query=q_clean,
                ),
                eval_metrics=EvalMetrics(
                    document_relevance_score=0.0,
                    relevance_level="Low (Needs Clarification)",
                    context_alignment=0.0,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                ),
                suggestions=DEFAULT_SUGGESTIONS,
            )

        # PART 3: Fast Out-of-Domain heuristic check
        for pat in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pat, q_clean.lower()):
                return ChatResponse(
                    success=False,
                    status="out_of_domain",
                    answer="This question is outside the scope of the uploaded document. I can only answer questions based on the document.",
                    citations=[],
                    sources=[],
                    loop=LoopStatus(
                        query_relevant=False,
                        retrieval_attempts=0,
                        rewrites=0,
                        grounding_passed=False,
                        original_query=q_clean,
                    ),
                    eval_metrics=EvalMetrics(
                        document_relevance_score=0.0,
                        relevance_level="Low (Out of Scope)",
                        context_alignment=0.0,
                        faithfulness=0.0,
                        answer_relevancy=0.0,
                    ),
                    suggestions=DEFAULT_SUGGESTIONS,
                )

        # Execute existing LangGraph reasoning loop with current-session conversation context
        recent_history = document_service.conversation_history[-4:]
        result = answer(
            index,
            q_clean,
            conversation_history=recent_history,
        )

        answer_text = result["answer"]
        raw_citations = result.get("citations", [])
        raw_sources = result.get("retrieved_chunks", [])
        is_grounded = result.get("is_grounded", True)
        confidence = result.get("confidence", "high")
        rewrite_count = result.get("rewrite_count", 0)
        resolved_query = result.get("retrieval_query", q_clean)
        follow_ups = result.get("follow_up_suggestions", [])

        # Map citations
        citations = [
            Citation(page=c.get("page", 1), text=c.get("text"))
            for c in raw_citations
        ]

        # Map top retrieved sources (excerpts only, no huge chunks)
        sources = [
            Source(
                page=doc.metadata.get("page_num", 1),
                text=doc.page_content.strip()[:280] + ("..." if len(doc.page_content.strip()) > 280 else ""),
                chunk_type=doc.metadata.get("chunk_type", "text"),
            )
            for doc in raw_sources[:5]
        ]

        # Classify the outcome
        loop_status = LoopStatus(
            query_relevant=True,
            retrieval_attempts=rewrite_count + 1,
            rewrites=rewrite_count,
            grounding_passed=is_grounded,
            original_query=q_clean,
            resolved_query=resolved_query,
        )

        # Detect out-of-domain refusal from LLM/agent
        refusal_phrases = [
            "outside the scope",
            "not mentioned in the provided",
            "cannot answer this question based on",
            "could not find enough information",
            "couldn't find enough information",
        ]
        is_refusal = any(p in answer_text.lower() for p in refusal_phrases)

        if not is_grounded:
            status = "grounding_failed"
            success = False
            final_answer = "I couldn't verify this answer against the document."
        elif confidence == "low" and is_refusal:
            if rewrite_count >= 2:
                status = "retrieval_failed"
                success = False
                final_answer = "I couldn't find enough information in the document to answer this question reliably."
            else:
                status = "out_of_domain"
                success = False
                final_answer = "This question doesn't appear to be related to the uploaded document. I can only answer questions based on the uploaded document."
        else:
            status = "answered"
            success = True
            final_answer = answer_text

        # Update in-memory session conversation history
        document_service.conversation_history.append({"role": "user", "content": q_clean})
        document_service.conversation_history.append({"role": "assistant", "content": final_answer})

        suggestions = follow_ups if follow_ups else DEFAULT_SUGGESTIONS

        # Compute Evaluation & Relevance Metrics (Faithfulness, Context Alignment, Relevancy)
        faith_score = calculate_faithfulness(final_answer, raw_sources, is_grounded, result.get("unsupported_claims", []))
        ans_rel_score = calculate_answer_relevancy(q_clean, final_answer)

        # Measure lexical/semantic context alignment between query and retrieved chunks
        q_tokens = set(re.findall(r"\b\w+\b", q_clean.lower())) - {"what", "which", "where", "when", "does", "have", "with", "from", "the", "and", "for", "about", "tell", "me", "are", "is"}
        all_chunk_text = " ".join(d.page_content.lower() for d in raw_sources)
        if q_tokens:
            matches = sum(1 for tok in q_tokens if tok in all_chunk_text)
            alignment_score = round(min(1.0, matches / len(q_tokens)), 2)
        else:
            alignment_score = 0.85

        # Overall document relevance score (weighted blend: 40% alignment, 35% faithfulness, 25% relevancy)
        if status == "out_of_domain":
            doc_rel_score = 0.05
            rel_level = "Low (Out of Scope)"
            faith_score = 0.0
            alignment_score = 0.0
            ans_rel_score = 0.0
        elif status == "retrieval_failed":
            doc_rel_score = 0.20
            rel_level = "Low (Insufficient Context)"
            faith_score = round(min(faith_score, 0.20), 2)
            ans_rel_score = round(min(ans_rel_score, 0.25), 2)
        elif status == "grounding_failed":
            doc_rel_score = 0.30
            rel_level = "Low (Unverified Claims)"
            faith_score = round(min(faith_score, 0.35), 2)
        else:
            doc_rel_score = round(0.40 * alignment_score + 0.35 * faith_score + 0.25 * ans_rel_score, 2)
            if doc_rel_score >= 0.75:
                rel_level = "High"
            elif doc_rel_score >= 0.45:
                rel_level = "Moderate"
            else:
                rel_level = "Low"

        eval_metrics = EvalMetrics(
            document_relevance_score=doc_rel_score,
            relevance_level=rel_level,
            context_alignment=alignment_score,
            faithfulness=faith_score,
            answer_relevancy=ans_rel_score,
        )

        return ChatResponse(
            success=success,
            answer=final_answer,
            status=status,
            citations=citations if success else [],
            sources=sources,
            loop=loop_status,
            eval_metrics=eval_metrics,
            suggestions=suggestions,
        )


chat_service = ChatService()
