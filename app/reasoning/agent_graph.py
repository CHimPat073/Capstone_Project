"""
agent_graph.py — Phase 3: Agentic Reasoning Loop via LangGraph.

WORKFLOW ARCHITECTURE:
    START
      ↓
    retrieve             (calls Phase 2 retrieve(k=5))
      ↓
    grade                (Is context sufficient?)
      ↓
    [conditional router]
      ├─ Sufficient = YES              ──► generate ──► hallucination_check ──► END
      ├─ Sufficient = NO & rewrites < 2 ──► rewrite  ──► retrieve (loop back)
      └─ Sufficient = NO & rewrites >= 2──► generate ──► hallucination_check ──► END
                                          (with low-confidence indicator)

RULES ENFORCED:
    - Maximum 2 query rewrites (bounded loop)
    - Grounding & citation validation: validates cited pages exist in retrieved chunks
    - No persistent storage introduced
"""

import json
import re
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from app.retrieval.hybrid_retriever import retrieve
from app.reasoning.query_router import decompose_query, route_query
from app.reasoning.llm_client import (
    LLMClient,
    GRADER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    FOLLOWUP_RESOLVER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    HALLUCINATION_CHECK_SYSTEM_PROMPT,
)


class AgentState(TypedDict):
    """Minimal shared state for the LangGraph reasoning workflow."""
    query: str
    current_query: str
    conversation_history: List[Dict[str, str]]
    retrieved_chunks: List[Document]
    rewrite_count: int
    is_sufficient: bool
    grade_reason: str
    answer: str
    citations: List[Dict[str, Any]]
    is_grounded: bool
    confidence: str                 # "high", "medium", or "low"
    unsupported_claims: List[str]
    follow_up_suggestions: List[str]
    sub_queries: List[str]
    retrieved_pages: List[int]


def _format_context(docs: List[Document]) -> str:
    """Format retrieved documents with unambiguous [Page X] tags."""
    formatted_pieces = []
    for doc in docs:
        page_num = doc.metadata.get("page_num", "Unknown")
        formatted_pieces.append(f"[Page {page_num}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_pieces)


def _is_overview_query(question: str) -> bool:
    """Detect requests that need representative coverage across the PDF."""
    return bool(re.search(
        r"\b(summar(?:y|ize|ise)|overview|main points|key points|highlights|entire document|whole document)\b",
        question or "",
        re.IGNORECASE,
    ))


def _representative_chunks(index: Any, limit: int = 8) -> List[Document]:
    """Choose text chunks spread across the document for overview questions."""
    chunks = [doc for doc in getattr(index, "child_chunks", [])
              if doc.metadata.get("chunk_type", "text") == "text" and doc.page_content.strip()]
    by_page: Dict[int, List[Document]] = {}
    for doc in chunks:
        try:
            page = int(doc.metadata.get("page_num", 0))
        except (TypeError, ValueError):
            continue
        by_page.setdefault(page, []).append(doc)
    pages = sorted(by_page)
    if not pages:
        return chunks[:limit]
    if len(pages) <= limit:
        chosen_pages = pages
    else:
        chosen_pages = sorted({pages[round(i * (len(pages) - 1) / (limit - 1))]
                               for i in range(limit)})
    return [max(by_page[page], key=lambda doc: len(doc.page_content)) for page in chosen_pages]


def _is_full_document_extraction_query(question: str) -> bool:
    """Detect exhaustive extraction requests that cannot be answered from top-k retrieval."""
    text = (question or "").lower()
    asks_for_collection = bool(re.search(r"\b(all|every|list|mention|extract|identify|find|which|what)\b", text))
    asks_for_field = bool(re.search(
        r"\b(names?|people|persons?|authors?|organizations?|companies|dates?|deadlines?|results?|metrics?|skills?|tools?|technologies)\b",
        text,
    ))
    return asks_for_collection and asks_for_field


def _offline_full_document_answer(index: Any, question: str) -> Optional[Dict[str, Any]]:
    """Extract common entity lists from every page when no external LLM is configured."""
    page_docs = sorted(getattr(index, "page_docs", []), key=lambda doc: int(doc.metadata.get("page_num", 0)))
    if not page_docs:
        return None
    query = question.lower()
    wants_names = bool(re.search(r"\b(names?|people|persons?|authors?|organizations?|companies)\b", query))
    wants_dates = bool(re.search(r"\b(dates?|deadlines?)\b", query))
    wants_results = bool(re.search(r"\b(results?|metrics?)\b", query))
    answer_sections: List[str] = []
    citation_rows: List[tuple] = []

    if wants_names:
        entities = _extract_named_entities(page_docs)
        if entities:
            answer_sections.append("Names and named entities found:\n" + "\n".join(
                f"- {name} [Page {page}]" for name, page in entities
            ))
            citation_rows.extend((page, name) for name, page in entities)

    if wants_dates:
        dates = _extract_dates(page_docs)
        if dates:
            answer_sections.append("Dates found:\n" + "\n".join(
                f"- {date} [Page {page}]" for date, page in dates
            ))
            citation_rows.extend((page, date) for date, page in dates)

    if wants_results:
        results = _extract_result_sentences(page_docs)
        if results:
            answer_sections.append("Reported results:\n" + "\n".join(
                f"- {sentence} [Page {page}]" for sentence, page in results
            ))
            citation_rows.extend((page, sentence) for sentence, page in results)

    if not answer_sections:
        return None

    relevant_chunks = []
    for page, text in citation_rows:
        match = next((doc for doc in getattr(index, "child_chunks", [])
                      if int(doc.metadata.get("page_num", 0)) == page
                      and any(token in doc.page_content.lower()
                              for token in re.findall(r"[a-zA-Z]{4,}", text.lower())[:3])), None)
        if match and match not in relevant_chunks:
            relevant_chunks.append(match)
    citations = []
    for page, text in citation_rows:
        citation: Dict[str, Any] = {"page": page, "text": text[:180]}
        geometry = _bbox_for_text(index, page, text)
        if geometry:
            citation.update(geometry)
        citations.append(citation)
    unique_pages = sorted({page for page, _ in citation_rows})
    return {
        "query": question,
        "retrieval_query": question,
        "answer": "\n\n".join(answer_sections),
        "citations": citations,
        "retrieved_chunks": relevant_chunks or page_docs[:5],
        "is_grounded": True,
        "confidence": "high",
        "rewrite_count": 0,
        "unsupported_claims": [],
        "grade_reason": "Extracted requested fields from all document pages.",
        "follow_up_suggestions": [
            "Show the source passage for one of these items.",
            "Group these items by document section.",
        ],
        "sub_queries": [question],
        "retrieved_pages": unique_pages,
        "route_strategy": "full_document_extraction",
    }


def _extract_named_entities(page_docs: List[Document], limit: int = 40) -> List[tuple]:
    """Extract readable person and organization names with conservative filtering."""
    titled_pattern = re.compile(
        r"\b(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+(?:[A-Z]\.?\s*){0,3}[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b"
    )
    general_pattern = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+(?:[A-Z][a-z]{2,}|[A-Z]{2,})){1,3}\b")
    blocked_words = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "week", "date", "day", "name", "topic", "module",
        "completed", "contents", "chapter", "table", "figure", "page", "project", "report",
        "internship", "certificate", "declaration", "acknowledgement", "abstract", "introduction",
    }
    role_words = {
        "manager", "senior", "chief", "dean", "professor", "chair", "officer", "director",
        "head", "supervisor", "coordinator", "engineer", "department", "division", "training",
    }
    non_name_phrases = {
        "under supervision", "computing science engineering", "artificial intelligence", "job training",
        "executive management", "general manager", "unit head", "managing director", "program chair",
        "assistant professor", "chief manager", "section officer", "systems department", "systems division",
    }
    generic_entity_words = {
        "program", "vocational", "dean", "pro", "vice", "tech", "cse", "mean", "absolute", "error",
        "data", "pipeline", "temporal", "features", "supervised", "forecasting", "random", "forest",
        "reagent", "optimization", "anomaly", "detection", "isolation", "drift", "monitor", "population",
        "stability", "index", "agentic", "reporting", "stack", "integration", "information", "learning",
        "objectives", "proficiency", "recovery", "implement", "engine", "school", "organization",
    }
    organization_suffixes = {"limited", "university", "institute", "corporation", "company"}
    found: Dict[str, tuple] = {}
    for doc in page_docs:
        page = int(doc.metadata.get("page_num", 1))
        text = doc.page_content.replace("_", " ")
        titled_matches = [(match, True) for match in titled_pattern.finditer(text)]
        general_matches = [(match, False) for match in general_pattern.finditer(text)]
        for match, has_title in sorted(titled_matches + general_matches, key=lambda item: item[0].start()):
            value = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:()[]")
            parts = value.split()
            role_index = next((index for index, word in enumerate(parts)
                               if word.lower().strip(".") in role_words), None)
            if role_index is not None:
                if has_title and role_index >= 2:
                    value = " ".join(parts[:role_index])
                else:
                    continue
            words = {word.lower().strip(".") for word in value.split()}
            if len(value) < 5 or words & blocked_words or value.lower() in non_name_phrases:
                continue
            if not has_title:
                suffix_position = next((index for index, word in enumerate(value.split())
                                        if word.lower().strip(".") in organization_suffixes), None)
                if suffix_position is not None:
                    value = " ".join(value.split()[:suffix_position + 1])
                    words = {word.lower().strip(".") for word in value.split()}
                elif page > 4:
                    continue
                elif words & generic_entity_words:
                    continue
            canonical = re.sub(r"^(?:mr|ms|mrs|dr|prof)\.?\s+", "", value, flags=re.I)
            key = re.sub(r"\W+", "", canonical.lower())
            if any(key in existing_key or existing_key in key for existing_key in found
                   if min(len(key), len(existing_key)) >= 8):
                continue
            if key not in found:
                found[key] = (value, page)
            elif has_title and not re.match(r"^(?:Mr|Ms|Mrs|Dr|Prof)\.?\s", found[key][0]):
                found[key] = (value, page)
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    return list(found.values())


def _bbox_for_text(index: Any, page_number: int, quote: str) -> Optional[Dict[str, Any]]:
    """Find a tight PDF word box for an exact or near-exact citation passage."""
    page_doc = next((doc for doc in getattr(index, "page_docs", [])
                     if int(doc.metadata.get("page_num", 0)) == int(page_number)), None)
    if page_doc is None:
        return None
    try:
        words = json.loads(page_doc.metadata.get("bbox_words_json", "[]"))
    except (TypeError, ValueError):
        return None
    quote_tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", (quote or "").lower()) if len(token) > 1]
    word_tokens = [re.sub(r"\W+", "", str(word.get("text", "")).lower()) for word in words]
    if not quote_tokens or not word_tokens:
        return None
    best_start = -1
    best_length = 0
    for start, token in enumerate(word_tokens):
        if token != quote_tokens[0]:
            continue
        length = 0
        for offset, wanted in enumerate(quote_tokens[:60]):
            if start + offset >= len(word_tokens) or word_tokens[start + offset] != wanted:
                break
            length += 1
        if length > best_length:
            best_start, best_length = start, length
    required = 1 if len(quote_tokens) == 1 else 2
    if best_length < required:
        return None
    matched = words[best_start:best_start + best_length]
    return {
        "bbox": [
            round(min(float(word["x0"]) for word in matched), 2),
            round(min(float(word["top"]) for word in matched), 2),
            round(max(float(word["x1"]) for word in matched), 2),
            round(max(float(word["bottom"]) for word in matched), 2),
        ],
        "page_width": page_doc.metadata.get("page_width"),
        "page_height": page_doc.metadata.get("page_height"),
    }


def _extract_dates(page_docs: List[Document], limit: int = 50) -> List[tuple]:
    patterns = (
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*|\s+)\d{4}\b", re.I),
    )
    found: Dict[str, tuple] = {}
    for doc in page_docs:
        page = int(doc.metadata.get("page_num", 1))
        for pattern in patterns:
            for value in pattern.findall(doc.page_content):
                key = value.lower()
                found.setdefault(key, (value, page))
                if len(found) >= limit:
                    return list(found.values())
    return list(found.values())


def _extract_result_sentences(page_docs: List[Document], limit: int = 12) -> List[tuple]:
    markers = re.compile(r"\b(result|achiev|improv|accuracy|precision|recall|latency|verified|reduced|increased|completed|implemented)\w*\b|\d+(?:\.\d+)?%", re.I)
    found = []
    seen = set()
    for doc in page_docs:
        page = int(doc.metadata.get("page_num", 1))
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", doc.page_content):
            cleaned = re.sub(r"\s+", " ", sentence).strip(" |-")
            if len(cleaned) < 30 or not markers.search(cleaned):
                continue
            key = re.sub(r"\W+", "", cleaned.lower())[:140]
            if key in seen:
                continue
            seen.add(key)
            found.append((cleaned[:280], page))
            if len(found) >= limit:
                return found
    return found


def _format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Format recent messages (up to last 4) for context resolution."""
    if not history:
        return ""
    recent = history[-4:]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip().replace("\n", " ")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def resolve_followup_query(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Check if a question is a follow-up referring to previous conversation.
    If clear and standalone, avoids unnecessary LLM calls (Simple > Clever).
    """
    if not history or len(history) == 0:
        return question

    q_lower = question.strip().lower()

    # Fast check: pronouns or elliptical indicators requiring context
    followup_indicators = [
        "that", "this", "it", "they", "them", "their", "he", "she",
        "who can do that", "what about", "and who", "what if",
        "why", "how come", "does it", "can they", "is there any"
    ]
    is_followup = any(re.search(r"\b" + re.escape(ind) + r"\b", q_lower) for ind in followup_indicators)

    # If it's already a complete explicit sentence without pronoun dependency, keep it
    if not is_followup:
        return question

    # Use LLM to resolve pronoun reference into standalone search query
    llm = client or LLMClient()
    conv_text = _format_conversation_history(history)
    user_prompt = f"Recent Conversation:\n{conv_text}\n\nCurrent Question: {question}"

    try:
        res = llm.call_json(FOLLOWUP_RESOLVER_SYSTEM_PROMPT, user_prompt)
        resolved = res.get("resolved_query", "").strip()
        if resolved and len(resolved) > 3:
            print(f"[FOLLOW-UP RESOLVER] '{question}' -> '{resolved}'")
            return resolved
    except Exception as exc:
        print(f"[FOLLOW-UP RESOLVER] Resolution fallback: {exc}")

    return question


def build_agent_graph(index: Any, llm: Optional[LLMClient] = None, retrieval_mode: str = "hybrid"):
    """
    Build and compile the LangGraph StateGraph bound to the provided index and LLM client.
    """
    client = llm or LLMClient()

    # ── NODE 1: Retrieve Node ─────────────────────────────────────────────────
    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        query = state["current_query"]
        print(f"\n[RETRIEVE] Query: \"{query}\"")
        if _is_overview_query(state["query"]):
            chunks = _representative_chunks(index)
            pages = sorted({int(doc.metadata["page_num"]) for doc in chunks
                            if str(doc.metadata.get("page_num", "")).isdigit()})
            print(f"[RETRIEVE] Selected {len(chunks)} representative chunks for document overview.")
            return {"retrieved_chunks": chunks, "sub_queries": [query], "retrieved_pages": pages}
        route = route_query(query)
        queries = [query]
        if route.requires_multiple_hops:
            # Search each side of a comparison, plus the full question for links.
            queries = decompose_query(query)
        chunks = []
        seen = set()
        for subquery in queries:
            for doc in retrieve(query=subquery, index=index, k=route.top_k, mode=retrieval_mode):
                key = doc.metadata.get("chunk_id") or (doc.metadata.get("page_num"), doc.page_content)
                if key not in seen:
                    seen.add(key)
                    chunks.append(doc)
        chunks = chunks[:max(route.top_k, 5)]
        print(f"[RETRIEVE] Fetched {len(chunks)} chunks.")
        pages = sorted({int(doc.metadata["page_num"]) for doc in chunks
                        if str(doc.metadata.get("page_num", "")).isdigit()})
        return {"retrieved_chunks": chunks, "sub_queries": queries, "retrieved_pages": pages}

    # ── NODE 2: Grade Node ────────────────────────────────────────────────────
    def grade_node(state: AgentState) -> Dict[str, Any]:
        context_str = _format_context(state["retrieved_chunks"])
        user_prompt = f"Question: {state['query']}\n\nContext:\n{context_str}"
        response = client.call_json(GRADER_SYSTEM_PROMPT, user_prompt)

        sufficient = bool(response.get("sufficient", False))
        reason = response.get("reason", "")
        print(f"[GRADE] Sufficient: {'YES' if sufficient else 'NO'} | Reason: {reason}")
        return {"is_sufficient": sufficient, "grade_reason": reason}

    # ── CONDITIONAL ROUTER: Route after grade ────────────────────────────────
    def route_after_grade(state: AgentState) -> str:
        if state["is_sufficient"]:
            return "generate"
        if state["rewrite_count"] < 2:
            return "rewrite"
        # Reached 2 failed rewrites — stop loop and generate best-effort response
        print(f"[ROUTE] Maximum rewrites reached ({state['rewrite_count']}). Proceeding to generate low-confidence answer.")
        return "generate"

    # ── NODE 3: Rewrite Node ──────────────────────────────────────────────────
    def rewrite_node(state: AgentState) -> Dict[str, Any]:
        new_count = state["rewrite_count"] + 1
        user_prompt = (
            f"Question: {state['query']}\n"
            f"Previous Search: {state['current_query']}\n"
            f"Reason Previous Context Failed: {state['grade_reason']}\n"
        )
        response = client.call_json(REWRITER_SYSTEM_PROMPT, user_prompt)
        new_query = response.get("rewritten_query", state["query"])
        print(f"[REWRITE #{new_count}] Rewritten query: \"{new_query}\"")
        return {
            "current_query": new_query,
            "rewrite_count": new_count,
        }

    # ── NODE 4: Generate Node ─────────────────────────────────────────────────
    def generate_node(state: AgentState) -> Dict[str, Any]:
        context_str = _format_context(state["retrieved_chunks"])
        print("[GENERATE] Generating answer using retrieved context...")

        # If context was insufficient and we reached max rewrites, mark low confidence
        is_low_confidence = (not state["is_sufficient"]) and (state["rewrite_count"] >= 2)

        # Incorporate conversation context if available
        conv_context = _format_conversation_history(state.get("conversation_history", []))
        if conv_context:
            user_prompt = (
                f"Recent Conversation:\n{conv_context}\n\n"
                f"Question: {state['query']}\n\n"
                f"Context:\n{context_str}"
            )
        else:
            user_prompt = f"Question: {state['query']}\n\nContext:\n{context_str}"

        response = client.call_json(GENERATOR_SYSTEM_PROMPT, user_prompt)

        raw_answer = response.get("answer", "")
        citations = response.get("citations", [])

        # Validate citations: check that cited pages actually exist in retrieved docs
        retrieved_pages = {int(d.metadata["page_num"]) for d in state["retrieved_chunks"]
                           if str(d.metadata.get("page_num", "")).isdigit()}
        validated_citations = []
        for citation in citations if isinstance(citations, list) else []:
            if not isinstance(citation, dict):
                continue
            try:
                citation_page = int(citation.get("page"))
            except (TypeError, ValueError):
                continue
            if citation_page in retrieved_pages:
                citation["page"] = citation_page
                validated_citations.append(citation)
        for citation in validated_citations:
            page_docs = [doc for doc in state["retrieved_chunks"]
                         if doc.metadata.get("page_num") == citation.get("page")]
            quote_tokens = set(re.findall(r"\w+", (citation.get("text") or "").lower()))
            doc = max(page_docs, key=lambda candidate: len(
                quote_tokens & set(re.findall(r"\w+", candidate.page_content.lower()))
            ), default=None)
            if doc:
                geometry = _bbox_for_text(index, citation.get("page"), citation.get("text") or "")
                if geometry:
                    citation.update(geometry)
                else:
                    citation["bbox"] = None
                    citation["page_width"] = doc.metadata.get("page_width")
                    citation["page_height"] = doc.metadata.get("page_height")

        # Strip or flag citations in answer text if page does not exist
        # Check citations in text like [Page X]
        for cited_p in re.findall(r"\[Page\s+(\d+)\]", raw_answer):
            if int(cited_p) not in retrieved_pages:
                # Remove unverified citation tag
                raw_answer = raw_answer.replace(f"[Page {cited_p}]", "")

        confidence = "low" if is_low_confidence else "high"
        if is_low_confidence and not raw_answer.lower().startswith("low confidence"):
            raw_answer = f"[Low Confidence] I could not find enough information in the provided document to answer this confidently.\n\nClosest evidence:\n{raw_answer}"

        follow_ups = response.get("follow_up_suggestions", [])
        if not isinstance(follow_ups, list):
            follow_ups = []

        return {
            "answer": raw_answer.strip(),
            "citations": validated_citations,
            "confidence": confidence,
            "follow_up_suggestions": [q.strip() for q in follow_ups if isinstance(q, str) and q.strip()],
        }

    # ── NODE 5: Hallucination Check Node ──────────────────────────────────────
    def hallucination_check_node(state: AgentState) -> Dict[str, Any]:
        context_str = _format_context(state["retrieved_chunks"])
        user_prompt = f"Answer: {state['answer']}\n\nContext:\n{context_str}"
        response = client.call_json(HALLUCINATION_CHECK_SYSTEM_PROMPT, user_prompt)

        grounded = bool(response.get("grounded", True))
        unsupported = response.get("unsupported_claims", [])
        print(f"[GROUNDING CHECK] Grounded: {'YES' if grounded else 'NO'}")

        confidence = state["confidence"]
        if not grounded:
            confidence = "low"
            print(f"[GROUNDING CHECK] Found unsupported claims: {unsupported}")

        return {
            "is_grounded": grounded,
            "confidence": confidence,
            "unsupported_claims": unsupported,
        }

    # ── StateGraph Construction ──────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # Wire edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_edge("hallucination_check", END)

    return workflow.compile()


def answer(
    pdf_index: Any,
    question: str,
    llm_client: Optional[LLMClient] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    retrieval_mode: str = "hybrid",
) -> Dict[str, Any]:
    """
    Public entry point for Phase 3 & Final Chat:
    Executes the agentic reasoning loop over an existing DocumentIndex.

    Args:
        pdf_index:            DocumentIndex object produced by Phase 1/2 build_index().
        question:             Natural language query from user.
        llm_client:           Optional LLMClient instance (uses default if None).
        conversation_history: Optional recent chat history (e.g., last 4 messages).

    Returns:
        Dict containing:
            - query
            - retrieval_query
            - answer
            - citations
            - retrieved_chunks
            - is_grounded
            - confidence
            - rewrite_count
            - unsupported_claims
            - grade_reason
    """
    if not question or not question.strip():
        return {
            "query": "",
            "retrieval_query": "",
            "answer": "Please provide a non-empty question.",
            "citations": [],
            "retrieved_chunks": [],
            "is_grounded": True,
            "confidence": "low",
            "rewrite_count": 0,
            "unsupported_claims": [],
            "grade_reason": "",
        }

    client = llm_client or LLMClient()

    if client.provider == "fallback" and _is_full_document_extraction_query(question):
        extracted = _offline_full_document_answer(pdf_index, question)
        if extracted:
            return extracted

    # Step 1: Follow-up resolution (interprets pronouns like "that", "who can do it?" using recent messages)
    history = conversation_history or []
    retrieval_query = resolve_followup_query(question, history, client)

    graph = build_agent_graph(pdf_index, client, retrieval_mode=retrieval_mode)

    initial_state: AgentState = {
        "query": question,
        "current_query": retrieval_query,
        "conversation_history": history,
        "retrieved_chunks": [],
        "rewrite_count": 0,
        "is_sufficient": False,
        "grade_reason": "",
        "answer": "",
        "citations": [],
        "is_grounded": False,
        "confidence": "high",
        "unsupported_claims": [],
        "follow_up_suggestions": [],
        "sub_queries": [],
        "retrieved_pages": [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "query": final_state["query"],
        "retrieval_query": final_state["current_query"],
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "is_grounded": final_state["is_grounded"],
        "confidence": final_state["confidence"],
        "rewrite_count": final_state["rewrite_count"],
        "unsupported_claims": final_state["unsupported_claims"],
        "grade_reason": final_state.get("grade_reason", ""),
        "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
        "sub_queries": final_state.get("sub_queries", []),
        "retrieved_pages": final_state.get("retrieved_pages", []),
        "route_strategy": retrieval_mode if retrieval_mode != "hybrid" else route_query(retrieval_query).strategy,
    }
