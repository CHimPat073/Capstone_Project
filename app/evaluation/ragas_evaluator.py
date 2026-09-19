"""
ragas_evaluator.py — Phase 4: Automated Ragas & Retrieval Metric Evaluator.

Calculates standard Ragas metrics:
    - Faithfulness: Groundedness of generated answer in retrieved context.
    - Context Precision: Rank-weighted precision of relevant context chunks.
    - Context Recall: Extent to which ground truth is captured in retrieved context.
    - Answer Relevancy: Semantic alignment between user query and generated answer.
    - Hit@5: Dense, BM25, and Hybrid top-5 hit rate.
    - Refusal Accuracy: Proportion of out-of-domain questions correctly refused.
"""

import math
import re
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document


def calculate_faithfulness(
    answer: str,
    context_chunks: List[Document],
    is_grounded: bool = True,
    unsupported_claims: List[str] = None,
) -> float:
    """
    Measures proportion of answer claims directly grounded in retrieved context chunks.
    Evaluates factual content overlap of individual claims against context, penalizing
    unsupported assertions or failed grounding checks.
    """
    if not answer or not answer.strip():
        return 0.0

    if unsupported_claims is None:
        unsupported_claims = []

    # If no context chunks exist, there is no document context to be faithful to
    if not context_chunks:
        return 0.0

    # Low-confidence refusals or out-of-scope messages contain no document-grounded facts
    refusal_markers = [
        "outside the scope",
        "could not find enough information",
        "couldn't find enough information",
        "not mentioned in the provided",
        "cannot answer this question based on",
        "no document is currently loaded",
    ]
    if any(m in answer.lower() for m in refusal_markers):
        return 0.0

    if not is_grounded:
        return 0.25

    # Remove citation tags like [Page 1] before extracting claims so they don't distort textual overlap
    clean_answer = re.sub(r"\[Page \d+\]", "", answer, flags=re.IGNORECASE).strip()

    # Split into distinct sentences/claims
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean_answer) if len(s.strip()) > 8]
    if not sentences:
        sentences = [clean_answer]

    combined_context = " ".join(doc.page_content.lower() for doc in context_chunks)
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was",
        "were", "been", "have", "has", "had", "will", "would", "shall",
        "should", "under", "upon", "each", "such", "than", "more", "also",
        "into", "their", "there", "page", "pages", "clause", "section",
        "based", "according", "document", "provided", "contract", "agreement",
        "buyer", "seller", "party", "parties"
    }

    claim_scores = []
    for sent in sentences:
        # Extract meaningful alphanumeric content tokens (length >= 3)
        tokens = [
            w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", sent)
            if len(w) >= 3 and w.lower() not in stopwords
        ]
        # Also extract numeric figures/percentages as key contract entities
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", sent)
        all_check_tokens = list(set(tokens + [n.lower() for n in numbers]))

        if not all_check_tokens:
            continue

        supported_tokens = sum(1 for tok in all_check_tokens if tok in combined_context)
        claim_scores.append(supported_tokens / len(all_check_tokens))

    if not claim_scores:
        base_faith = 0.85
    else:
        # Average support across all individual claims
        base_faith = sum(claim_scores) / len(claim_scores)

    # Penalize if unsupported claims were flagged by LangGraph hallucination checker
    if unsupported_claims:
        penalty = min(0.6, len(unsupported_claims) * 0.2)
        base_faith = max(0.1, base_faith - penalty)

    return round(max(0.0, min(1.0, base_faith)), 2)


def calculate_context_recall(ground_truth: str, retrieved_chunks: List[Document]) -> float:
    """
    Measures whether key entities/information from the ground truth context were retrieved.
    Formula: ground_truth_terms_found / total_ground_truth_terms
    """
    if not ground_truth or not ground_truth.strip():
        return 1.0  # Out of domain has no ground truth to recall

    combined_context = " ".join(doc.page_content.lower() for doc in retrieved_chunks)
    gt_words = [
        w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", ground_truth)
        if len(w) >= 4 and w.lower() not in {"this", "that", "with", "from", "were", "have"}
    ]
    if not gt_words:
        return 1.0

    found = sum(1 for w in gt_words if w in combined_context)
    return round(min(1.0, found / len(gt_words)), 3)


def calculate_context_precision(ground_truth: str, retrieved_chunks: List[Document], k: int = 5) -> float:
    """
    Measures whether relevant chunks are ranked high up in top-k.
    Formula: Average Precision at K (Mean of P@i * v_i)
    """
    if not ground_truth or not ground_truth.strip() or not retrieved_chunks:
        return 1.0

    gt_words = {
        w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", ground_truth)
        if len(w) >= 4 and w.lower() not in {"this", "that", "with", "from"}
    }
    if not gt_words:
        return 1.0

    hits = []
    for doc in retrieved_chunks[:k]:
        doc_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", doc.page_content.lower()))
        overlap = len(gt_words.intersection(doc_words))
        is_relevant = 1 if overlap >= max(1, len(gt_words) * 0.3) else 0
        hits.append(is_relevant)

    total_relevant = sum(hits)
    if total_relevant == 0:
        return 0.0

    precisions = []
    running_hits = 0
    for rank, h in enumerate(hits, 1):
        if h == 1:
            running_hits += 1
            precisions.append(running_hits / rank)

    return round(sum(precisions) / total_relevant, 3)


def calculate_answer_relevancy(query: str, answer: str) -> float:
    """
    Measures how pertinent the generated answer is to the given question.
    Uses token overlap / semantic proxy between question and answer.
    """
    if not answer or not answer.strip():
        return 0.0

    # Refusals for out-of-scope queries have low relevancy to the specific question subject
    refusal_markers = [
        "outside the scope",
        "not mentioned in the provided",
        "cannot answer this question based on",
        "no document is currently loaded",
    ]
    if any(m in answer.lower() for m in refusal_markers):
        return 0.0

    if "could not find enough information" in answer.lower():
        return 0.20

    q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    a_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    stopwords = {
        "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
        "does", "have", "with", "from", "the", "and", "for", "are", "is", "about", "tell", "me"
    }
    meaningful_q = q_tokens - stopwords
    if not meaningful_q:
        return 0.85

    overlap = len(meaningful_q.intersection(a_tokens))
    ratio = overlap / len(meaningful_q)
    return round(min(1.0, 0.25 + 0.75 * ratio), 2)


def is_correct_refusal(answer: str, confidence: str, is_out_of_domain: bool) -> bool:
    """
    Determines if an out-of-domain question was correctly refused with low confidence.
    """
    if not is_out_of_domain:
        return True  # In-domain questions don't require refusal

    refusal_signals = [
        confidence == "low",
        "low confidence" in answer.lower(),
        "could not find enough information" in answer.lower(),
        "cannot find" in answer.lower(),
        "insufficient" in answer.lower(),
    ]
    return any(refusal_signals)
