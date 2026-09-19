"""
evaluate.py — Phase 4: Full Automated Ragas & Retrieval Evaluation Pipeline.

Usage:
    python evaluate.py

Outputs:
    - Console summary table (Ragas metrics + Hit@5 + Targets Comparison)
    - evaluation/results.csv  (Detailed row-by-row question evaluations)
    - evaluation/summary.json (Aggregated JSON metrics)
"""

import csv
import glob
import json
import os
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ingestion.index_builder import build_index
from app.retrieval.hybrid_retriever import retrieve
from app.reasoning.agent_graph import answer
from app.evaluation.ragas_evaluator import (
    calculate_faithfulness,
    calculate_context_precision,
    calculate_context_recall,
    calculate_answer_relevancy,
    is_correct_refusal,
)


def _load_pdf_bytes() -> bytes:
    """Load test contract PDF."""
    patterns = [
        os.path.join("tests", "data", "*.pdf"),
        "*.pdf",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0], "rb") as f:
                return f.read()
    raise FileNotFoundError("No evaluation PDF contract found.")


def run_pipeline_evaluation():
    print("\n" + "=" * 60)
    print("  DOCUMENT INTELLIGENCE ASSISTANT — Full System Evaluation")
    print("=" * 60)

    # 1. Load Dataset
    dataset_path = os.path.join("evaluation", "cuad_eval_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n[EVALUATION] Loaded {len(dataset)} evaluation questions.")
    in_domain_count = sum(1 for item in dataset if not item.get("is_out_of_domain", False))
    out_of_domain_count = sum(1 for item in dataset if item.get("is_out_of_domain", False))
    print(f"             In-domain contract clauses: {in_domain_count}")
    print(f"             Out-of-domain refusal cases: {out_of_domain_count}")

    # 2. Ingest Unique Contracts into In-Memory Index Cache
    print("\n[INDEXING] Ingesting contract documents into Ephemeral ChromaDB...")
    t0 = time.time()
    index_cache = {}

    def get_index_for_contract(contract_id: str):
        if contract_id in index_cache:
            return index_cache[contract_id]

        candidates = [
            os.path.join("tests", "data", contract_id),
            contract_id,
            os.path.join("tests", "data", os.path.basename(contract_id)),
            "Detailed_4Phase_Plan.pdf",
        ]
        chosen_path = next((p for p in candidates if os.path.exists(p)), None)
        if chosen_path:
            with open(chosen_path, "rb") as f:
                b = f.read()
        else:
            b = _load_pdf_bytes()

        idx = build_index(b)
        index_cache[contract_id] = idx
        return idx

    # Pre-warm indices for unique contracts
    unique_contracts = list({item.get("contract_id", "Detailed_4Phase_Plan.pdf") for item in dataset})
    for cid in unique_contracts:
        get_index_for_contract(cid)
    indexing_time = time.time() - t0
    print(f"[INDEXING] Completed indexing {len(unique_contracts)} contracts in {indexing_time:.2f}s.")

    # 3. Evaluate Questions
    detailed_rows = []
    dense_hits = 0
    bm25_hits = 0
    hybrid_hits = 0

    faithfulness_scores = []
    context_precision_scores = []
    context_recall_scores = []
    answer_relevancy_scores = []

    correct_refusals = 0

    print("\n[EVALUATION] Executing reasoning loop and computing metrics...")
    start_eval_time = time.time()

    for idx, item in enumerate(dataset, 1):
        q = item["question"]
        expected_ans = item["expected_answer"]
        gt_context = item.get("ground_truth_context", "")
        is_ood = item.get("is_out_of_domain", False)
        cid = item.get("contract_id", "cuad_contract_1.pdf")

        index = get_index_for_contract(cid)

        # Retrieval comparisons
        dense_results = index.vector_retriever.invoke(q)[:5]
        bm25_results = index.bm25_retriever.invoke(q)[:5]
        hybrid_results = retrieve(q, index, k=5)

        # Hit@5 evaluation (for in-domain questions)
        d_hit = 0
        b_hit = 0
        h_hit = 0
        if not is_ood and gt_context:
            gt_keywords = [w.lower() for w in gt_context.split() if len(w) >= 4][:3]
            d_hit = 1 if any(any(k in doc.page_content.lower() for k in gt_keywords) for doc in dense_results) else 0
            b_hit = 1 if any(any(k in doc.page_content.lower() for k in gt_keywords) for doc in bm25_results) else 0
            h_hit = 1 if any(any(k in doc.page_content.lower() for k in gt_keywords) for doc in hybrid_results) else 0

            dense_hits += d_hit
            bm25_hits += b_hit
            hybrid_hits += h_hit

        # Phase 3 agent reasoning
        ans_res = answer(index, q)
        gen_answer = ans_res["answer"]
        confidence = ans_res.get("confidence", "high")
        is_grounded = ans_res.get("is_grounded", True)
        unsupported = ans_res.get("unsupported_claims", [])
        rewrite_count = ans_res.get("rewrite_count", 0)

        # Compute Ragas metrics
        faithfulness = calculate_faithfulness(gen_answer, hybrid_results, is_grounded, unsupported)
        ctx_precision = calculate_context_precision(gt_context, hybrid_results, k=5)
        ctx_recall = calculate_context_recall(gt_context, hybrid_results)
        ans_relevancy = calculate_answer_relevancy(q, gen_answer)

        faithfulness_scores.append(faithfulness)
        context_precision_scores.append(ctx_precision)
        context_recall_scores.append(ctx_recall)
        answer_relevancy_scores.append(ans_relevancy)

        # Refusal check
        refusal_ok = is_correct_refusal(gen_answer, confidence, is_ood)
        if is_ood and refusal_ok:
            correct_refusals += 1

        detailed_rows.append({
            "id": item["id"],
            "question": q,
            "category": item.get("clause_category", ""),
            "is_out_of_domain": is_ood,
            "dense_hit": d_hit,
            "bm25_hit": b_hit,
            "hybrid_hit": h_hit,
            "rewrites": rewrite_count,
            "confidence": confidence,
            "faithfulness": faithfulness,
            "context_precision": ctx_precision,
            "context_recall": ctx_recall,
            "answer_relevancy": ans_relevancy,
            "answer_preview": gen_answer[:120].replace("\n", " "),
        })

    eval_duration = time.time() - start_eval_time

    # 4. Aggregations
    avg_dense_hit5 = round(dense_hits / max(1, in_domain_count), 3)
    avg_bm25_hit5 = round(bm25_hits / max(1, in_domain_count), 3)
    avg_hybrid_hit5 = round(hybrid_hits / max(1, in_domain_count), 3)

    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
    avg_precision = round(sum(context_precision_scores) / len(context_precision_scores), 3)
    avg_recall = round(sum(context_recall_scores) / len(context_recall_scores), 3)
    avg_relevancy = round(sum(answer_relevancy_scores) / len(answer_relevancy_scores), 3)

    refusal_accuracy = round(correct_refusals / max(1, out_of_domain_count), 3)

    # 5. Save Results Files
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(os.path.join("evaluation", "results"), exist_ok=True)

    csv_path = os.path.join("evaluation", "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_data = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(dataset),
        "in_domain_questions": in_domain_count,
        "out_of_domain_questions": out_of_domain_count,
        "indexing_time_seconds": round(indexing_time, 2),
        "evaluation_duration_seconds": round(eval_duration, 2),
        "metrics": {
            "dense_hit_at_5": avg_dense_hit5,
            "bm25_hit_at_5": avg_bm25_hit5,
            "hybrid_hit_at_5": avg_hybrid_hit5,
            "faithfulness": avg_faithfulness,
            "context_precision": avg_precision,
            "context_recall": avg_recall,
            "answer_relevancy": avg_relevancy,
            "refusal_accuracy": refusal_accuracy,
        },
        "target_comparison": {
            "faithfulness": {"target": ">0.85", "actual": avg_faithfulness, "status": "PASS" if avg_faithfulness >= 0.85 else "BELOW_TARGET"},
            "context_recall_at_5": {"target": ">0.75", "actual": avg_recall, "status": "PASS" if avg_recall >= 0.75 else "BELOW_TARGET"},
            "refusal_accuracy": {"target": ">0.90", "actual": refusal_accuracy, "status": "PASS" if refusal_accuracy >= 0.90 else "BELOW_TARGET"},
        }
    }

    json_path = os.path.join("evaluation", "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 6. Display Formatted Output
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Number of evaluation questions : {len(dataset)}")
    print(f"  Evaluation duration            : {eval_duration:.1f}s")
    print(f"  Results saved to               : {csv_path} and {json_path}")
    print("\n  RETRIEVAL METRICS (In-Domain Hit@5)")
    print("  " + "-" * 40)
    print(f"  Dense Hit@5                    : {avg_dense_hit5:.2f}")
    print(f"  BM25 Hit@5                     : {avg_bm25_hit5:.2f}")
    print(f"  Hybrid Hit@5                   : {avg_hybrid_hit5:.2f}")

    print("\n  RAGAS GENERATION & CONTEXT METRICS")
    print("  " + "-" * 40)
    print(f"  Faithfulness                   : {avg_faithfulness:.2f}")
    print(f"  Context Precision              : {avg_precision:.2f}")
    print(f"  Context Recall                 : {avg_recall:.2f}")
    print(f"  Answer Relevancy               : {avg_relevancy:.2f}")
    print(f"  Out-of-Domain Refusal Accuracy : {refusal_accuracy:.2f}")

    print("\n" + "=" * 60)
    print("  TARGET COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Target':<10} | {'Actual':<10} | {'Status'}")
    print("  " + "-" * 56)
    f_stat = "MET" if avg_faithfulness >= 0.85 else "BELOW"
    r_stat = "MET" if avg_recall >= 0.75 else "BELOW"
    ref_stat = "MET" if refusal_accuracy >= 0.90 else "BELOW"
    print(f"  {'Faithfulness':<25} | {'>0.85':<10} | {avg_faithfulness:<10.2f} | {f_stat}")
    print(f"  {'Context Recall@5':<25} | {'>0.75':<10} | {avg_recall:<10.2f} | {r_stat}")
    print(f"  {'Refusal Accuracy':<25} | {'>0.90':<10} | {refusal_accuracy:<10.2f} | {ref_stat}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline_evaluation()
