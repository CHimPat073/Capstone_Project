# Document Intelligence Assistant — Technical & Evaluation Report

**Project Type:** Ephemeral, Session-Free RAG System for Legal Contracts & Documents  
**Evaluation Dataset:** 45-item CUAD Benchmark Dataset (`evaluation/cuad_eval_dataset.json`) derived from [The Atticus Project CUAD Repository](https://github.com/The-Atticus-Project/cuad.git)  
**Evaluation Date:** 2026-09-19  

---

## 1. System Architecture

The project is structured into four distinct, loosely coupled layers:

1. **Phase 1: Ingestion & Ephemeral Indexing**
   - In-memory PDF parsing (`pypdf` / `pdfplumber`).
   - Table detection and extraction into Markdown tables (`camelot-py`).
   - Parent-child chunking via `RecursiveCharacterTextSplitter` (Parent: ~1000 chars, Child: ~800 chars, Overlap: 200 chars).
   - CPU-friendly local dense embedding generation (`sentence-transformers/all-MiniLM-L6-v2`).
   - In-memory ChromaDB vector store (`chromadb.EphemeralClient()`) and lexical `BM25Retriever`.
2. **Phase 2: Hybrid Retrieval & Fusion**
   - Dense retrieval candidate pool ($k=20$).
   - Sparse lexical retrieval candidate pool ($k=20$).
   - Rank fusion via LangChain `EnsembleRetriever` with Reciprocal Rank Fusion (RRF) and equal weighting (`[0.5, 0.5]`).
   - Standardized `retrieve(query, index, k=5)` entry point.
3. **Phase 3: Agentic Reasoning Loop**
   - Controlled state machine using LangGraph `StateGraph`.
   - Four dedicated nodes sharing a single direct OpenAI SDK client:
     - **Grader Node:** Evaluates whether retrieved context is sufficient to answer the question.
     - **Query Rewriter Node:** Rephrases vague/colloquial queries into legal terminology.
     - **Generator Node:** Answers strictly using context, outputting verifiable `[Page X]` citations.
     - **Hallucination Checker Node:** Compares answer claims against context to ensure factual grounding.
   - Bounded cycle: hard limit of **2 query rewrites** before generating an explicit low-confidence response.
4. **Phase 4: User Interface & Evaluation Guardrails**
   - Clean Streamlit web application (`streamlit run streamlit_app.py`).
   - Input guardrails: validates `%PDF` header, rejects empty files, and identifies scanned/image-only PDFs.
   - Evidence inspector expander showing source chunks, types, and page numbers.
   - Ephemeral session state: indices live in RAM and disappear upon session termination.

---

## 2. Technology Stack

| Component | Library / Model | Purpose |
|---|---|---|
| Document Parsing | `pypdf`, `pdfplumber` | Memory-based page extraction |
| Table Detection | `camelot-py` | Table boundary parsing to Markdown |
| Text Chunking | `langchain-text-splitters` | Parent/child recursive splitting |
| Embeddings | `all-MiniLM-L6-v2` | CPU-based 384-dimensional dense vectors |
| Vector Database | `chromadb` (EphemeralClient) | Purely in-memory dense vector store |
| Lexical Retrieval | `rank-bm25` | Sparse keyword index |
| Hybrid Fusion | `langchain-classic` | EnsembleRetriever with RRF |
| Agentic Workflow | `langgraph` (StateGraph) | Self-correcting loop control |
| Reasoning LLM | `openai` (GPT-4o-mini) | Grading, rewriting, generation, checking |
| User Interface | `streamlit` | Clean chat interface with source expanders |

---

## 3. Measured Evaluation Results (CUAD Benchmark)

Evaluated across **45 questions** (35 in-domain contract clause queries across commercial agreements from The Atticus Project + 10 out-of-domain refusal queries).

### Retrieval Performance (Hit Rate @ 5)

| Retrieval Strategy | Hit Rate @ 5 | Notes |
|---|---|---|
| **Dense Only (Chroma)** | **0.971** (34 / 35) | High semantic recall across complex legal clauses |
| **BM25 Only** | **0.829** (29 / 35) | Strong on exact keywords, but missed paraphrased legal queries |
| **Hybrid Ensemble (RRF)** | **0.971** (34 / 35) | Preserved the best of dense semantic coverage with lexical grounding |

### Ragas Generation & Context Metrics

| Metric | Measured Score | Target Threshold | Status | Analysis |
|---|---|---|---|---|
| **Faithfulness** | **1.000** | > 0.85 | **MET** | Grounding check and strict prompt rules eliminated hallucinations. |
| **Context Recall @ 5** | **0.788** | > 0.75 | **MET** | Fused top-5 chunks successfully captured key clause provisions across contracts. |
| **Context Precision** | **0.635** | — | — | Relevant chunks were ranked near the top of the 5 candidates. |
| **Answer Relevancy** | **0.739** | — | — | Concise answers directly addressed the legal questions without fluff. |
| **Refusal Accuracy** | **1.000** (10 / 10) | > 0.90 | **MET** | 100% of out-of-domain questions were correctly identified and refused with low confidence. |

---

## 4. Known Limitations

1. **Scanned / Rasterized PDFs:**  
   The pipeline relies on native PDF text extraction (`pypdf`/`pdfplumber`). Scanned/rasterized contract scans are rejected by guardrails because OCR (Tesseract) is intentionally excluded to keep the setup lightweight.
2. **Table Structure Complexity:**  
   Complex tables with merged multi-level column headers are converted to raw text/Markdown; deeply nested tabular relationships can lose formatting nuances.
3. **Single-Turn Focus:**  
   Each query executes an independent agentic reasoning loop over the ephemeral index; multi-turn follow-up coreference resolution is not currently modeled.

---

## 5. Potential Future Improvements

1. **Cross-Encoder Re-Ranking:** Introducing a lightweight re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the top 20 candidate pool before taking top 5.
2. **Context Window Expansion:** Dynamically fetching the full parent chunk when a child chunk is selected, rather than returning child chunk text alone.
3. **OCR Plug-in:** Adding optional fallback to `pytesseract` for scanned contracts when native text extraction yields zero characters.
