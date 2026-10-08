# Document Intelligence Assistant

A portfolio/capstone project: an **ephemeral, session-free RAG system** for user-uploaded PDFs, evaluated on CUAD legal contracts.

No database. No persistent storage. Upload a PDF -> build an in-memory hybrid index in seconds -> self-correcting agentic reasoning loop -> grounded answer with page citations -> interactive Streamlit interface.

---

## Complete End-to-End Architecture

```
                               ┌─────────────────────────┐
                               │  User Uploads PDF File  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ PDF Validation & Guards │ (Rejects non-PDF, empty, scanned)
                               └────────────┬────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: INGESTION & EPHEMERAL INDEXING                                                │
│                                                                                        │
│   PDF Bytes ──► Page Extraction (pypdf/pdfplumber) + Table Detection (camelot-py)      │
│                     │                                                                  │
│                     ▼                                                                  │
│   Parent Chunks (~1000 chars) ──► Child Chunks (~800 chars, overlap 200)               │
│                                           │                                            │
│                     ┌─────────────────────┴─────────────────────┐                      │
│                     ▼                                           ▼                      │
│      ChromaDB (RAM only via EphemeralClient)              BM25 Keyword Index           │
└─────────────────────┬───────────────────────────────────────────┬──────────────────────┘
                      │                                           │
┌─────────────────────┼───────────────────────────────────────────┼──────────────────────┐
│ PHASE 2: HYBRID RETRIEVAL & FUSION                              │                      │
│                     ▼                                           ▼                      │
│             Dense Top-20 Chunks                         BM25 Top-20 Chunks             │
│                     │                                           │                      │
│                     └─────────────────────┬─────────────────────┘                      │
│                                           ▼                                            │
│                      LangChain EnsembleRetriever (RRF 0.5 / 0.5)                       │
│                                           │                                            │
│                                           ▼                                            │
│                               Top-5 Retrieved Contexts                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────┼────────────────────────────────────────────┐
│ PHASE 3: AGENTIC REASONING LOOP (LangGraph StateGraph)                                 │
│                                           ▼                                            │
│                               ┌──────────────────────┐                                 │
│                               │   Grader Node (LLM)  │                                 │
│                               │ Context sufficient?  │                                 │
│                               └───────────┬──────────┘                                 │
│                                           │                                            │
│                        ┌──────────────────┴──────────────────┐                         │
│                       YES                                    NO                        │
│                        │                                     │                         │
│                        │                           rewrite_count < 2?                  │
│                        │                              /              \                 │
│                        │                            YES              NO (Low Conf)     │
│                        │                             │                │                │
│                        ▼                             ▼                ▼                │
│             ┌──────────────────────┐        ┌────────────────┐ ┌─────────────────────┐ │
│             │    Generate Node     │        │  Rewrite Node  │ │  Generate Best-     │ │
│             │ Answer + [Page X]    │        │ Legal Term Exp │ │  Effort Low-Conf    │ │
│             └──────────┬───────────┘        └────────┬───────┘ └──────────┬──────────┘ │
│                        │                             │                    │            │
│                        │                             ▼                    │            │
│                        │                    Re-Retrieve (k=5)             │            │
│                        │                             │                    │            │
│                        │                             └────────────────────┤            │
│                        ▼                                                  ▼            │
│             ┌────────────────────────────────────────────────────────────────┐         │
│             │               Hallucination Checker Node (LLM)                 │         │
│             │          Verifies every factual claim against context          │         │
│             └────────────────────────────────┬───────────────────────────────┘         │
└──────────────────────────────────────────────┼─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: STREAMLIT UI & CITATION INSPECTION                                            │
│                                                                                        │
│   - Verified Answer with inline [Page X] citations                                     │
│   - Grounding and confidence indicator                                                 │
│   - Expandable source evidence inspector (page number, chunk type, preview)           │
│   - Complete memory clearance upon session reset                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Visualizations

- [Demo video (brag.mp4)](brag-output-2026-09-28-225024/brag.mp4)
- [Interactive runtime architecture flow (Archify)](.archify/architecture-document-intelligence-20261003-220841/document-intelligence.html)

---

## Technical Viva & Interview FAQ

### 1. Why ChromaDB?
Chroma provides a clean, standard vector database API while supporting a completely **ephemeral, in-memory mode** via `chromadb.EphemeralClient()`. This gives us production-grade vector similarity search in RAM with zero database files, zero `.sqlite` files, and zero disk residue.

### 2. Why BM25?
Dense embeddings represent semantic meaning well, but frequently fail on verbatim alphanumeric strings: clause identifiers (*e.g., Section 12.3(b)*), exact party names, specific financial amounts, and defined contract terms. BM25 catches these exact lexical tokens.

### 3. Why Hybrid Retrieval & EnsembleRetriever?
Dense retrieval provides high semantic recall; BM25 provides precision for specific phrases. LangChain's `EnsembleRetriever` merges both ranked candidate lists ($k=20$ from dense, $k=20$ from BM25) using **Reciprocal Rank Fusion (RRF)** with equal weighting `[0.5, 0.5]`.

### 4. Why LangGraph?
LangGraph is used strictly for **controlling the state machine workflow**. Because self-correcting RAG requires conditional branching (evaluating context sufficiency, deciding whether to rewrite, and looping back to retrieve), LangGraph makes these cyclic state transitions explicit and debuggable.
*Crucial distinction:* **LangGraph is NOT the LLM** — LangGraph controls the graph; the LLM performs the node tasks (grading, rewriting, answering, verification).

### 5. Why Ephemeral Storage?
Legal contracts are confidential. A session-free, ephemeral architecture ensures that user-uploaded documents exist solely in memory for the duration of the analysis. Once the session ends or is reset, the index object and collection are freed from memory without any persistent footprint.

### 6. Why CUAD?
The Contract Understanding Atticus Dataset (CUAD) is the industry-standard benchmark for legal document intelligence, featuring 510 contracts and 41 annotated clause types with complex layouts, nested tables, and cross-referencing.

### 7. How does Hallucination Checking work?
After answer generation, the answer text is passed alongside the retrieved context to the Hallucination Checker node. The LLM verifies whether each factual claim can be directly inferred from the retrieved excerpts. If any claim is ungrounded, the system flags the claim and downgrades confidence.

---

## Project Structure

```
capstone/
├── app/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # load_pdf()            — Memory-based PDF page extraction
│   │   ├── table_extractor.py  # extract_tables()       — camelot table detection to Markdown
│   │   ├── chunker.py          # create_chunks()        — parent/child recursive splitting
│   │   └── index_builder.py    # build_index()          — ephemeral Chroma + BM25
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py # build_hybrid_retriever(), retrieve(k=5)
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── llm_client.py       # Direct OpenAI SDK client & prompt templates
│   │   └── agent_graph.py      # LangGraph StateGraph & answer() entry point
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py    # Streamlit UI implementation & PDF validation
│   └── evaluation/
│       ├── __init__.py
│       └── ragas_evaluator.py  # Ragas metric algorithms (Faithfulness, Precision, Recall, Relevancy)
├── tests/
│   ├── test_phase1.py         # Phase 1 ingestion & indexing tests
│   ├── test_phase2.py         # Phase 2 hybrid retrieval tests
│   ├── test_phase3.py         # Phase 3 agentic loop & rewrite tests
│   └── test_phase4.py         # Phase 4 guardrails, ephemeral lifecycle, and UI tests
├── evaluation/
│   ├── cuad_eval_dataset.json  # 45-item evaluation dataset (in-domain + refusal cases)
│   ├── results.csv             # Detailed row-by-row evaluation output
│   └── summary.json            # Aggregated evaluation metrics
├── evaluate.py                 # Automated evaluation pipeline runner
├── streamlit_app.py            # Streamlit root launcher
├── DEMO.md                     # Step-by-step presentation script for demo/video
├── REPORT.md                   # Full technical report & benchmark findings
├── requirements.txt
└── README.md
```

---

## Quick Start & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The Python 3.11 environment is recommended. The local sentence embedding model is downloaded on first indexing. Cross-encoder reranking is optional and disabled by default; enable it with `RERANKER_ENABLED=true` (and optionally set `RERANKER_MODEL`).

### 2. Launch the Production FastAPI + HTML Frontend (All-in-One Server)
```bash
uvicorn backend.main:app --reload
```
Open `http://localhost:8000` to interact with the modern HTML/CSS/JS frontend connected to the FastAPI endpoints.

### 3. (Alternative) Launch the Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501`.

### 4. Run the Automated Evaluation Pipeline
```bash
python evaluate.py
```
Computes Hit@5, Faithfulness, Context Precision, Context Recall, Answer Relevancy, and Refusal Accuracy, saving `results.csv` and `summary.json`.

### 4. Run All Pytest Test Suites
```bash
pytest tests/ -v
```

### Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

The API and web UI are available at `http://localhost:8000`. Keep secrets in `.env`; do not commit it.

### Retrieval and reasoning options

- A local cross-encoder reranks hybrid retrieval candidates when `RERANKER_ENABLED=true`.
- The query router selects candidate depth and decomposes comparison questions into up to three searches. Results are merged before answer generation.
- PDF citations include page dimensions and an evidence bounding box in PDF points when pdfplumber can extract layout. These boxes are approximate word-match regions, not a rendered highlight overlay.
- `POST /api/evaluation/feedback` accepts `{ "evaluation_id": "…", "rating": 1, "helpful": false, "comment": "…" }`. `GET /api/evaluation/summary` reports in-memory aggregate ratings. Feedback is cleared when the server restarts.

### Optional QLoRA fine-tuning

QLoRA training requires a compatible CUDA GPU and is separate from normal app installation:

```bash
pip install -r requirements-training.txt
python scripts/train_qlora.py --dataset data/instructions.jsonl --model mistralai/Mistral-7B-Instruct-v0.3 --output artifacts/qlora-adapter
```

The JSONL dataset needs a `text` or `messages` field per row. Training writes an adapter and tokenizer files to the output directory; it does not replace the inference model automatically. Use data you have permission to train on and keep contract data private.

---

## Human-Friendly Document QA

The system does **not** simply retrieve text and mechanically display it, nor does it blindly copy-paste contract clauses. Instead, it executes an end-to-end grounded reasoning cycle:

1. **Follow-Up & Pronoun Resolution**:
   - If the user asks a follow-up question like *"And who can do that?"* or *"What if they don't give notice?"*, the conversational resolver disambiguates the pronouns using recent conversation history (`st.session_state.messages[-4:]`) into a standalone retrieval query without creating long-term databases.
2. **Context Sufficiency Grading**:
   - Evaluates whether the retrieved hybrid chunks contain the necessary factual evidence to answer.
3. **Adaptive Simplification & Synthesis**:
   - Explains complex legal and technical phrasing in natural, clear language while strictly preserving the legal obligations, rights, and conditions in the document.
   - **Human-friendly $\neq$ less accurate**: The generator explains conditions with structured bullets or concise explanations, directly answering the user's intent.
4. **Document Grounding & Anti-Hallucination**:
   - Every factual claim is verified against retrieved chunks. If the document does not contain enough information or leaves fields blank, the system explicitly refuses or indicates low confidence rather than fabricating facts.
5. **Exact Page Citations**:
   - Every factual assertion is linked to an exact `[Page X]` marker derived from document metadata.

---

## Evaluation Benchmark Summary

Evaluated across 45 questions on the contract document:

| Metric | Measured Score | Target | Status |
|---|---|---|---|
| **Faithfulness** | **1.00** | >0.85 | **MET** |
| **Context Recall@5** | **0.74** | >0.75 | **0.74 (Detailed in REPORT.md)** |
| **Refusal Accuracy** | **1.00** | >0.90 | **MET** |
| **Hybrid Hit@5** | **0.97** | Baseline | **MET** |
| **Dense Hit@5** | **0.94** | Baseline | **MET** |
| **BM25 Hit@5** | **0.97** | Baseline | **MET** |
