# Demo Walkthrough Guide — Document Intelligence Assistant

This guide provides an exact, step-by-step walkthrough script for presenting the system during a viva, interview, or video demonstration.

---

## 1. Launch the Application

In your terminal, start the Streamlit UI:
```bash
streamlit run streamlit_app.py
```
The browser will automatically open to `http://localhost:8501`.

---

## 2. Step-by-Step Presentation Script

### Step 1: Explain the Architecture (Before Uploading)
- **What to say:**
  > *"This is the Document Intelligence Assistant. It is an ephemeral, session-free RAG system. Unlike traditional RAG with persistent databases, every document upload builds an in-memory hybrid index (ChromaDB + BM25) strictly in RAM that is completely discarded when the session ends."*
- **What to show on screen:**
  - Notice the header and the sidebar showing *"No active document. Please upload a PDF to begin."*

---

### Step 2: Upload a Contract & Show Ingestion (Phase 1 & 2)
- **What to do:**
  - In the sidebar file uploader, upload `Detailed_4Phase_Plan.pdf` (or any unseen contract PDF).
- **What to say:**
  > *"As soon as the PDF is uploaded, our validation guardrails verify file integrity and extractable text. In approximately 8-15 seconds, the ingestion pipeline extracts text pages, parses tables with Camelot, generates parent/child chunks, embeds them with MiniLM-L6-v2 on CPU, and indexes them into an in-memory ChromaDB and BM25 retriever."*
- **What to show on screen:**
  - Watch the spinner *"Building ephemeral Chroma index..."*
  - Watch the sidebar update with:
    - **Active Document:** `Detailed_4Phase_Plan.pdf`
    - **Pages:** 8
    - **Tables:** 5
    - **Searchable Chunks:** 38
    - **Status:** *Ready (Ephemeral ChromaDB in RAM)*

---

### Step 3: Ask First Contract Question (Direct Grounded Answer)
- **Prompt to type:**
  ```text
  What embedding model is used for dense vector indexing?
  ```
- **What to say:**
  > *"The query enters Phase 3's LangGraph agentic loop. First, it performs hybrid retrieval (top 20 Chroma + top 20 BM25 fused via EnsembleRetriever into top 5). The Grader node evaluates whether the context is sufficient. Here, it is sufficient, so it generates an answer with page citations and verifies grounding."*
- **What to show on screen:**
  - Status badge: `Status: Grounded & Verified (Rewrites: 0)`
  - Answer text showing exact model name.
  - Verified Citations: `Page 2`.
  - Click on **Retrieved Evidence Sources** expander to inspect the raw chunk text and page numbers.

---

### Step 4: Ask Second Contract Question (Table / Multi-Source Retrieval)
- **Prompt to type:**
  ```text
  Which tool is used for detecting and extracting tables from PDFs?
  ```
- **What to say:**
  > *"The system pinpoints the table extraction tool (camelot-py) directly from the retrieved document, noting that detected tables are preserved as single unsplit Markdown chunks."*
- **What to show on screen:**
  - Answer citing camelot-py.
  - Exact citation to the relevant page.
  - Expand source panel to demonstrate metadata preservation.

---

### Step 5: Ask an Out-of-Domain Question (Refusal Guardrail)
- **Prompt to type:**
  ```text
  Who won the 2022 FIFA World Cup?
  ```
- **What to say:**
  > *"Now observe the self-correcting guardrails. The Grader identifies that the retrieved context does not contain the answer. The Rewriter attempts to optimize the query twice. After hitting the bounded 2-rewrite limit, the system refuses to answer rather than hallucinating from external LLM knowledge."*
- **What to show on screen:**
  - Status badge: `Status: Low Confidence (Rewrites: 2)`
  - Answer: `"[Low Confidence] I could not find enough information in the provided document to answer this confidently."`
  - Demonstrates that external LLM knowledge is suppressed in favor of document grounding.

---

### Step 6: Session Cleanup & Ephemeral Verification
- **What to do:**
  - Click **Clear Document & Session** in the sidebar.
- **What to say:**
  > *"When the user clears or ends the session, the in-memory Chroma collection and index object are immediately freed from RAM. There are zero database files, zero SQLite traces, and zero persistent storage left on the host system."*
