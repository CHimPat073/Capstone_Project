"""
streamlit_app.py — Phase 4: Clean Streamlit UI for Document Intelligence Assistant.

Features:
    - Sidebar: PDF upload with format validation, page count, and status badge.
    - Validation guardrails: Rejects non-PDF, empty files, and scanned PDFs with no text.
    - Ephemeral session state: Index lives only in st.session_state (zero disk persistence).
    - Chat interface: st.chat_message() and st.chat_input() calling Phase 3 answer().
    - Source evidence panel: st.expander() showing page, chunk type, and text preview.
    - Out-of-domain query handling: Explicit low-confidence indicator.
"""

import io
import os
import sys
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ingestion.index_builder import build_index
from app.reasoning.agent_graph import answer


def validate_pdf_bytes(pdf_bytes: bytes) -> tuple[bool, str]:
    """
    Validate uploaded file bytes for PDF header and extractable text.

    Returns:
        (is_valid, error_or_warning_message)
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        return False, "The uploaded file is empty. Please upload a valid PDF file."

    if not pdf_bytes.startswith(b"%PDF"):
        return False, "Please upload a valid PDF file. The file header is missing %PDF."

    # Verify if PDF has extractable text (guard against scanned/image-only PDFs)
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_text_len = sum(len(p.extract_text() or "") for p in reader.pages)
        if total_text_len < 10:
            return False, "This PDF does not contain extractable text. Scanned/image-only PDFs are not supported yet."
    except Exception as exc:
        return False, f"Could not parse the uploaded PDF: {exc}"

    return True, ""


def main():
    st.set_page_config(
        page_title="Document Intelligence Assistant",
        page_icon="📄",
        layout="wide",
    )

    # ── Header & Description ──────────────────────────────────────────────────
    st.title("📄 Document Intelligence Assistant")
    st.markdown(
        "Upload a contract or document and ask questions about its contents. "
        "Answers are **grounded** in the uploaded document and include **exact page citations**."
    )

    # ── Session State Initialization ─────────────────────────────────────────
    if "index" not in st.session_state:
        st.session_state.index = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None

    # ── Sidebar: Document Management ─────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        groq_key_input = st.text_input(
            "Groq API Key (Recommended)",
            type="password",
            value=os.environ.get("GROQ_API_KEY", "") if os.environ.get("GROQ_API_KEY") != "gsk_paste_your_groq_api_key_here" else "",
            help="Free, ultra-fast Llama 3.3 model from https://console.groq.com/keys",
        )
        if groq_key_input:
            os.environ["GROQ_API_KEY"] = groq_key_input

        openai_key_input = st.text_input(
            "OpenAI API Key (Alternative)",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", "") if os.environ.get("OPENAI_API_KEY") != "your_openai_api_key_here" else "",
            help="Optional. If left blank, uses Groq or local deterministic evaluation mode.",
        )
        if openai_key_input:
            os.environ["OPENAI_API_KEY"] = openai_key_input

        debug_mode = st.toggle("Debug Mode", value=False, help="Show retrieval queries, page routing, and grounding details.")

        st.divider()
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Contract PDF",
            type=["pdf"],
            help="Upload a searchable PDF document (contracts, agreements, plans).",
        )

        # Handle new file upload (resets chat and index to ensure document isolation)
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.current_file_name:
                pdf_bytes = uploaded_file.getvalue()
                is_valid, err_msg = validate_pdf_bytes(pdf_bytes)

                if not is_valid:
                    st.error(err_msg)
                    st.session_state.index = None
                    st.session_state.current_file_name = None
                else:
                    with st.spinner("Building ephemeral Chroma index..."):
                        try:
                            index = build_index(pdf_bytes)
                            st.session_state.index = index
                            st.session_state.current_file_name = uploaded_file.name
                            st.session_state.messages = []  # Clear previous document's conversation
                            st.success("Document ready.")
                        except Exception as exc:
                            st.error(f"Failed to index document: {exc}")
                            st.session_state.index = None

        # Display document metadata if index is active
        if st.session_state.index:
            st.divider()
            st.subheader("Active Document")
            st.write(f"**File:** `{st.session_state.current_file_name}`")
            st.write(f"**Pages:** {st.session_state.index.num_pages}")
            st.write(f"**Tables:** {st.session_state.index.num_tables}")
            st.write(f"**Searchable Chunks:** {len(st.session_state.index.child_chunks)}")
            st.info("Status: Ready (Ephemeral ChromaDB in RAM)")

            if st.button("Clear Document & Session", use_container_width=True):
                st.session_state.index = None
                st.session_state.messages = []
                st.session_state.current_file_name = None
                st.rerun()
        else:
            st.info("No active document. Please upload a PDF to begin.")

    # ── Main Chat Interface ───────────────────────────────────────────────────
    if not st.session_state.index:
        st.warning("Please upload a PDF document in the sidebar to start asking questions.")
        return

    # Render suggested starter questions if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions")
        cols = st.columns(3)
        suggestions = [
            "What are the payment terms?",
            "What are the termination conditions?",
            "What is the governing law?",
        ]
        chosen_prompt = None
        for i, prompt_text in enumerate(suggestions):
            if cols[i].button(prompt_text, key=f"sug_{i}", use_container_width=True):
                chosen_prompt = prompt_text
    else:
        chosen_prompt = None

    # Render past chat messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                citations_str = ", ".join(f"Page {c.get('page')}" for c in msg["citations"])
                st.caption(f"Verified Citations: **{citations_str}**")

            # Source evidence expander
            if msg.get("sources"):
                with st.expander(f"Retrieved Evidence Sources ({len(msg['sources'])} chunks)"):
                    for idx, src in enumerate(msg["sources"], 1):
                        p_num = src.metadata.get("page_num", "Unknown")
                        c_type = src.metadata.get("chunk_type", "text")
                        preview = src.page_content.strip().replace("\n", " ")[:250]
                        st.markdown(f"**Source {idx}** — Page {p_num} (`{c_type}`)")
                        st.markdown(f"> *\"{preview}...\"*")

            # Debug expander
            if debug_mode and msg.get("debug_info"):
                with st.expander("🛠️ Debug Info", expanded=False):
                    deb = msg["debug_info"]
                    st.write(f"**Original Question:** {deb.get('query')}")
                    st.write(f"**Resolved Retrieval Query:** {deb.get('retrieval_query')}")
                    st.write(f"**Rewrite Count:** {deb.get('rewrite_count')}")
                    st.write(f"**Confidence:** {deb.get('confidence')}")
                    st.write(f"**Grounded:** {deb.get('is_grounded')}")
                    st.write(f"**Grade Reason:** {deb.get('grade_reason')}")

            # Dynamic follow-up question suggestions from the assistant
            if msg.get("follow_ups"):
                st.markdown("💬 *Suggested follow-ups:*")
                f_cols = st.columns(len(msg["follow_ups"]))
                for f_idx, follow_q in enumerate(msg["follow_ups"]):
                    if f_cols[f_idx].button(f"👉 {follow_q}", key=f"fup_{msg_idx}_{f_idx}", use_container_width=True):
                        chosen_prompt = follow_q

    # Chat input
    user_input = st.chat_input("Ask a question about the uploaded document...") or chosen_prompt
    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Extract recent conversation history for pronoun / follow-up resolution (last 4 messages)
        recent_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ][-4:]

        # Generate response using Phase 3 / Phase 4 agentic loop
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document with self-correcting reasoning loop..."):
                try:
                    result = answer(
                        st.session_state.index,
                        user_input,
                        conversation_history=recent_history,
                    )

                    answer_text = result["answer"]
                    citations = result.get("citations", [])
                    confidence = result.get("confidence", "high")
                    is_grounded = result.get("is_grounded", True)
                    rewrite_count = result.get("rewrite_count", 0)
                    sources = result.get("retrieved_chunks", [])

                    # Display badge for confidence
                    if confidence == "low":
                        st.warning(f"Status: Low Confidence (Rewrites: {rewrite_count})")
                    else:
                        st.success(f"Status: Grounded & Verified (Rewrites: {rewrite_count})")

                    st.markdown(answer_text)

                    if citations:
                        citations_str = ", ".join(f"Page {c.get('page')}" for c in citations)
                        st.caption(f"Verified Citations: **{citations_str}**")

                    # Evidence sources panel
                    if sources:
                        with st.expander(f"Retrieved Evidence Sources ({len(sources)} chunks)"):
                            for idx, src in enumerate(sources, 1):
                                p_num = src.metadata.get("page_num", "Unknown")
                                c_type = src.metadata.get("chunk_type", "text")
                                preview = src.page_content.strip().replace("\n", " ")[:250]
                                st.markdown(f"**Source {idx}** — Page {p_num} (`{c_type}`)")
                                st.markdown(f"> *\"{preview}...\"*")

                    # Debug info panel if enabled
                    debug_info = {
                        "query": user_input,
                        "retrieval_query": result.get("retrieval_query", user_input),
                        "rewrite_count": rewrite_count,
                        "confidence": confidence,
                        "is_grounded": is_grounded,
                        "grade_reason": result.get("grade_reason", ""),
                    }
                    if debug_mode:
                        with st.expander("🛠️ Debug Info", expanded=False):
                            st.write(f"**Original Question:** {debug_info['query']}")
                            st.write(f"**Resolved Retrieval Query:** {debug_info['retrieval_query']}")
                            st.write(f"**Rewrite Count:** {debug_info['rewrite_count']}")
                            st.write(f"**Confidence:** {debug_info['confidence']}")
                            st.write(f"**Grounded:** {debug_info['is_grounded']}")
                            st.write(f"**Grade Reason:** {debug_info['grade_reason']}")

                    # Capture assistant follow-up questions
                    follow_ups = result.get("follow_up_suggestions", [])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                        "confidence": confidence,
                        "is_grounded": is_grounded,
                        "sources": sources,
                        "debug_info": debug_info,
                        "follow_ups": follow_ups,
                    })

                    st.rerun()

                except Exception as exc:
                    st.error(f"Error processing question: {exc}")


if __name__ == "__main__":
    main()
