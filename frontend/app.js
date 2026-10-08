/**
 * app.js — Frontend controller for Document Intelligence Assistant.
 * Handles document uploads, chat communication, dynamic RAG loop status UI,
 * suggestion buttons, citations, and source rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("pdf-file");
  const fileLabel = document.getElementById("file-label-text");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadStatus = document.getElementById("upload-status");

  const docCard = document.getElementById("doc-card");
  const docName = document.getElementById("doc-name");
  const docSub = document.getElementById("doc-sub");
  const docMeta = document.getElementById("doc-meta");
  const metaPages = document.getElementById("meta-pages");
  const metaChunks = document.getElementById("meta-chunks");

  const chatHistory = document.getElementById("chat-history");
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");

  const suggestionsBar = document.getElementById("suggestions-bar");
  const suggestionsContainer = document.getElementById("suggestions-container");

  // Check initial backend status
  checkStatus();

  // File selection UI update
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileLabel.textContent = fileInput.files[0].name;
      uploadBtn.disabled = false;
    } else {
      fileLabel.textContent = "Choose PDF...";
      uploadBtn.disabled = true;
    }
  });

  // Handle PDF upload
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Ingesting PDF into ephemeral RAM...";
    uploadStatus.style.color = "#3b82f6";

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to upload PDF");
      }

      uploadStatus.textContent = "✓ Ingestion complete!";
      uploadStatus.style.color = "#10b981";

      // Clear chat history for fresh document
      chatHistory.innerHTML = "";

      // Update UI state
      updateDocState(data.filename, data.pages, data.chunks);
      enableChat();

      // Show starter suggestions
      showSuggestions([
        "What are the payment terms?",
        "What are the termination conditions?",
        "Who are the parties to this agreement?",
        "What is the governing law?",
      ]);
    } catch (err) {
      uploadStatus.textContent = `✗ ${err.message}`;
      uploadStatus.style.color = "#ef4444";
      uploadBtn.disabled = false;
    }
  });

  // Handle Chat Form Submit
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (!q) return;
    sendQuestion(q);
  });

  async function sendQuestion(question) {
    queryInput.value = "";
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Append user message
    appendMessage("user", question);

    // Create assistant placeholder with loading state
    const assistantMsgEl = createAssistantLoadingBubble();
    chatHistory.appendChild(assistantMsgEl);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Error querying assistant");
      }

      // Populate assistant bubble with complete response & loop UI
      populateAssistantResponse(assistantMsgEl, data);

      // Render suggestions if provided
      if (data.suggestions && data.suggestions.length > 0) {
        showSuggestions(data.suggestions);
      }
    } catch (err) {
      assistantMsgEl.querySelector(".bubble").textContent = `Error: ${err.message}`;
    } finally {
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  }

  function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function createAssistantLoadingBubble() {
    const msg = document.createElement("div");
    msg.className = "message assistant";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = "<em>Analyzing document with self-correcting RAG loop...</em>";
    msg.appendChild(bubble);
    return msg;
  }

  function populateAssistantResponse(msgEl, data) {
    const bubble = msgEl.querySelector(".bubble");
    bubble.innerHTML = "";

    // Card styling based on status (Part 11, 12, 13, 14)
    if (data.status === "out_of_domain") {
      bubble.classList.add("card-ood");
      addStatusMessage(bubble, "⚠ Outside Document Scope", data.answer);
    } else if (data.status === "needs_clarification") {
      bubble.classList.add("card-clarify");
      addStatusMessage(bubble, "? Clarification Needed", data.answer);
    } else if (data.status === "grounding_failed") {
      bubble.classList.add("card-ood");
      addStatusMessage(bubble, "⚠ Grounding Verification Failed", data.answer);
    } else {
      bubble.innerHTML = formatMarkdown(data.answer);
    }

    // Verified citations tag (Part 9, 15)
    if (data.citations && data.citations.length > 0) {
      const citTag = document.createElement("div");
      citTag.className = "citations-tag";
      const pages = data.citations.map((c) => `Page ${c.page}`).join(", ");
      citTag.textContent = `✓ Verified Citations: ${pages}`;
      bubble.appendChild(citTag);
      data.citations.filter((c) => Array.isArray(c.bbox) && c.bbox.length === 4).forEach((citation) => {
        const box = document.createElement("div");
        box.className = "citation-bbox";
        box.textContent = `Page ${citation.page} evidence area: x=${citation.bbox[0]}, y=${citation.bbox[1]}, width=${(citation.bbox[2] - citation.bbox[0]).toFixed(1)}, height=${(citation.bbox[3] - citation.bbox[1]).toFixed(1)} pt`;
        bubble.appendChild(box);
      });
    }

    if (data.evaluation_id) {
      const feedback = document.createElement("div");
      feedback.className = "human-feedback";
      const label = document.createElement("label");
      label.textContent = "Rate this answer (1–5): ";
      const rating = document.createElement("select");
      rating.setAttribute("aria-label", "Answer rating from 1 to 5");
      rating.innerHTML = '<option value="">Choose</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option>';
      const submit = document.createElement("button");
      submit.type = "button";
      submit.textContent = "Submit rating";
      const status = document.createElement("span");
      submit.addEventListener("click", async () => {
        if (!rating.value) return;
        submit.disabled = true;
        try {
          const response = await fetch("/api/evaluation/feedback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ evaluation_id: data.evaluation_id, rating: Number(rating.value) }),
          });
          if (!response.ok) throw new Error("Rating could not be saved");
          status.textContent = "Thanks for the rating.";
        } catch (error) { status.textContent = error.message; submit.disabled = false; }
      });
      label.appendChild(rating);
      feedback.append(label, submit, status);
      bubble.appendChild(feedback);
    }

    // Evaluation & Relevance Metrics Card (Part 16)
    if (data.eval_metrics) {
      const em = data.eval_metrics;
      const emCard = document.createElement("div");
      emCard.className = "eval-metrics-card";
      emCard.innerHTML = `
        <div class="eval-metrics-header">
          <span class="eval-metrics-title">Document Relevance & Quality Metrics</span>
          <span class="eval-score-badge">${em.relevance_level} (${Math.round(em.document_relevance_score * 100)}%)</span>
        </div>
        <div class="eval-metrics-grid">
          <div class="eval-metric-box">
            <span class="metric-name">Doc Relevance</span>
            <span class="metric-val">${Math.round(em.document_relevance_score * 100)}%</span>
          </div>
          <div class="eval-metric-box">
            <span class="metric-name">Context Alignment</span>
            <span class="metric-val">${Math.round(em.context_alignment * 100)}%</span>
          </div>
          <div class="eval-metric-box">
            <span class="metric-name">Faithfulness</span>
            <span class="metric-val">${Math.round(em.faithfulness * 100)}%</span>
          </div>
          <div class="eval-metric-box">
            <span class="metric-name">Answer Relevancy</span>
            <span class="metric-val">${Math.round(em.answer_relevancy * 100)}%</span>
          </div>
        </div>
      `;
      bubble.appendChild(emCard);
    }

    // RAG Process Visual Panel (Part 9)
    const ragCard = document.createElement("div");
    ragCard.className = "rag-process-card";
    ragCard.innerHTML = `
      <div class="rag-process-title">RAG REASONING PIPELINE</div>
      <div class="rag-steps">
        <span class="step-badge status-success">↗ ${data.loop.route_strategy || "semantic_hybrid"}</span>
        <span class="step-badge ${data.loop.query_relevant ? "status-success" : "status-error"}">
          ${data.loop.query_relevant ? "✓ Query Understood" : "✗ Query Unclear"}
        </span>
        <span class="step-badge ${data.status !== "out_of_domain" ? "status-success" : "status-error"}">
          ${data.status !== "out_of_domain" ? "✓ Relevant Context" : "✗ Out of Scope"}
        </span>
        <span class="step-badge ${data.loop.rewrites > 0 ? "status-warning" : "status-success"}">
          ${data.loop.rewrites > 0 ? `↻ Reformulated (${data.loop.rewrites} rewrites)` : "✓ Direct Match"}
        </span>
        <span class="step-badge ${data.loop.grounding_passed ? "status-success" : "status-error"}">
          ${data.loop.grounding_passed ? "✓ Grounding Verified" : "✗ Grounding Failed"}
        </span>
      </div>
    `;

    // Reformulation box if query was rewritten (Part 10)
    if (data.loop.rewrites > 0 && data.loop.resolved_query) {
      const refBox = document.createElement("div");
      refBox.className = "reformulate-box";
      const refTitle = document.createElement("strong");
      refTitle.textContent = "↻ Query Reformulated for Legal Retrieval:";
      refBox.append(refTitle, document.createElement("br"), document.createTextNode(`"${data.loop.resolved_query}"`));
      ragCard.appendChild(refBox);
    }

    bubble.appendChild(ragCard);

    // Expandable Sources List (Part 10, 15)
    if (data.sources && data.sources.length > 0) {
      const details = document.createElement("details");
      details.className = "sources-details";
      const summary = document.createElement("summary");
      summary.textContent = `Retrieved Evidence Sources (${data.sources.length} chunks)`;
      details.appendChild(summary);

      const list = document.createElement("div");
      list.className = "sources-list";
      data.sources.forEach((src, idx) => {
        const item = document.createElement("div");
        item.className = "source-item";
        const page = document.createElement("span");
        page.className = "source-page";
        page.textContent = `Source ${idx + 1} — Page ${src.page} (${src.chunk_type || "text"})`;
        const snippet = document.createElement("span");
        snippet.className = "source-snippet";
        snippet.textContent = `"${src.text}"`;
        item.append(page, snippet);
        list.appendChild(item);
      });

      details.appendChild(list);
      bubble.appendChild(details);
    }
  }

  function showSuggestions(suggestions) {
    if (!suggestions || suggestions.length === 0) {
      suggestionsBar.style.display = "none";
      return;
    }
    suggestionsContainer.innerHTML = "";
    suggestions.forEach((text) => {
      const pill = document.createElement("button");
      pill.className = "sug-pill";
      pill.textContent = text;
      pill.addEventListener("click", () => sendQuestion(text));
      suggestionsContainer.appendChild(pill);
    });
    suggestionsBar.style.display = "flex";
  }

  function updateDocState(filename, pages, chunks) {
    docCard.classList.remove("empty");
    docCard.classList.add("active");
    docCard.querySelector(".doc-icon").textContent = "✓";
    docName.textContent = filename;
    docSub.textContent = "Ready for querying";
    metaPages.textContent = pages;
    metaChunks.textContent = chunks;
    docMeta.style.display = "flex";
  }

  function enableChat() {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.placeholder = "Ask a question about the document...";
    queryInput.focus();
  }

  async function checkStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.document_loaded) {
        updateDocState(data.filename, data.pages, data.chunks);
        enableChat();
      }
    } catch (e) {
      console.log("Status check:", e);
    }
  }

  function formatMarkdown(text) {
    if (!text) return "";
    return text.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]))
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n- /g, "<br>• ")
      .replace(/\n/g, "<br>")
      .replace(/\[Page\s+(\d+)\]/g, '<strong style="color: #10b981;">[Page $1]</strong>');
  }

  function addStatusMessage(container, title, message) {
    const heading = document.createElement("strong");
    heading.textContent = title;
    const body = document.createElement("div");
    body.textContent = message || "";
    container.append(heading, document.createElement("br"), document.createElement("br"), body);
  }
});
