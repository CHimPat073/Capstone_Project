if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8000/");
}

document.addEventListener("DOMContentLoaded", () => {
  const $ = (selector) => document.querySelector(selector);
  const shell = $("#app-shell");
  const uploadForm = $("#upload-form");
  const fileInput = $("#pdf-file");
  const dropzone = $("#upload-dropzone");
  const fileLabel = $("#file-label-text");
  const uploadButton = $("#upload-btn");
  const uploadProgress = $("#upload-progress");
  const chatHistory = $("#chat-history");
  const queryInput = $("#query-input");
  const sendButton = $("#send-btn");
  const pageInput = $("#page-input");
  const pdfImage = $("#pdf-page-image");
  const pageWrap = $("#pdf-page-wrap");
  const bboxOverlay = $("#bbox-overlay");

  let pageCount = 0;
  let currentPage = 1;
  let currentZoom = 1;
  let currentCitation = null;
  let currentDocument = null;
  let showBoxes = true;
  let conversation = [];
  let pageText = "";
  let searchTimer = null;

  refreshStatus();
  checkHealth();

  fileInput.addEventListener("change", reflectSelectedFile);
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    if (name === "drop" || !dropzone.contains(event.relatedTarget)) dropzone.classList.remove("is-dragover");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    reflectSelectedFile();
  });
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  $("#new-document-button").addEventListener("click", () => fileInput.click());

  function reflectSelectedFile() {
    const file = fileInput.files[0];
    if (!file) {
      fileLabel.textContent = "Drop a PDF here";
      uploadButton.disabled = true;
      return;
    }
    fileLabel.textContent = file.name;
    const valid = (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) && file.size <= 20 * 1024 * 1024;
    uploadButton.disabled = !valid;
    $("#upload-status").textContent = valid ? "Ready to index" : "Choose a PDF smaller than 20 MB.";
  }

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file || uploadButton.disabled) return;
    uploadButton.disabled = true;
    uploadProgress.hidden = false;
    uploadProgress.classList.add("running");
    $("#upload-status").textContent = "Reading PDF and building the in-memory index…";
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not index this PDF.");
      conversation = [];
      chatHistory.replaceChildren();
      $("#assistant-onboarding").hidden = false;
      $("#upload-status").textContent = "Indexed in this session.";
      $("#upload-status").classList.add("success-text");
      await refreshStatus();
      await loadOutline();
      await goToPage(1);
      enableChat(true);
    } catch (error) {
      $("#upload-status").textContent = error.message;
      $("#upload-status").classList.add("error-text");
    } finally {
      uploadProgress.hidden = true;
      uploadProgress.classList.remove("running");
      uploadButton.disabled = !fileInput.files[0];
    }
  });

  async function refreshStatus() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      const status = await response.json();
      if (!status.document_loaded) return;
      currentDocument = status;
      pageCount = status.pages;
      $("#doc-card").classList.remove("empty");
      $("#doc-card").classList.add("active");
      $("#doc-name").textContent = status.filename;
      $("#doc-sub").textContent = "Ready to explore";
      $("#doc-status-badge").textContent = "Indexed";
      $("#doc-status-badge").classList.add("is-indexed");
      $("#doc-meta").hidden = false;
      $("#meta-pages").textContent = status.pages;
      $("#meta-chunks").textContent = status.chunks;
      $("#meta-size").textContent = formatBytes(status.file_size_bytes);
      $("#meta-uploaded").textContent = status.uploaded_at ? new Date(status.uploaded_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "This session";
      $("#pdf-title").textContent = status.filename;
      $("#chat-document-title").textContent = status.filename;
      $("#composer-document-state").textContent = `${status.pages} pages · in memory`;
      $("#index-copy").textContent = `${status.chunks} text segments indexed`;
      $("#index-card")?.classList.add("is-ready");
      $("#page-total").textContent = status.pages;
      pageInput.max = status.pages;
      pageInput.disabled = false;
      $("#pdf-empty-state").hidden = true;
      enableChat(true);
      if (!pageWrap.hidden) return;
      await loadOutline();
      await goToPage(1);
    } catch (error) {
      console.warn("Could not read document status", error);
    }
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error("offline");
      $("#health-label").textContent = "FastAPI connected";
      $("#health-dot").classList.add("connected");
    } catch (_) {
      $("#health-label").textContent = "Reconnecting…";
      $("#health-dot").classList.remove("connected");
    }
  }

  function enableChat(ready) {
    queryInput.disabled = !ready;
    sendButton.disabled = !ready;
    if (ready) queryInput.placeholder = "Ask a question or request a comparison…";
  }

  async function loadOutline() {
    const list = $("#outline-list");
    if (!currentDocument) return;
    try {
      const response = await fetch("/api/document/outline", { cache: "no-store" });
      const result = await response.json();
      list.replaceChildren();
      $("#outline-count").textContent = result.entries.length ? result.entries.length : "0";
      if (!result.entries.length) {
        const empty = document.createElement("p");
        empty.className = "quiet-empty";
        empty.textContent = "No PDF bookmarks in this file.";
        list.append(empty);
        return;
      }
      result.entries.forEach((entry) => {
        const button = document.createElement("button");
        button.className = "outline-entry";
        button.style.setProperty("--depth", Math.min(entry.depth, 3));
        button.textContent = entry.title;
        button.title = `Go to page ${entry.page}`;
        button.addEventListener("click", () => goToPage(entry.page));
        list.append(button);
      });
    } catch (_) {
      list.innerHTML = '<p class="quiet-empty">Contents unavailable for this file.</p>';
    }
  }

  async function goToPage(page, citation = currentCitation) {
    if (!pageCount) return;
    currentPage = Math.max(1, Math.min(pageCount, Number(page) || 1));
    pageInput.value = currentPage;
    bboxOverlay.replaceChildren();
    bboxOverlay.classList.remove("visible");
    pdfImage.alt = `Page ${currentPage} of ${pageCount}`;
    pdfImage.onload = citation && citation.page === currentPage ? () => showCitationBox(citation) : null;
    pdfImage.src = `/api/document/page/${currentPage}?t=${Date.now()}`;
    pageWrap.hidden = false;
    $("#pdf-empty-state").hidden = true;
    try {
      const response = await fetch(`/api/document/text/${currentPage}`, { cache: "no-store" });
      const data = await response.json();
      pageText = data.text || "";
      updateSearchCount();
    } catch (_) {
      pageText = "";
      updateSearchCount();
    }
    $("#pdf-scroll-region").scrollTop = 0;
  }

  function showCitationBox(citation) {
    bboxOverlay.replaceChildren();
    const bounds = citation?.bbox;
    if (showBoxes && Array.isArray(bounds) && bounds.length === 4 && citation.page_width && citation.page_height) {
      const widthRatio = Math.max(0, (bounds[2] - bounds[0]) / citation.page_width);
      const heightRatio = Math.max(0, (bounds[3] - bounds[1]) / citation.page_height);
      if (widthRatio > .96 || heightRatio > .55 || widthRatio * heightRatio > .35) return;
      const box = document.createElement("div");
      box.className = "citation-highlight";
      box.style.left = `${Math.max(0, bounds[0] / citation.page_width * 100)}%`;
      box.style.top = `${Math.max(0, bounds[1] / citation.page_height * 100)}%`;
      box.style.width = `${Math.min(100, (bounds[2] - bounds[0]) / citation.page_width * 100)}%`;
      box.style.height = `${Math.min(100, (bounds[3] - bounds[1]) / citation.page_height * 100)}%`;
      box.title = citation.text || `Evidence on page ${citation.page}`;
      bboxOverlay.append(box);
      bboxOverlay.classList.add("visible");
      requestAnimationFrame(() => {
        const region = $("#pdf-scroll-region");
        region.scrollTop = Math.max(0, box.offsetTop - region.clientHeight * .28);
      });
    }
  }

  pageInput.addEventListener("change", () => goToPage(pageInput.value));
  $("#page-prev").addEventListener("click", () => goToPage(currentPage - 1));
  $("#page-next").addEventListener("click", () => goToPage(currentPage + 1));
  $("#zoom-out").addEventListener("click", () => setZoom(currentZoom - .1));
  $("#zoom-in").addEventListener("click", () => setZoom(currentZoom + .1));
  $("#fit-width").addEventListener("click", () => setZoom(1));
  function setZoom(value) {
    currentZoom = Math.max(.5, Math.min(2, Math.round(value * 10) / 10));
    $("#zoom-value").textContent = `${Math.round(currentZoom * 100)}%`;
    pageWrap.style.width = `${currentZoom * 100}%`;
    pdfImage.style.maxWidth = "none";
  }
  $("#toggle-boxes").addEventListener("click", (event) => {
    showBoxes = !showBoxes;
    event.currentTarget.setAttribute("aria-pressed", String(showBoxes));
    event.currentTarget.textContent = showBoxes ? "Boxes on" : "Boxes off";
    event.currentTarget.classList.toggle("toggle-active", showBoxes);
    showCitationBox(currentCitation);
  });

  $("#pdf-search-input").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(updateSearchCount, 140);
  });
  function updateSearchCount() {
    const term = $("#pdf-search-input").value.trim();
    if (!term || !pageText) {
      $("#search-count").textContent = "0 / 0";
      return;
    }
    const matches = pageText.match(new RegExp(escapeRegExp(term), "gi")) || [];
    $("#search-count").textContent = `${matches.length ? 1 : 0} / ${matches.length}`;
  }

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = queryInput.value.trim();
    if (question) sendQuestion(question);
  });
  queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = `${Math.min(queryInput.scrollHeight, 120)}px`;
  });

  async function sendQuestion(question, targetMessage = null) {
    $("#assistant-onboarding").hidden = true;
    queryInput.value = "";
    queryInput.style.height = "auto";
    queryInput.disabled = true;
    sendButton.disabled = true;
    conversation.push({ role: "user", content: question });
    const userMessage = createMessage("user", question);
    chatHistory.append(userMessage);
    const assistantMessage = createMessage("assistant", "Searching the document…");
    assistantMessage.classList.add("is-loading");
    chatHistory.append(assistantMessage);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    if (targetMessage) assistantMessage.dataset.regenerates = targetMessage;
    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, retrieval_mode: $("#retrieval-mode").value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The question could not be answered.");
      assistantMessage.classList.remove("is-loading");
      assistantMessage.innerHTML = renderAnswer(data.answer, data.citations || []);
      renderReasoning(assistantMessage, data);
      renderSources(assistantMessage, data.sources || []);
      renderFeedback(assistantMessage, data, question);
      conversation.push({ role: "assistant", content: data.answer });
      renderSuggestions(data.suggestions || []);
      bindCitationClicks(assistantMessage, data.citations || []);
      if (data.citations?.length) selectCitation(data.citations[0]);
    } catch (error) {
      assistantMessage.classList.remove("is-loading");
      assistantMessage.textContent = error.message;
      assistantMessage.classList.add("message-error");
    } finally {
      queryInput.disabled = false;
      sendButton.disabled = false;
      queryInput.focus();
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  }

  function createMessage(role, text) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrapper.append(bubble);
    return wrapper;
  }

  function renderAnswer(answer, citations) {
    const escaped = escapeHtml(answer || "");
    return `<div class="bubble answer-copy">${escaped.replace(/\[Page\s+(\d+)\]/g, (_, page) => {
      const citation = citations.find((item) => Number(item.page) === Number(page));
      const label = citation?.text ? escapeHtml(citation.text.slice(0, 180)) : `Jump to page ${page}`;
      return `<button type="button" class="inline-citation" data-page="${Number(page)}" title="${label}">[p. ${Number(page)}]</button>`;
    }).replace(/\n/g, "<br>")}</div>`;
  }

  function renderReasoning(wrapper, data) {
    const loop = data.loop || {};
    const card = document.createElement("details");
    card.className = "reasoning-accordion";
    const summary = document.createElement("summary");
    summary.innerHTML = `Retrieval and reasoning details <span class="reasoning-route">${escapeHtml(loop.route_strategy || "hybrid")}</span>`;
    card.append(summary);
    const steps = document.createElement("ol");
    steps.className = "reasoning-steps";
    const queries = loop.sub_queries?.length ? loop.sub_queries : [loop.resolved_query || loop.original_query || "Your question"];
    queries.slice(0, 3).forEach((query, index) => addStep(steps, `Search ${index + 1}`, query));
    if (loop.retrieved_pages?.length) addStep(steps, "Evidence pages", loop.retrieved_pages.join(", "));
    if (loop.rewrites) addStep(steps, "Query refinements", `${loop.rewrites} rewrite${loop.rewrites === 1 ? "" : "s"}`);
    if (!loop.grounding_passed) addStep(steps, "Grounding", "Needs review");
    card.append(steps);
    wrapper.append(card);
  }

  function addStep(list, label, value) {
    const item = document.createElement("li");
    const key = document.createElement("span");
    key.textContent = label;
    const text = document.createElement("strong");
    text.textContent = value;
    item.append(key, text);
    list.append(item);
  }

  function renderSources(wrapper, sources) {
    if (!sources.length) return;
    const details = document.createElement("details");
    details.className = "source-drawer-inline";
    const summary = document.createElement("summary");
    summary.textContent = `Evidence returned · ${sources.length} passage${sources.length === 1 ? "" : "s"}`;
    const list = document.createElement("div");
    list.className = "source-list";
    sources.forEach((source) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "source-result";
      const page = document.createElement("span");
      page.className = "source-result-page";
      page.textContent = `p. ${source.page} · ${source.chunk_type || "text"}`;
      const quote = document.createElement("span");
      quote.textContent = source.text;
      button.append(page, quote);
      button.addEventListener("click", () => selectCitation(source));
      list.append(button);
    });
    details.append(summary, list);
    wrapper.append(details);
  }

  function renderFeedback(wrapper, data, question) {
    const bar = document.createElement("div");
    bar.className = "message-actions";
    const copy = actionButton("Copy answer", "Copy");
    copy.addEventListener("click", async () => {
      await navigator.clipboard?.writeText(data.answer || "");
      copy.textContent = "Copied";
    });
    const up = actionButton("Helpful", "Helpful");
    const down = actionButton("Not helpful", "Report issue");
    up.addEventListener("click", () => submitRating(up, data.evaluation_id, 5, true));
    down.addEventListener("click", () => submitRating(down, data.evaluation_id, 1, false));
    const regenerate = actionButton("Regenerate answer", "Regenerate");
    regenerate.addEventListener("click", () => sendQuestion(question));
    bar.append(copy, up, down, regenerate);
    wrapper.append(bar);
  }

  function actionButton(label, glyph) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-action";
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = glyph;
    return button;
  }

  async function submitRating(button, evaluationId, rating, helpful) {
    if (!evaluationId) return;
    try {
      const response = await fetch("/api/evaluation/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evaluation_id: evaluationId, rating, helpful }),
      });
      if (!response.ok) throw new Error("Could not save rating");
      button.classList.add("selected");
      button.title = "Rating saved";
    } catch (_) { button.title = "Rating could not be saved"; }
  }

  function bindCitationClicks(wrapper, citations) {
    wrapper.querySelectorAll(".inline-citation").forEach((button) => {
      const citation = citations.find((item) => Number(item.page) === Number(button.dataset.page));
      if (citation?.text) button.title = citation.text;
      button.addEventListener("click", () => citation && selectCitation(citation));
    });
  }

  async function selectCitation(citation) {
    currentCitation = citation;
    const page = Number(citation.page) || 1;
    await goToPage(page, citation);
    $("#excerpt-toggle").setAttribute("aria-expanded", "true");
    $("#excerpt-content").hidden = false;
    $("#excerpt-drawer").classList.add("expanded");
    $("#excerpt-page-label").textContent = `Page ${page}`;
    $("#excerpt-summary").textContent = `Page ${page} · selected evidence`;
    $("#excerpt-text").textContent = citation.text || "Passage text was not returned for this citation.";
    const box = citation.bbox;
    $("#excerpt-location").textContent = Array.isArray(box) ? `Highlighted on page ${page}` : `Source text from page ${page}`;
    if (window.matchMedia("(max-width: 960px)").matches) setActivePanel("pdf-pane");
  }

  function renderSuggestions(suggestions) {
    const bar = $("#suggestions-bar");
    const container = $("#followup-suggestions");
    container.replaceChildren();
    if (!suggestions.length) { bar.hidden = true; return; }
    suggestions.slice(0, 3).forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", () => sendQuestion(question));
      container.append(button);
    });
    bar.hidden = false;
  }

  $("#suggestions-container").addEventListener("click", (event) => {
    const button = event.target.closest("[data-question]");
    if (button && !queryInput.disabled) sendQuestion(button.dataset.question);
  });

  $("#excerpt-toggle").addEventListener("click", (event) => {
    const expanded = event.currentTarget.getAttribute("aria-expanded") !== "true";
    event.currentTarget.setAttribute("aria-expanded", String(expanded));
    $("#excerpt-content").hidden = !expanded;
    $("#excerpt-drawer").classList.toggle("expanded", expanded);
  });

  $("#clear-history").addEventListener("click", async () => {
    try { await fetch("/api/chat/history", { method: "DELETE" }); } catch (_) { /* local transcript still clears */ }
    conversation = [];
    chatHistory.replaceChildren();
    $("#assistant-onboarding").hidden = false;
    $("#suggestions-bar").hidden = true;
    $(".options-menu").open = false;
  });

  $("#export-chat").addEventListener("click", () => {
    if (!conversation.length) return;
    const text = conversation.map((message) => `${message.role === "user" ? "You" : "Folio"}: ${message.content}`).join("\n\n");
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "folio-conversation.txt";
    anchor.click();
    URL.revokeObjectURL(url);
    $(".options-menu").open = false;
  });

  $("#collapse-sidebar").addEventListener("click", () => {
    const collapsed = shell.classList.toggle("sidebar-collapsed");
    $("#collapse-sidebar").setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
  });
  $(".mobile-switcher").addEventListener("click", (event) => {
    const button = event.target.closest("[data-panel]");
    if (button) setActivePanel(button.dataset.panel);
  });
  function setActivePanel(panel) {
    shell.dataset.activePanel = panel;
    document.querySelectorAll(".mobile-switcher button").forEach((button) => button.classList.toggle("active", button.dataset.panel === panel));
  }

  setupResizeHandles();
  function setupResizeHandles() {
    const root = document.documentElement;
    const sizes = { sidebar: 272, chat: 430 };
    document.querySelectorAll(".resize-handle").forEach((handle) => {
      const kind = handle.dataset.resize;
      let startX = 0;
      let startWidth = 0;
      handle.addEventListener("pointerdown", (event) => {
        if (window.matchMedia("(max-width: 960px)").matches) return;
        handle.setPointerCapture(event.pointerId);
        startX = event.clientX;
        startWidth = sizes[kind];
        document.body.classList.add("is-resizing");
      });
      handle.addEventListener("pointermove", (event) => {
        if (!handle.hasPointerCapture(event.pointerId)) return;
        const delta = event.clientX - startX;
        const next = kind === "sidebar" ? startWidth + delta : startWidth - delta;
        sizes[kind] = Math.max(kind === "sidebar" ? 210 : 340, Math.min(kind === "sidebar" ? 360 : 560, next));
        root.style.setProperty(kind === "sidebar" ? "--sidebar-width" : "--chat-width", `${sizes[kind]}px`);
      });
      const stop = () => document.body.classList.remove("is-resizing");
      handle.addEventListener("pointerup", stop);
      handle.addEventListener("pointercancel", stop);
      handle.addEventListener("dblclick", () => {
        sizes[kind] = kind === "sidebar" ? 272 : 430;
        root.style.setProperty(kind === "sidebar" ? "--sidebar-width" : "--chat-width", `${sizes[kind]}px`);
      });
    });
  }

  function formatBytes(bytes) {
    if (!bytes) return "Not available";
    return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
  }
  function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
});
