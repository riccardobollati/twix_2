const stages = [
  {id: "clustering", label: "Page clustering", detail: "Inspect table & KV groups"},
  {id: "template", label: "Template inference", detail: "Learn the schema"},
  {id: "blocks", label: "Block separation", detail: "Inspect unseen pages"},
  {id: "sonar", label: "Missed sonar", detail: "Find uncovered pages"},
  {id: "record_separation", label: "Record separation", detail: "Inspect logical records"},
  {id: "extraction", label: "Data extraction", detail: "Build & train-evaluate"},
  {id: "results", label: "Final results", detail: "Test & compare"},
];

let activeRun = null;
let activeEventSource = null;
let workflow = {};
let selectedStage = null;
let followLive = true;
let selectedBlock = 0;
let selectedCluster = null;
let selectedRecord = 0;
let selectedSonarPage = null;
let hoveredSonarPage = null;
let sonarThreshold = null;
let sonarGeneration = 0;
let selectedResultTab = "output";
let selectedOutputView = "records";
const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}

function pretty(value, waiting) {
  return value == null ? waiting : JSON.stringify(value, null, 2);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast ${error ? "error" : ""}`;
  window.setTimeout(() => toast.classList.add("hidden"), 4200);
}

function statusLabel(status) {
  return ({queued:"Queued",running:"Running",awaiting_ground_truth:"Awaiting ground truth",completed:"Completed",failed:"Failed",interrupted:"Interrupted"}[status] || "Ready");
}

async function loadModelOptions() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("Could not load model options.");
  const data = await response.json();
  const operational = '<option value="">Environment default</option>' + data.models.map((model) => `<option value="${escapeHtml(model.value)}">${escapeHtml(model.label)}</option>`).join("");
  $("#easy-model").innerHTML = operational;
  $("#hard-model").innerHTML = operational;
  $("#single-model").innerHTML = operational;
  $("#reference-model").innerHTML = data.referenceModels.map((model) => `<option value="${escapeHtml(model.value)}">${escapeHtml(model.label)}</option>`).join("");
  $("#reference-model").value = "openai/gpt-5.4";
  $("#template-model").innerHTML = data.templateModels.map((model) => `<option value="${escapeHtml(model.value)}">${escapeHtml(model.label)}</option>`).join("");
  $("#template-model").value = "openai/gpt-5.4";
}

function updateRoutingControls() {
  const single = $("#no-routing").checked;
  $("#routed-models").classList.toggle("hidden", single);
  $("#single-model-wrap").classList.toggle("hidden", !single);
}

function artifactUrl(path, revision = 0) {
  if (!activeRun || !path) return "";
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return `/api/runs/${encodeURIComponent(activeRun.id)}/artifact/${encoded}${revision ? `?v=${revision}` : ""}`;
}

function currentStageIndex() {
  const index = stages.findIndex((stage) => stage.id === activeRun?.stage);
  return index < 0 ? 0 : index;
}

function renderStages() {
  const current = currentStageIndex();
  if (!selectedStage || followLive) selectedStage = stages[current].id;
  $("#stages").innerHTML = stages.map((stage, index) => {
    let state = index < current || activeRun?.status === "completed" ? "done" : "upcoming";
    if (index === current && ["running", "awaiting_ground_truth"].includes(activeRun?.status)) state = "active";
    if (index === current && ["failed", "interrupted"].includes(activeRun?.status)) state = "failed";
    const selected = selectedStage === stage.id ? "selected" : "";
    return `<button class="stage ${state} ${selected}" data-stage="${stage.id}" type="button"><span class="stage-dot">${state === "done" ? "✓" : index + 1}</span><span class="stage-copy"><b>${escapeHtml(stage.label)}</b><small>${escapeHtml(stage.detail)}</small></span></button>`;
  }).join("");
  document.querySelectorAll("[data-stage]").forEach((button) => button.addEventListener("click", () => {
    selectedStage = button.dataset.stage;
    followLive = selectedStage === stages[currentStageIndex()].id;
    renderStages();
    showStagePanel();
  }));
  $("#follow-live").classList.toggle("hidden", followLive);
  showStagePanel();
}

function renderModelAssignments() {
  const assignments = workflow.modelAssignments || [];
  const container = $("#model-assignments");
  const liveStage = activeRun?.stage || "clustering";
  const liveLabel = stages.find((stage) => stage.id === liveStage)?.label || "Pipeline";
  const activeIds = new Set(workflow.activeModelAssignments || []);
  const activeTasks = assignments.filter((assignment) => activeIds.has(assignment.id)).map((assignment) => assignment.task);
  $("#model-routing-status").textContent = activeRun?.status === "running" ? (activeTasks.join(" + ") || `${liveLabel} is active`) : "Effective assignments for this run";
  if (!assignments.length) {
    container.innerHTML = '<p class="model-assignment-empty">Resolving effective models…</p>';
    return;
  }
  container.innerHTML = assignments.map((assignment) => {
    const active = activeRun?.status === "running" && activeIds.has(assignment.id);
    const noModel = assignment.transport === "No model call";
    return `<article class="model-assignment ${active ? "active" : ""} ${assignment.detail ? "detail" : ""} ${noModel ? "local" : ""}">
      <div class="model-assignment-task"><span class="model-live-dot"></span><strong>${escapeHtml(assignment.task)}</strong><small>${escapeHtml(assignment.transport)}</small></div>
      <code title="${escapeHtml(assignment.model)}">${escapeHtml(assignment.model)}</code>
    </article>`;
  }).join("");
}

function showStagePanel() {
  document.querySelectorAll("[data-stage-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.stagePanel !== selectedStage));
}

function setStatus() {
  const badge = $("#run-status");
  badge.textContent = statusLabel(activeRun?.status);
  badge.className = `status-pill ${activeRun?.status || "idle"}`;
  $("#resume-run").classList.toggle("hidden", !activeRun?.canResume);
  $("#resume-run").textContent = activeRun?.resumePhase === "train_eval"
    ? "Resume training evaluation" : "Resume from block separation";
  $("#cancel-run").classList.toggle("hidden", activeRun?.status !== "running");
}

function renderLog() {
  const lines = activeRun?.logs || [];
  const log = $("#run-log");
  log.textContent = lines.length ? lines.join("\n") : "Waiting for pipeline output…";
  $("#log-count").textContent = lines.length ? `${lines.length} lines` : "";
  log.scrollTop = log.scrollHeight;
  const terminal = $("#terminal-output");
  terminal.textContent = lines.length ? lines.join("\n") : "Waiting for pipeline output…";
  $("#terminal-command").textContent = activeRun?.command || "Preparing the Python command…";
  terminal.scrollTop = terminal.scrollHeight;
}

function setPdf(frameSelector, emptySelector, path) {
  const frame = $(frameSelector);
  const empty = $(emptySelector);
  if (!path) {
    frame.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  const url = artifactUrl(path);
  if (frame.dataset.path !== path) { frame.src = url; frame.dataset.path = path; }
  frame.classList.remove("hidden");
  empty.classList.add("hidden");
}

function percent(value) {
  return value == null ? null : `${Math.round(Number(value) * 100)}%`;
}

function compactNumber(value) {
  if (value == null) return null;
  return new Intl.NumberFormat(undefined,{notation:"compact",maximumFractionDigits:1}).format(Number(value));
}

function showResultTab(tab) {
  selectedResultTab = tab;
  document.querySelectorAll("[data-result-tab]").forEach((button) => {
    const active = button.dataset.resultTab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll("[data-result-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.resultPanel !== tab));
}

function showOutputView(view) {
  selectedOutputView = view;
  document.querySelectorAll("[data-output-view]").forEach((button) => {
    const active = button.dataset.outputView === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $("#output-artifact-name").textContent = view === "records" ? "pipeline/grouped.json" : "pipeline/extracted.json";
  $("#extracted-json").textContent = pretty(
    view === "records" ? workflow.grouped : workflow.extracted,
    "Waiting for test extraction…",
  );
}

function pageRanges(pages) {
  const sorted = [...new Set(pages.map(Number))].sort((a, b) => a - b);
  const ranges = [];
  for (let i = 0; i < sorted.length;) {
    let end = i;
    while (end + 1 < sorted.length && sorted[end + 1] === sorted[end] + 1) end++;
    ranges.push(end === i ? String(sorted[i]) : `${sorted[i]}–${sorted[end]}`);
    i = end + 1;
  }
  return ranges.join(", ");
}

function renderClustering() {
  const selection = workflow.clusterSelection;
  const groups = workflow.clusterGroups || [];
  const finished = ["completed", "failed", "interrupted", "awaiting_ground_truth"].includes(activeRun?.status);
  const state = $("#cluster-state");
  state.textContent = selection ? "Selection ready" : groups.length ? "Clusters ready" : "Waiting";
  state.className = `artifact-state ${selection || groups.length ? "ready" : "waiting"}`;
  $("#cluster-selection").innerHTML = selection ? [
    `${selection.table_cluster_count} table clusters · ${selection.kv_cluster_count} KV clusters`,
    `Scanned ${selection.scanned_pages} pages from source page ${selection.source_start_page}`,
    `Ordering pages: ${pageRanges(selection.first_source_pages) || "none"}`,
    `Additional cluster samples: ${pageRanges(selection.cluster_sample_pages) || "none"}`,
    `All training pages: ${pageRanges(selection.train_source_pages)}`,
    `Training-page cap: ${selection.max_train_pages ?? "none"}`,
    `Uncovered clusters: ${(selection.uncovered_clusters || []).join(", ") || "none"}`,
    `Independent evaluation pages: ${pageRanges(selection.test_source_pages) || "none"}`,
  ].map((item) => `<span>${escapeHtml(item)}</span>`).join("")
    : finished && !groups.length ? "No saved clustering results for this run." : "Scanning table and KV layouts…";

  if (!groups.some((group) => group.id === selectedCluster)) selectedCluster = groups[0]?.id || null;
  const list = $("#cluster-list");
  if (!groups.length) {
    list.innerHTML = `<p class="muted">${selection ? "No table or KV page clusters were detected." : finished ? "No cluster groups are available for this run." : "Clusters will appear here when the scan finishes."}</p>`;
  } else {
    list.innerHTML = ["table", "kv"].map((kind) => {
      const matching = groups.filter((group) => group.kind === kind);
      if (!matching.length) return "";
      return `<h4 class="cluster-group-title">${kind === "table" ? "Table structures" : "KV patterns"}</h4>` + matching.map((group) => {
        const number = group.id.split(":")[1];
        const training = group.samplePage == null
          ? selection ? "Not included in training" : "Selection pending"
          : `Template sample: page ${group.samplePage}`;
        return `<button class="cluster-card ${selectedCluster === group.id ? "selected" : ""}" data-cluster-id="${escapeHtml(group.id)}" type="button"><strong>${kind === "table" ? "Table" : "KV"} cluster ${escapeHtml(number)}</strong><small>${group.pages.length} source page${group.pages.length === 1 ? "" : "s"} · representative page ${group.representative}</small><small>${escapeHtml(training)}</small></button>`;
      }).join("");
    }).join("");
    list.querySelectorAll("[data-cluster-id]").forEach((button) => button.addEventListener("click", () => {
      selectedCluster = button.dataset.clusterId;
      renderClustering();
    }));
  }

  const group = groups.find((item) => item.id === selectedCluster);
  $("#cluster-preview-title").textContent = group
    ? `${group.kind === "table" ? "Table" : "KV"} cluster ${group.id.split(":")[1]} · representative source page ${group.representative}`
    : "Representative overlay";
  $("#cluster-membership").textContent = group ? `Member source pages: ${pageRanges(group.pages)}` : "";
  const image = $("#cluster-image");
  const open = $("#cluster-open-image");
  if (group?.overlay) {
    const url = artifactUrl(group.overlay);
    if (image.dataset.path !== group.overlay) { image.src = url; image.dataset.path = group.overlay; }
    image.classList.remove("hidden");
    $("#cluster-preview-empty").classList.add("hidden");
    open.href = url;
    open.classList.remove("hidden");
  } else {
    image.classList.add("hidden");
    image.removeAttribute("src");
    image.dataset.path = "";
    open.classList.add("hidden");
    $("#cluster-preview-empty").textContent = group ? "Representative overlay is not available yet." : "Cluster overlays appear after the scan.";
    $("#cluster-preview-empty").classList.remove("hidden");
  }
}

function renderStatistics() {
  const stats = workflow.statistics;
  const images = workflow.statsImages || [];
  const summary = $("#stats-summary");
  const empty = $("#stats-empty");
  const gallery = $("#stats-gallery");
  if (!stats) {
    summary.innerHTML = "";
    gallery.innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }

  const totalTokens = Number(stats.input_tokens || 0) + Number(stats.output_tokens || 0);
  const metrics = [
    ["Test accuracy", percent(stats.test_accuracy)],
    ["Train accuracy", percent(stats.train_accuracy)],
    ["Break-even", stats.break_even_pages == null ? null : `${Math.round(stats.break_even_pages)} pages`],
    ["Block types", stats.block_types],
    ["Train / test pages", `${stats.train_pages ?? "—"} / ${stats.test_pages ?? "—"}`],
    ["Total tokens", compactNumber(totalTokens)],
    ["Est. generation cost", stats.generation_cost_usd == null ? null : `$${Number(stats.generation_cost_usd).toFixed(4)}`],
    ["Duration", stats.duration_s == null ? null : `${Math.round(stats.duration_s)}s`],
    ["Reference model", stats.reference_model],
    ["Template model", stats.template_model],
  ].filter(([,value]) => value !== null && value !== undefined);
  summary.innerHTML = metrics.map(([label,value]) => `<div class="stat-card"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join("");

  const labels = {"overview.png":"Accuracy, break-even & blocks","phase_cost.png":"Generation & fixing spend","token_breakdown.png":"Token breakdown by phase"};
  gallery.innerHTML = images.map((path) => { const name=path.split("/").pop(); return `<article class="stat-plot"><div class="viewer-title">${escapeHtml(labels[name] || name)}<a href="${artifactUrl(path)}" target="_blank" rel="noopener">Open</a></div><img src="${artifactUrl(path)}" alt="${escapeHtml(labels[name] || name)}" /></article>`; }).join("");
  empty.classList.toggle("hidden", images.length > 0);
}

function renderManualGroundTruth() {
  const manual = workflow.manualGroundTruth || {};
  const enabled = Boolean(manual.enabled);
  $("#ground-truth-tab").classList.toggle("hidden", !enabled);
  if (!enabled) {
    if (selectedResultTab === "ground-truth") selectedResultTab = "output";
    return;
  }
  $("#ground-truth-skill-prompt").textContent = manual.skillPrompt || "Use $twix-ground-truth for this run";
  const state = $("#manual-ground-truth-state");
  if (manual.awaiting) {
    state.textContent = "Ready for paste";
    state.className = "artifact-state warning";
  } else if (manual.model) {
    state.textContent = `Submitted · ${manual.model}`;
    state.className = "artifact-state ready";
  } else {
    state.textContent = "Waiting for test extraction";
    state.className = "artifact-state waiting";
  }
  $("#submit-ground-truth").disabled = !manual.awaiting;
  $("#ground-truth-json").disabled = !manual.awaiting;
  $("#ground-truth-model").disabled = !manual.awaiting;
  if (manual.model && !$("#ground-truth-model").value) $("#ground-truth-model").value = manual.model;
}

function renderWorkflow() {
  renderModelAssignments();
  renderClustering();
  setPdf("#template-pdf", "#template-pdf-empty", workflow.templatePdf);
  $("#template-json").textContent = pretty(workflow.template, "Waiting for template inference…");
  $("#copy-template").disabled = !workflow.template;
  const templateState = $("#template-state");
  templateState.textContent = workflow.template ? "Template ready" : "Waiting";
  templateState.className = `artifact-state ${workflow.template ? "ready" : "waiting"}`;

  renderBlocks();
  renderSonar();
  renderRecordSeparation();
  renderLog();
  const trainAccuracy = percent(workflow.trainEvaluation?.accuracy);
  const trainScore = $("#train-score");
  trainScore.textContent = trainAccuracy ? `Train accuracy ${trainAccuracy}` : "In progress";
  trainScore.className = `artifact-state ${trainAccuracy ? "ready" : "waiting"}`;

  setPdf("#result-pdf", "#result-pdf-empty", workflow.testPdf);
  showOutputView(selectedOutputView);
  const evaluation = workflow.evaluation;
  $("#result-score strong").textContent = percent(evaluation?.accuracy) || "—";
  const nodeScore = percent(evaluation?.components?.NodeMatchF1?.typed_f1);
  const contentScore = percent(evaluation?.components?.ContentScore?.overall);
  const details = [];
  if (nodeScore) details.push(`Node match ${nodeScore}`);
  if (contentScore) details.push(`Content ${contentScore}`);
  if (evaluation?.exact_match != null) details.push(evaluation.exact_match ? "Exact match" : "Differences found");
  $("#score-breakdown").innerHTML = (details.length ? details : ["Evaluation is waiting for the test extraction."]).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  renderManualGroundTruth();
  renderStatistics();
  showResultTab(selectedResultTab);
}

function sonarFlaggedPages() {
  const sonar = workflow.sonar;
  if (!sonar || sonar.averageCoverage == null) return [];
  if (sonar.averageCoverage === 0) return sonar.pages.filter((page) => page.totalWords > 0);
  const cutoff = sonar.averageCoverage * (1 - sonarThreshold / 100);
  return sonar.pages.filter((page) => page.coverage != null && page.coverage < cutoff);
}

function sonarX(page) {
  const pages = workflow.sonar?.pages || [];
  const index = pages.findIndex((row) => row.page === page);
  return index < 0 ? null : 55 + index * 13 + 5;
}

function highlightSonarPlot() {
  const line = $("#sonar-hover-line");
  if (!line) return;
  const x = sonarX(hoveredSonarPage ?? selectedSonarPage);
  line.style.display = x == null ? "none" : "block";
  if (x != null) { line.setAttribute("x1", x); line.setAttribute("x2", x); }
  if (hoveredSonarPage != null && x != null) {
    const plot = $("#sonar-plot");
    if (x < plot.scrollLeft + 30 || x > plot.scrollLeft + plot.clientWidth - 30) {
      plot.scrollLeft = Math.max(0, x - plot.clientWidth / 2);
    }
  }
}

function renderSonarPlot() {
  const sonar = workflow.sonar;
  const container = $("#sonar-plot");
  if (!sonar?.pages?.length) { container.innerHTML = ""; return; }
  const previousScroll = container.scrollLeft;
  const pages = sonar.pages;
  const width = Math.max(900, pages.length * 13 + 105);
  const top = 25, bottom = 252, chartHeight = bottom - top;
  const maxWords = Math.max(1, ...pages.map((page) => page.totalWords));
  const wordY = (count) => bottom - count / maxWords * chartHeight;
  const coverageY = (ratio) => bottom - ratio * chartHeight;
  const linePoints = pages.filter((page) => page.coverage != null).map((page) => `${sonarX(page.page)},${coverageY(page.coverage).toFixed(1)}`).join(" ");
  const cutoff = sonar.averageCoverage == null ? null : sonar.averageCoverage * (1 - sonarThreshold / 100);
  const bars = pages.map((page) => {
    const x = sonarX(page.page) - 5;
    const insideY = wordY(page.insideBlockWords);
    const outsideY = wordY(page.totalWords);
    const title = `Page ${page.page}: ${page.insideBlockWords} inside, ${page.outsideBlockWords} outside, ${page.coverage == null ? "no words" : Math.round(page.coverage * 100) + "% coverage"}`;
    return `<g class="sonar-bar" data-sonar-plot-page="${page.page}"><title>${escapeHtml(title)}</title><rect x="${x}" y="${insideY}" width="10" height="${bottom - insideY}" fill="#2563eb"/><rect x="${x}" y="${outsideY}" width="10" height="${insideY - outsideY}" fill="#cbd5e1"/><rect x="${x}" y="${top}" width="10" height="${chartHeight}" fill="transparent"/></g>`;
  }).join("");
  const ticks = pages.filter((_, index) => index === 0 || index === pages.length - 1 || index % Math.max(1, Math.ceil(pages.length / 12)) === 0).map((page) => `<text x="${sonarX(page.page)}" y="275" text-anchor="middle" fill="#5f6862" font-size="10">${page.page}</text>`).join("");
  container.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="310" viewBox="0 0 ${width} 310" aria-label="Inside and outside block word counts and coverage by page">
    <line x1="50" x2="${width - 20}" y1="${bottom}" y2="${bottom}" stroke="#aab2aa"/>
    <line x1="50" x2="${width - 20}" y1="${top}" y2="${top}" stroke="#e1e5e1"/>
    ${bars}
    ${sonar.averageCoverage == null ? "" : `<line x1="50" x2="${width - 20}" y1="${coverageY(sonar.averageCoverage)}" y2="${coverageY(sonar.averageCoverage)}" stroke="#a94f43" stroke-width="1" stroke-dasharray="5 4"/>`}
    ${cutoff == null ? "" : `<line x1="50" x2="${width - 20}" y1="${coverageY(cutoff)}" y2="${coverageY(cutoff)}" stroke="#d97706" stroke-width="1.5" stroke-dasharray="3 3"/>`}
    <polyline points="${linePoints}" fill="none" stroke="#dc2626" stroke-width="2" pointer-events="none"/>
    <line id="sonar-hover-line" x1="0" x2="0" y1="${top}" y2="${bottom}" stroke="#1b4d39" stroke-width="2" style="display:none" pointer-events="none"/>
    <text x="8" y="${top + 8}" fill="#5f6862" font-size="10">${maxWords} words</text>
    <text x="8" y="${bottom + 3}" fill="#5f6862" font-size="10">0</text>
    <text x="${width - 40}" y="${top + 8}" fill="#dc2626" font-size="10">100%</text>
    <text x="${width - 30}" y="${bottom + 3}" fill="#dc2626" font-size="10">0%</text>
    ${ticks}
    <text x="${width / 2}" y="299" text-anchor="middle" fill="#5f6862" font-size="11">Source PDF page · blue: inside · gray: outside · red: coverage · dashed: average and threshold</text>
  </svg>`;
  container.scrollLeft = previousScroll;
  document.querySelectorAll("[data-sonar-plot-page]").forEach((item) => item.addEventListener("click", () => selectSonarPage(Number(item.dataset.sonarPlotPage))));
  highlightSonarPlot();
}

function selectSonarPage(page) {
  selectedSonarPage = page;
  const row = workflow.sonar?.pages?.find((item) => item.page === page);
  if (!row) return;
  document.querySelectorAll("[data-sonar-list-page]").forEach((button) => button.classList.toggle("selected", Number(button.dataset.sonarListPage) === page));
  $("#sonar-preview-title").textContent = `Block overlay · source page ${page}`;
  $("#sonar-preview-coverage").textContent = row.coverage == null ? "No words" : `${Math.round(row.coverage * 100)}% covered`;
  const image = $("#sonar-preview-image");
  const empty = $("#sonar-preview-empty");
  image.classList.add("hidden"); empty.classList.remove("hidden"); empty.textContent = "Rendering page overlay…";
  const runId = activeRun?.id;
  image.onload = () => { if (activeRun?.id === runId && selectedSonarPage === page) { image.classList.remove("hidden"); empty.classList.add("hidden"); } };
  image.onerror = () => { if (selectedSonarPage === page) empty.textContent = "Could not render this page."; };
  image.src = `/api/runs/${encodeURIComponent(runId)}/sonar/page/${page}?scan=${sonarGeneration}`;
  highlightSonarPlot();
}

function renderSonar() {
  const sonar = workflow.sonar;
  const state = $("#sonar-state");
  if (!sonar) {
    const available = Boolean(workflow.sonarAvailable);
    if (sonarThreshold == null) sonarThreshold = Number(activeRun?.config?.sonarDropPercent ?? 15);
    $("#sonar-threshold-slider").value = sonarThreshold;
    $("#sonar-threshold-number").value = sonarThreshold;
    $("#sonar-run-existing").classList.toggle("hidden", !available);
    $("#sonar-choose-range").classList.add("hidden");
    $("#sonar-range-controls").classList.add("hidden");
    state.textContent = available ? "Ready to scan" : workflow.sonarError?.error ? "Scan failed" : "Waiting";
    state.className = `artifact-state ${available || workflow.sonarError?.error ? "warning" : "waiting"}`;
    $("#sonar-description").textContent = workflow.sonarError?.error || (available
      ? `Run the saved block separator on up to 100 source pages starting at page ${workflow.sonarStartPage}. No model calls are needed.`
      : "Sonar will scan up to 100 pages after block separation.");
    $("#sonar-plot").innerHTML = ""; $("#sonar-page-list").innerHTML = '<p class="sonar-list-empty">Waiting for scan…</p>';
    $("#sonar-preview-image").classList.add("hidden");
    $("#sonar-preview-empty").classList.remove("hidden");
    $("#sonar-preview-empty").textContent = "Select a flagged page to inspect its blocks.";
    return;
  }
  $("#sonar-run-existing").classList.add("hidden");
  $("#sonar-choose-range").classList.remove("hidden");
  if (sonarThreshold == null) sonarThreshold = Number(sonar.thresholdPercent ?? activeRun?.config?.sonarDropPercent ?? 15);
  $("#sonar-threshold-slider").value = sonarThreshold;
  $("#sonar-threshold-number").value = sonarThreshold;
  $("#sonar-range-start").max = sonar.totalPages;
  $("#sonar-range-end").max = sonar.totalPages;
  $("#sonar-page-max").textContent = `Available source pages: 1–${sonar.totalPages}`;
  state.textContent = `Pages ${sonar.startPage}–${sonar.endPage}`;
  state.className = "artifact-state ready";
  const flagged = sonarFlaggedPages();
  const average = sonar.averageCoverage == null ? "—" : `${Math.round(sonar.averageCoverage * 100)}%`;
  const cutoff = sonar.averageCoverage == null ? "—" : `${Math.round(sonar.averageCoverage * (1 - sonarThreshold / 100) * 100)}%`;
  $("#sonar-description").textContent = sonar.averageCoverage === 0
    ? `Scanned source pages ${sonar.startPage}–${sonar.endPage} of ${sonar.totalPages}. No words were assigned to blocks; all ${flagged.length} nonblank pages are flagged.`
    : `Scanned source pages ${sonar.startPage}–${sonar.endPage} of ${sonar.totalPages}. Average coverage ${average}; cutoff ${cutoff} (${sonarThreshold}% relative drop). ${flagged.length} page${flagged.length === 1 ? "" : "s"} flagged.`;
  $("#sonar-flag-count").textContent = flagged.length;
  if (!flagged.some((row) => row.page === selectedSonarPage)) selectedSonarPage = flagged[0]?.page ?? null;
  $("#sonar-page-list").innerHTML = flagged.length ? flagged.map((row) => {
    const drop = sonar.averageCoverage > 0 ? Math.round((1 - row.coverage / sonar.averageCoverage) * 100) : 100;
    return `<button type="button" class="sonar-page-button ${selectedSonarPage === row.page ? "selected" : ""}" data-sonar-list-page="${row.page}"><span>Page ${row.page}</span><small>${Math.round(row.coverage * 100)}% · ↓${drop}%</small></button>`;
  }).join("") : '<p class="sonar-list-empty">No pages meet this threshold.</p>';
  document.querySelectorAll("[data-sonar-list-page]").forEach((button) => {
    const page = Number(button.dataset.sonarListPage);
    button.addEventListener("click", () => selectSonarPage(page));
    button.addEventListener("mouseenter", () => { hoveredSonarPage = page; highlightSonarPlot(); });
    button.addEventListener("mouseleave", () => { hoveredSonarPage = null; highlightSonarPlot(); });
    button.addEventListener("focus", () => { hoveredSonarPage = page; highlightSonarPlot(); });
    button.addEventListener("blur", () => { hoveredSonarPage = null; highlightSonarPlot(); });
  });
  renderSonarPlot();
  if (selectedSonarPage == null) {
    $("#sonar-preview-image").classList.add("hidden");
    $("#sonar-preview-empty").classList.remove("hidden");
    $("#sonar-preview-empty").textContent = "No flagged page to preview.";
  } else if (!$("#sonar-preview-image").src.includes(`/sonar/page/${selectedSonarPage}?scan=${sonarGeneration}`)) {
    selectSonarPage(selectedSonarPage);
  }
}

function renderBlocks() {
  const images = workflow.blockImages || [];
  selectedBlock = Math.min(selectedBlock, Math.max(images.length - 1, 0));
  const source = $("#block-source");
  source.textContent = images.length ? `${workflow.blockSource === "test" ? "Evaluation test" : "Train"} · ${images.length} pages` : "Waiting";
  source.className = `artifact-state ${images.length ? (workflow.blockSource === "test" ? "ready" : "warning") : "waiting"}`;
  $("#block-empty").classList.toggle("hidden", images.length > 0);
  $("#block-image").classList.toggle("hidden", images.length === 0);
  if (!images.length) { $("#block-thumbnails").innerHTML = ""; $("#block-counter").textContent = ""; return; }
  $("#block-image").src = artifactUrl(images[selectedBlock], workflow.blockRevision);
  $("#block-page-title").textContent = `Block overlay · page ${selectedBlock + 1}`;
  $("#block-counter").textContent = `${selectedBlock + 1} / ${images.length}`;
  $("#block-thumbnails").innerHTML = images.map((path, index) => `<button class="block-thumb ${index === selectedBlock ? "selected" : ""}" data-block-index="${index}" type="button"><img src="${artifactUrl(path, workflow.blockRevision)}" alt="Block overlay page ${index + 1}" loading="lazy" />Page ${index + 1}</button>`).join("");
  document.querySelectorAll("[data-block-index]").forEach((button) => button.addEventListener("click", () => { selectedBlock = Number(button.dataset.blockIndex); renderBlocks(); }));
}

function renderRecordSeparation() {
  const images = workflow.recordImages || [];
  selectedRecord = Math.min(selectedRecord, Math.max(images.length - 1, 0));
  const source = $("#record-source");
  source.textContent = images.length ? `${workflow.recordSource === "test" ? "Evaluation test" : "Train"} · ${images.length} pages` : "Waiting";
  source.className = `artifact-state ${images.length ? (workflow.recordSource === "test" ? "ready" : "warning") : "waiting"}`;
  $("#record-empty").classList.toggle("hidden", images.length > 0);
  $("#record-image").classList.toggle("hidden", images.length === 0);
  if (!images.length) { $("#record-thumbnails").innerHTML = ""; $("#record-counter").textContent = ""; return; }
  $("#record-image").src = artifactUrl(images[selectedRecord], workflow.recordRevision);
  $("#record-page-title").textContent = `Record overlay · page ${selectedRecord + 1}`;
  $("#record-counter").textContent = `${selectedRecord + 1} / ${images.length}`;
  $("#record-thumbnails").innerHTML = images.map((path, index) => `<button class="block-thumb ${index === selectedRecord ? "selected" : ""}" data-record-index="${index}" type="button"><img src="${artifactUrl(path, workflow.recordRevision)}" alt="Record overlay page ${index + 1}" loading="lazy" />Page ${index + 1}</button>`).join("");
  document.querySelectorAll("[data-record-index]").forEach((button) => button.addEventListener("click", () => { selectedRecord = Number(button.dataset.recordIndex); renderRecordSeparation(); }));
}

function activateRun(run) {
  const previousStage = activeRun?.stage;
  activeRun = {...activeRun, ...run};
  if (followLive && previousStage !== activeRun.stage) selectedStage = activeRun.stage;
  $("#run-title").textContent = activeRun.filename || "Untitled run";
  $("#upload-panel").classList.add("hidden");
  $("#dashboard").classList.remove("hidden");
  $("#show-logs").classList.remove("hidden");
  setStatus(); renderStages(); renderLog(); renderRunList();
}

async function loadRuns(selectNewest = false) {
  const response = await fetch("/api/runs");
  const {runs} = await response.json();
  window.runs = runs;
  if ((!activeRun || selectNewest) && runs.length) await selectRun(runs[0].id);
  else renderRunList();
}

function renderRunList() {
  const runs = window.runs || [];
  $("#run-list").innerHTML = runs.length ? runs.map((run) => `<button class="run-item ${activeRun?.id === run.id ? "selected" : ""}" data-run="${escapeHtml(run.id)}" type="button"><span class="run-state ${escapeHtml(run.status)}"></span><span><b>${escapeHtml(run.filename)}</b><small>${escapeHtml(statusLabel(run.status))}</small></span></button>`).join("") : '<p class="empty-runs">No runs yet.</p>';
  document.querySelectorAll("[data-run]").forEach((button) => button.addEventListener("click", () => selectRun(button.dataset.run)));
}

async function loadWorkflow() {
  if (!activeRun) return;
  const response = await fetch(`/api/runs/${encodeURIComponent(activeRun.id)}/workflow`);
  if (!response.ok) return;
  workflow = await response.json();
  renderWorkflow();
}

async function selectRun(runId) {
  if (activeEventSource) activeEventSource.close();
  activeEventSource = null; workflow = {}; selectedBlock = 0; selectedCluster = null; selectedRecord = 0; selectedSonarPage = null; hoveredSonarPage = null; sonarThreshold = null; sonarGeneration = 0; followLive = true; selectedResultTab = "output";
  $("#ground-truth-json").value = "";
  $("#ground-truth-model").value = "";
  $("#sonar-preview-image").removeAttribute("src");
  $("#sonar-range-controls").classList.add("hidden");
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) return showToast("That run is no longer available.", true);
  const run = await response.json();
  selectedStage = run.stage;
  if (run.status === "awaiting_ground_truth") selectedResultTab = "ground-truth";
  activateRun(run);
  await Promise.all([loadWorkflow(), loadArtifacts()]);
  if (["queued", "running"].includes(run.status)) streamRun(run.id);
}

function streamRun(runId) {
  if (activeEventSource) activeEventSource.close();
  const cursor = activeRun?.logCursor || 0;
  activeEventSource = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events?cursor=${cursor}`);
  activeEventSource.onmessage = async (event) => {
    const payload = JSON.parse(event.data);
    if (activeRun?.id !== runId) return;
    if (payload.logs?.length) activeRun.logs = [...(activeRun.logs || []), ...payload.logs].slice(-1000);
    activateRun({...payload.run, logs: activeRun.logs});
    await loadWorkflow();
    if (["completed", "failed", "interrupted", "awaiting_ground_truth"].includes(activeRun.status)) {
      activeEventSource.close(); activeEventSource = null;
      await Promise.all([loadRuns(), loadArtifacts(), loadWorkflow()]);
      if (activeRun.status === "awaiting_ground_truth") {
        selectedStage = "results"; selectedResultTab = "ground-truth"; renderStages(); showResultTab(selectedResultTab);
        showToast("Test extraction is ready. Paste the held-out ground truth.");
      } else {
        showToast(activeRun.status === "completed" ? "Pipeline completed. Final results are ready." : "Pipeline stopped. Review the extraction log.", activeRun.status !== "completed");
      }
    }
  };
  activeEventSource.onerror = () => { if (activeRun?.status === "running") window.setTimeout(() => streamRun(runId), 1400); };
}

function treeHtml(entries) {
  const roots = new Map();
  entries.forEach((entry) => { let branch = roots; entry.path.split("/").forEach((part, index, parts) => { if (!branch.has(part)) branch.set(part,{entry:index === parts.length - 1 ? entry : null,children:new Map()}); branch = branch.get(part).children; }); });
  const render = (branch) => [...branch.entries()].map(([name,node]) => { const entry=node.entry||{path:"",directory:true}; const children=node.children.size?`<div class="tree-children">${render(node.children)}</div>`:""; const label=!entry.directory&&entry.path?`<button class="tree-file" data-artifact="${escapeHtml(entry.path)}" type="button">${escapeHtml(name)}</button>`:`<span class="tree-dir">${escapeHtml(name)}</span>`; return `<div class="tree-node">${label}${children}</div>`; }).join("");
  return render(roots);
}

async function loadArtifacts() {
  if (!activeRun) return;
  const response = await fetch(`/api/runs/${encodeURIComponent(activeRun.id)}/tree`);
  if (!response.ok) return;
  const {entries=[]} = await response.json();
  $("#artifact-tree").innerHTML = entries.length ? treeHtml(entries) : "Artifacts will appear as the pipeline runs.";
  document.querySelectorAll("[data-artifact]").forEach((button) => button.addEventListener("click", () => previewArtifact(button.dataset.artifact)));
}

async function previewArtifact(path) {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(`/api/runs/${encodeURIComponent(activeRun.id)}/preview/${encoded}`);
  if (!response.ok) return showToast("Could not read that artifact.", true);
  const artifact = await response.json();
  $("#preview-name").textContent = artifact.path;
  $("#download-artifact").href = artifactUrl(path);
  $("#download-artifact").classList.remove("hidden");
  $("#artifact-content").textContent = artifact.kind === "binary" ? `Binary artifact (${artifact.size.toLocaleString()} bytes). Use Open to view it.` : `${artifact.content}${artifact.truncated ? "\n\n… Preview truncated." : ""}`;
}

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("#pdf-file").files[0];
  if (!file) return showToast("Choose a PDF first.", true);
  const query = new URLSearchParams({pages:$("#pages").value,maxTrainPages:$("#max-train-pages").value,firstTrainPages:$("#first-train-pages").value,holdoutPages:$("#holdout-pages").value,evalPages:$("#eval-pages").value,trainThreshold:$("#threshold").value,sonarDropPercent:$("#sonar-drop-percent").value,trainMaxRetries:"3",eval:String($("#eval").checked),manualGroundTruth:String($("#manual-ground-truth").checked),noRouting:String($("#no-routing").checked),easyModel:$("#easy-model").value,hardModel:$("#hard-model").value,singleModel:$("#single-model").value,referenceModel:$("#reference-model").value,templateModel:$("#template-model").value});
  const button = $("#upload-form button[type=submit]"); button.disabled=true; button.textContent="Starting…";
  try {
    const response = await fetch(`/api/runs?${query}`,{method:"POST",headers:{"Content-Type":"application/pdf","X-Filename":file.name},body:file});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Could not start run.");
    await loadRuns(); await selectRun(data.id); showToast("Pipeline started. The view will advance with each stage.");
  } catch (error) { showToast(error.message,true); }
  finally { button.disabled=false; button.innerHTML='Start pipeline <span>→</span>'; }
});

$("#pdf-file").addEventListener("change",(event)=>{$("#file-name").textContent=event.target.files[0]?.name||"Choose a PDF";});
function updateEvalControls(){const enabled=$("#eval").checked;$("#eval-pages").disabled=!enabled;$("#manual-ground-truth").disabled=!enabled;if(!enabled)$("#manual-ground-truth").checked=false;}
$("#eval").addEventListener("change",updateEvalControls);
$("#no-routing").addEventListener("change",updateRoutingControls);
function changeSonarThreshold(value) {
  if (value === "") return;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return;
  sonarThreshold = Math.max(0, Math.min(100, parsed));
  hoveredSonarPage = null;
  renderSonar();
}
$("#sonar-threshold-slider").addEventListener("input", (event) => changeSonarThreshold(event.target.value));
$("#sonar-threshold-number").addEventListener("input", (event) => changeSonarThreshold(event.target.value));
async function submitSonarScan(payload, button, idleText) {
  const runId = activeRun?.id;
  if (!runId) return;
  button.disabled = true; button.textContent = "Scanning…";
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/sonar/scan`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...payload,thresholdPercent:sonarThreshold})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Sonar scan failed.");
    if (activeRun?.id !== runId) return;
    workflow.sonar = result; workflow.sonarError = null; selectedSonarPage = null; sonarGeneration++;
    $("#sonar-preview-image").removeAttribute("src");
    renderSonar(); await loadArtifacts();
    showToast(`Sonar scanned pages ${result.startPage}–${result.endPage}.`);
  } catch (error) { if (activeRun?.id === runId) showToast(error.message, true); }
  finally { button.disabled = false; button.textContent = idleText; }
}
$("#sonar-run-existing").addEventListener("click", () => {
  if (!workflow.sonarAvailable || workflow.sonar) return;
  submitSonarScan({}, $("#sonar-run-existing"), "Run sonar on this run");
});
$("#sonar-choose-range").addEventListener("click", () => {
  const sonar = workflow.sonar;
  if (!sonar) return showToast("Wait for the first sonar scan.", true);
  $("#sonar-range-controls").classList.toggle("hidden");
  $("#sonar-range-start").value = sonar.startPage;
  $("#sonar-range-end").value = sonar.endPage;
});
$("#sonar-replay").addEventListener("click", async () => {
  if (!activeRun || !workflow.sonar) return;
  const startPage = Number($("#sonar-range-start").value);
  const endPage = Number($("#sonar-range-end").value);
  const max = workflow.sonar.totalPages;
  if (!Number.isInteger(startPage) || !Number.isInteger(endPage) || startPage < 1 || endPage < startPage || endPage > max) return showToast(`Choose a range within pages 1–${max}.`, true);
  await submitSonarScan({startPage,endPage}, $("#sonar-replay"), "Replay scan");
});
document.querySelectorAll("[data-result-tab]").forEach((button)=>button.addEventListener("click",()=>showResultTab(button.dataset.resultTab)));
document.querySelectorAll("[data-output-view]").forEach((button)=>button.addEventListener("click",()=>showOutputView(button.dataset.outputView)));
$("#show-upload").addEventListener("click",()=>{$("#upload-panel").classList.remove("hidden");$("#dashboard").classList.add("hidden");$("#show-logs").classList.add("hidden");$("#run-title").textContent="Start an extraction";$("#run-status").className="status-pill idle";$("#run-status").textContent="Ready";});
function closeLogs(){ $("#logs-modal").classList.add("hidden"); document.body.classList.remove("modal-open"); }
$("#show-logs").addEventListener("click",()=>{renderLog();$("#logs-modal").classList.remove("hidden");document.body.classList.add("modal-open");$("#close-logs").focus();});
$("#close-logs").addEventListener("click",closeLogs);
document.querySelector("[data-close-logs]").addEventListener("click",closeLogs);
document.addEventListener("keydown",(event)=>{if(event.key==="Escape"&&!$("#logs-modal").classList.contains("hidden"))closeLogs();});
$("#follow-live").addEventListener("click",()=>{followLive=true;selectedStage=activeRun.stage;renderStages();});
$("#refresh-artifacts").addEventListener("click",loadArtifacts);
$("#open-run-folder").addEventListener("click",async(event)=>{
  event.preventDefault();
  if(!activeRun)return;
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(activeRun.id)}/open-folder`,{method:"POST"});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||"Could not open the run folder.");
  }catch(error){showToast(error.message,true);}
});
$("#cancel-run").addEventListener("click",async()=>{if(!activeRun)return;await fetch(`/api/runs/${activeRun.id}/cancel`,{method:"POST"});showToast("Cancellation requested.");});

$("#copy-skill-prompt").addEventListener("click",async()=>{try{await navigator.clipboard.writeText($("#ground-truth-skill-prompt").textContent);showToast("Skill prompt copied.");}catch{showToast("Could not copy the skill prompt.",true);}});
$("#copy-template").addEventListener("click",async()=>{
  if(!workflow.template)return;
  try{await navigator.clipboard.writeText($("#template-json").textContent);showToast("Template copied.");}
  catch{showToast("Could not copy the template.",true);}
});
$("#ground-truth-form").addEventListener("submit",async(event)=>{event.preventDefault();if(!activeRun||!workflow.manualGroundTruth?.awaiting)return;const button=$("#submit-ground-truth");button.disabled=true;button.textContent="Validating…";try{const runId=activeRun.id;const response=await fetch(`/api/runs/${encodeURIComponent(runId)}/ground-truth`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({groundTruth:$("#ground-truth-json").value,model:$("#ground-truth-model").value})});const data=await response.json();if(!response.ok)throw new Error(data.error||"Could not submit ground truth.");await loadRuns();await selectRun(runId);showToast("Ground truth accepted. Evaluation and metadata are complete.");}catch(error){showToast(error.message,true);await loadWorkflow();}finally{button.innerHTML='Validate and evaluate <span>→</span>';button.disabled=!workflow.manualGroundTruth?.awaiting;}});

Promise.all([loadModelOptions(),loadRuns()]).then(()=>{updateRoutingControls();updateEvalControls();}).catch(()=>showToast("Could not connect to the local dashboard server.",true));

$("#resume-run").addEventListener("click", async () => {
  if (!activeRun) return;
  const id = activeRun.id;
  $("#resume-run").disabled = true;
  try {
    const response = await fetch(`/api/runs/${id}/resume`, {method: "POST"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not resume run");
    await loadRuns();
    await selectRun(id);
    showToast("Run resumed.");
  } catch (error) { showToast(error.message); }
  finally { $("#resume-run").disabled = false; }
});
