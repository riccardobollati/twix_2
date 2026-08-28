const stages = [
  {id: "template", label: "Template inference", detail: "Learn the schema"},
  {id: "blocks", label: "Block separation", detail: "Inspect unseen pages"},
  {id: "extraction", label: "Data extraction", detail: "Build & train-evaluate"},
  {id: "results", label: "Final results", detail: "Test & compare"},
];

let activeRun = null;
let activeEventSource = null;
let workflow = {};
let selectedStage = null;
let followLive = true;
let selectedBlock = 0;
let selectedResultTab = "output";
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
  return ({queued:"Queued",running:"Running",completed:"Completed",failed:"Failed",interrupted:"Interrupted"}[status] || "Ready");
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
}

function updateRoutingControls() {
  const single = $("#no-routing").checked;
  $("#routed-models").classList.toggle("hidden", single);
  $("#single-model-wrap").classList.toggle("hidden", !single);
}

function artifactUrl(path) {
  if (!activeRun || !path) return "";
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return `/api/runs/${encodeURIComponent(activeRun.id)}/artifact/${encoded}`;
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
    if (index === current && activeRun?.status === "running") state = "active";
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
  const liveStage = activeRun?.stage || "template";
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
  ].filter(([,value]) => value !== null && value !== undefined);
  summary.innerHTML = metrics.map(([label,value]) => `<div class="stat-card"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join("");

  const labels = {"overview.png":"Accuracy, break-even & blocks","phase_cost.png":"Generation & fixing spend","token_breakdown.png":"Token breakdown by phase"};
  gallery.innerHTML = images.map((path) => { const name=path.split("/").pop(); return `<article class="stat-plot"><div class="viewer-title">${escapeHtml(labels[name] || name)}<a href="${artifactUrl(path)}" target="_blank" rel="noopener">Open</a></div><img src="${artifactUrl(path)}" alt="${escapeHtml(labels[name] || name)}" /></article>`; }).join("");
  empty.classList.toggle("hidden", images.length > 0);
}

function renderWorkflow() {
  renderModelAssignments();
  setPdf("#template-pdf", "#template-pdf-empty", workflow.templatePdf);
  $("#template-json").textContent = pretty(workflow.template, "Waiting for template inference…");
  const templateState = $("#template-state");
  templateState.textContent = workflow.template ? "Template ready" : "Waiting";
  templateState.className = `artifact-state ${workflow.template ? "ready" : "waiting"}`;

  renderBlocks();
  renderLog();
  const trainAccuracy = percent(workflow.trainEvaluation?.accuracy);
  const trainScore = $("#train-score");
  trainScore.textContent = trainAccuracy ? `Train accuracy ${trainAccuracy}` : "In progress";
  trainScore.className = `artifact-state ${trainAccuracy ? "ready" : "waiting"}`;

  setPdf("#result-pdf", "#result-pdf-empty", workflow.testPdf);
  $("#extracted-json").textContent = pretty(workflow.extracted, "Waiting for test extraction…");
  const evaluation = workflow.evaluation;
  $("#result-score strong").textContent = percent(evaluation?.accuracy) || "—";
  const nodeScore = percent(evaluation?.components?.NodeMatchF1?.typed_f1);
  const contentScore = percent(evaluation?.components?.ContentScore?.overall);
  const details = [];
  if (nodeScore) details.push(`Node match ${nodeScore}`);
  if (contentScore) details.push(`Content ${contentScore}`);
  if (evaluation?.exact_match != null) details.push(evaluation.exact_match ? "Exact match" : "Differences found");
  $("#score-breakdown").innerHTML = (details.length ? details : ["Evaluation is waiting for the test extraction."]).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  renderStatistics();
  showResultTab(selectedResultTab);
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
  $("#block-image").src = artifactUrl(images[selectedBlock]);
  $("#block-page-title").textContent = `Block overlay · page ${selectedBlock + 1}`;
  $("#block-counter").textContent = `${selectedBlock + 1} / ${images.length}`;
  $("#block-thumbnails").innerHTML = images.map((path, index) => `<button class="block-thumb ${index === selectedBlock ? "selected" : ""}" data-block-index="${index}" type="button"><img src="${artifactUrl(path)}" alt="Block overlay page ${index + 1}" loading="lazy" />Page ${index + 1}</button>`).join("");
  document.querySelectorAll("[data-block-index]").forEach((button) => button.addEventListener("click", () => { selectedBlock = Number(button.dataset.blockIndex); renderBlocks(); }));
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
  activeEventSource = null; workflow = {}; selectedBlock = 0; followLive = true; selectedResultTab = "output";
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) return showToast("That run is no longer available.", true);
  const run = await response.json();
  selectedStage = run.stage;
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
    if (["completed", "failed", "interrupted"].includes(activeRun.status)) {
      activeEventSource.close(); activeEventSource = null;
      await Promise.all([loadRuns(), loadArtifacts(), loadWorkflow()]);
      showToast(activeRun.status === "completed" ? "Pipeline completed. Final results are ready." : "Pipeline stopped. Review the extraction log.", activeRun.status !== "completed");
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
  const query = new URLSearchParams({pages:$("#pages").value,holdoutPages:$("#holdout-pages").value,evalPages:$("#eval-pages").value,trainThreshold:$("#threshold").value,trainMaxRetries:"3",eval:String($("#eval").checked),noRouting:String($("#no-routing").checked),easyModel:$("#easy-model").value,hardModel:$("#hard-model").value,singleModel:$("#single-model").value,referenceModel:$("#reference-model").value});
  const button = $("#upload-form button[type=submit]"); button.disabled=true; button.textContent="Starting…";
  try {
    const response = await fetch(`/api/runs?${query}`,{method:"POST",headers:{"Content-Type":"application/pdf","X-Filename":file.name},body:file});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Could not start run.");
    await loadRuns(); await selectRun(data.id); showToast("Pipeline started. The view will advance with each stage.");
  } catch (error) { showToast(error.message,true); }
  finally { button.disabled=false; button.innerHTML='Start pipeline <span>→</span>'; }
});

$("#pdf-file").addEventListener("change",(event)=>{$("#file-name").textContent=event.target.files[0]?.name||"Choose a PDF";});
$("#eval").addEventListener("change",()=>{$("#eval-pages").disabled=!$("#eval").checked;});
$("#no-routing").addEventListener("change",updateRoutingControls);
document.querySelectorAll("[data-result-tab]").forEach((button)=>button.addEventListener("click",()=>showResultTab(button.dataset.resultTab)));
$("#show-upload").addEventListener("click",()=>{$("#upload-panel").classList.remove("hidden");$("#dashboard").classList.add("hidden");$("#show-logs").classList.add("hidden");$("#run-title").textContent="Start an extraction";$("#run-status").className="status-pill idle";$("#run-status").textContent="Ready";});
function closeLogs(){ $("#logs-modal").classList.add("hidden"); document.body.classList.remove("modal-open"); }
$("#show-logs").addEventListener("click",()=>{renderLog();$("#logs-modal").classList.remove("hidden");document.body.classList.add("modal-open");$("#close-logs").focus();});
$("#close-logs").addEventListener("click",closeLogs);
document.querySelector("[data-close-logs]").addEventListener("click",closeLogs);
document.addEventListener("keydown",(event)=>{if(event.key==="Escape"&&!$("#logs-modal").classList.contains("hidden"))closeLogs();});
$("#follow-live").addEventListener("click",()=>{followLive=true;selectedStage=activeRun.stage;renderStages();});
$("#refresh-artifacts").addEventListener("click",loadArtifacts);
$("#cancel-run").addEventListener("click",async()=>{if(!activeRun)return;await fetch(`/api/runs/${activeRun.id}/cancel`,{method:"POST"});showToast("Cancellation requested.");});

Promise.all([loadModelOptions(),loadRuns()]).then(updateRoutingControls).catch(()=>showToast("Could not connect to the local dashboard server.",true));
