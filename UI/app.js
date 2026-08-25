const stages = [
  ["ocr", "OCR & layout"], ["beginning", "Find data start"], ["environment", "Create workspace"],
  ["template", "Infer template"], ["records", "Separate records"], ["blocks", "Separate blocks"],
  ["samples", "Sample blocks"], ["routing", "Route difficulty"], ["extractors", "Generate extractors"],
  ["package", "Package pipeline"], ["train_eval", "Train evaluation"], ["repair", "Repair, if needed"],
  ["group", "Group records"], ["merge", "Merge continuations"], ["ground_truth", "Ground truth"], ["evaluate", "Evaluate"],
];

let activeRun = null;
let activeEventSource = null;
let artifactEntries = [];

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

function pretty(value) {
  return value == null ? "Not available yet." : JSON.stringify(value, null, 2);
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast ${isError ? "error" : ""}`;
  setTimeout(() => toast.classList.add("hidden"), 4200);
}

function statusLabel(status) {
  return ({queued:"Queued", running:"Running", completed:"Completed", failed:"Failed", interrupted:"Interrupted"}[status] || "Ready");
}

function setStatus(run) {
  const badge = $("#run-status");
  badge.textContent = statusLabel(run.status);
  badge.className = `status-pill ${run.status || "idle"}`;
  $("#cancel-run").classList.toggle("hidden", run.status !== "running");
}

function renderStages(run) {
  const current = stages.findIndex(([id]) => id === run.stage);
  const complete = run.status === "completed";
  const failed = run.status === "failed";
  $("#stages").innerHTML = stages.map(([id, label], index) => {
    let state = "upcoming";
    if (complete || index < current) state = "done";
    if (index === current && !complete) state = failed ? "failed" : "active";
    if (!run.config.eval && ["ground_truth", "evaluate"].includes(id)) state = "skipped";
    return `<div class="stage ${state}"><span class="stage-dot">${state === "done" ? "✓" : index + 1}</span><span>${escapeHtml(label)}</span></div>`;
  }).join("");
}

function renderConfig(run) {
  const config = run.config || {};
  $("#run-config").innerHTML = [
    ["PDF", run.filename], ["Training pages", config.pages], ["Evaluation", config.eval ? "Enabled" : "Disabled"],
    ["Repair threshold", config.trainThreshold || "Disabled"], ["Started", run.startedAt ? new Date(run.startedAt).toLocaleString() : "Waiting"],
  ].map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
}

function renderMetrics(run, evaluation) {
  const cards = [];
  if (run.artifactCount != null) cards.push(["Artifacts", run.artifactCount]);
  if (evaluation?.accuracy != null) cards.push(["Accuracy", `${Math.round(evaluation.accuracy * 100)}%`]);
  if (evaluation?.exact_match != null) cards.push(["Exact match", evaluation.exact_match ? "Yes" : "No"]);
  $("#metric-cards").innerHTML = cards.map(([label, value]) => `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

function renderLog(lines) {
  const log = $("#run-log");
  log.textContent = lines.length ? lines.join("\n") : "Waiting for pipeline output…";
  $("#log-count").textContent = lines.length ? `${lines.length} lines` : "";
  log.scrollTop = log.scrollHeight;
}

function activateRun(run) {
  activeRun = {...activeRun, ...run};
  $("#run-title").textContent = activeRun.filename || "Untitled run";
  $("#upload-panel").classList.add("hidden");
  $("#dashboard").classList.remove("hidden");
  setStatus(activeRun);
  renderStages(activeRun);
  renderConfig(activeRun);
  renderRunList();
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

async function selectRun(runId) {
  if (activeEventSource) activeEventSource.close();
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) return showToast("That run is no longer available.", true);
  const run = await response.json();
  activateRun(run);
  renderLog(run.logs || []);
  await Promise.all([loadArtifacts(), loadComparison()]);
  if (["queued", "running"].includes(run.status)) streamRun(run.id);
}

function streamRun(runId) {
  if (activeEventSource) activeEventSource.close();
  activeEventSource = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  activeEventSource.onmessage = async (event) => {
    const payload = JSON.parse(event.data);
    if (activeRun?.id !== runId) return;
    activeRun = {...activeRun, ...payload.run};
    if (payload.logs?.length) {
      activeRun.logs = [...(activeRun.logs || []), ...payload.logs].slice(-1000);
      renderLog(activeRun.logs);
    }
    activateRun(activeRun);
    if (["completed", "failed", "interrupted"].includes(activeRun.status)) {
      activeEventSource.close();
      activeEventSource = null;
      await Promise.all([loadRuns(), loadArtifacts(), loadComparison()]);
      showToast(activeRun.status === "completed" ? "Pipeline completed. Output is ready to inspect." : "Pipeline stopped. Review the activity log.", activeRun.status !== "completed");
    }
  };
  activeEventSource.onerror = () => { if (activeRun?.status === "running") setTimeout(() => streamRun(runId), 1400); };
}

function treeHtml(entries) {
  const roots = new Map();
  entries.forEach((entry) => {
    const parts = entry.path.split("/"); let branch = roots;
    parts.forEach((part, index) => {
      if (!branch.has(part)) branch.set(part, {entry: index === parts.length - 1 ? entry : null, children: new Map()});
      if (index === parts.length - 1 && !branch.get(part).entry) branch.get(part).entry = entry;
      branch = branch.get(part).children;
    });
  });
  const render = (branch, depth = 0) => [...branch.entries()].map(([name, node]) => {
    const entry = node.entry || {path: "", directory: true};
    const children = node.children.size ? `<div class="tree-children">${render(node.children, depth + 1)}</div>` : "";
    const file = !entry.directory && entry.path;
    const nameHtml = file ? `<button class="tree-file" data-artifact="${escapeHtml(entry.path)}" type="button">${escapeHtml(name)}</button>` : `<span class="tree-dir">${escapeHtml(name)}</span>`;
    return `<div class="tree-node">${nameHtml}${children}</div>`;
  }).join("");
  return render(roots);
}

async function loadArtifacts() {
  if (!activeRun) return;
  const response = await fetch(`/api/runs/${activeRun.id}/tree`);
  if (!response.ok) return;
  const data = await response.json(); artifactEntries = data.entries || [];
  $("#artifact-tree").innerHTML = artifactEntries.length ? treeHtml(artifactEntries) : "Artifacts will appear as the pipeline runs.";
  document.querySelectorAll("[data-artifact]").forEach((button) => button.addEventListener("click", () => previewArtifact(button.dataset.artifact)));
}

async function previewArtifact(path) {
  const response = await fetch(`/api/runs/${activeRun.id}/preview/${path.split("/").map(encodeURIComponent).join("/")}`);
  if (!response.ok) return showToast("Could not read that artifact.", true);
  const artifact = await response.json();
  $("#preview-name").textContent = artifact.path;
  const download = $("#download-artifact");
  download.href = `/api/runs/${activeRun.id}/artifact/${path.split("/").map(encodeURIComponent).join("/")}`;
  download.classList.remove("hidden");
  $("#artifact-content").textContent = artifact.kind === "binary" ? `Binary artifact (${artifact.size.toLocaleString()} bytes). Use Open to view or download it.` : `${artifact.content}${artifact.truncated ? "\n\n… Preview truncated. Use Open for the full file." : ""}`;
}

async function loadComparison() {
  if (!activeRun) return;
  const response = await fetch(`/api/runs/${activeRun.id}/comparison`);
  if (!response.ok) return;
  const data = await response.json();
  $("#grouped-json").textContent = pretty(data.grouped);
  $("#ground-truth-json").textContent = pretty(data.groundTruth);
  const ev = data.evaluation;
  $("#evaluation-summary").innerHTML = ev ? `<div class="score-card"><span>Model-reference score</span><strong>${Math.round((ev.accuracy || 0) * 100)}%</strong><small>${ev.exact_match ? "Exact match" : "Inspect differences in eval.txt"}</small></div><div class="score-detail"><b>${escapeHtml(ev.reference || "model_reference")}</b> vs <b>${escapeHtml(ev.candidate || "code")}</b><span>Node match ${Math.round((ev.components?.NodeMatchF1?.typed_f1 || 0) * 100)}% · Content ${Math.round((ev.components?.ContentScore?.overall || 0) * 100)}%</span></div>` : "<p>Output comparison is available once evaluation artifacts are produced.</p>";
  renderMetrics(activeRun, ev);
}

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("#pdf-file").files[0];
  if (!file) return showToast("Choose a PDF first.", true);
  const pages = $("#pages").value;
  const threshold = $("#threshold").value;
  const evalEnabled = $("#eval").checked;
  const query = new URLSearchParams({pages, trainThreshold: threshold, trainMaxRetries: "3", eval: String(evalEnabled)});
  const button = $("#upload-form button[type=submit]"); button.disabled = true; button.textContent = "Starting…";
  try {
    const response = await fetch(`/api/runs?${query}`, {method: "POST", headers: {"Content-Type": "application/pdf", "X-Filename": file.name}, body: file});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not start run.");
    await loadRuns(); await selectRun(data.id); showToast("Pipeline started. You can inspect artifacts as they arrive.");
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.innerHTML = 'Run pipeline <span>→</span>'; }
});

$("#pdf-file").addEventListener("change", (event) => { $("#file-name").textContent = event.target.files[0]?.name || "Choose a PDF"; });
$("#show-upload").addEventListener("click", () => { $("#upload-panel").classList.remove("hidden"); $("#dashboard").classList.add("hidden"); $("#run-title").textContent = "Start an extraction"; $("#run-status").className = "status-pill idle"; $("#run-status").textContent = "Ready"; });
$("#refresh-artifacts").addEventListener("click", loadArtifacts);
$("#cancel-run").addEventListener("click", async () => { if (!activeRun) return; await fetch(`/api/runs/${activeRun.id}/cancel`, {method:"POST"}); showToast("Cancellation requested."); });
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => { document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab)); document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `${tab.dataset.tab}-tab`)); if (tab.dataset.tab === "artifacts") loadArtifacts(); if (tab.dataset.tab === "output") loadComparison(); }));

loadRuns().catch(() => showToast("Could not connect to the local dashboard server.", true));
