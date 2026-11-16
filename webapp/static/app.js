"use strict";

// Box colours, indexed by category id.
const COLORS = ["#3b9eff", "#f2c200", "#ff5d8f", "#b48cff"];

const $ = (s) => document.querySelector(s);
const state = { info: null, source: "sample", sample: null, file: null, fileUrl: null, response: null, imageUrl: null };

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) node.append(c);
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const catName = (id) => state.info?.categories?.[id]?.name ?? `class ${id}`;
const threshold = () => parseFloat($("#threshold").value);
const stem = (name) => name.replace(/\.\w+$/, "");

function iou(a, b) {
  const [x, y, w, h] = a, [xx, yy, ww, hh] = b;
  const ix = Math.max(0, Math.min(x + w, xx + ww) - Math.max(x, xx));
  const iy = Math.max(0, Math.min(y + h, yy + hh) - Math.max(y, yy));
  const inter = ix * iy, union = w * h + ww * hh - inter;
  return union > 0 ? inter / union : 0;
}

// Greedy matching by score, same class, IoU >= 0.5 (as in common/evaluation.py).
function match(detections, truth) {
  const used = new Set();
  const marked = detections.map((d) => ({ ...d, tp: false })).sort((a, b) => b.score - a.score);
  for (const d of marked) {
    let best = 0, idx = -1;
    truth.forEach((g, i) => {
      if (used.has(i) || g.category_id !== d.category_id) return;
      const o = iou(d.bbox, g.bbox);
      if (o > best) { best = o; idx = i; }
    });
    if (best >= 0.5) { d.tp = true; used.add(idx); }
  }
  return {
    marked,
    missed: truth.map((_, i) => !used.has(i)),
    tp: used.size, fp: marked.length - used.size, fn: truth.length - used.size,
  };
}

// ---------------------------------------------------------------- setup

async function init() {
  state.info = await (await fetch("/api/info")).json();
  $("#threshold").min = state.info.min_score;
  $("#threshold").value = state.info.default_threshold;
  $("#thr-out").textContent = threshold().toFixed(2);
  renderModels();
  renderLegend();
  renderThumbs();
  loadResults();
}

function renderModels() {
  $("#models").replaceChildren(...state.info.models.map((m) => {
    const input = el("input", { type: "checkbox", value: m.id });
    input.checked = m.available;
    input.disabled = !m.available;
    const kind = m.available ? (m.method === "yolo" ? "one-stage" : "two-stage") : "weights missing";
    return el("label", { class: `model${m.available ? "" : " disabled"}` }, [input, m.name, el("span", { class: "kind", text: kind })]);
  }));
}

function renderLegend() {
  const items = state.info.categories.map((c, i) =>
    el("div", { class: "legend-item" }, [el("span", { class: "swatch", style: `background:${COLORS[i]}` }), c.name]));
  items.push(el("div", { class: "legend-key", text: "solid = prediction, dashed = ground truth, ✗ = false positive" }));
  $("#legend").replaceChildren(...items);
}

function renderThumbs() {
  const samples = state.info.samples;
  $("#sample-count").textContent = samples.length ? `${samples.length} images` : "No prepared test set found";
  $("#random-btn").hidden = !samples.length;
  if (!samples.length) {
    document.querySelector('input[name=source][value=upload]').checked = true;
    setSource("upload");
    return;
  }
  $("#thumbs").replaceChildren(...samples.map((name) =>
    el("button", { class: "thumb", type: "button", "data-name": name, title: name, onclick: () => selectSample(name) }, [
      el("img", { src: `/api/thumbs/${name}`, loading: "lazy", alt: name }),
      el("span", { text: stem(name) }),
    ])));
  selectSample(samples.includes("370.png") ? "370.png" : samples[0]);
}

// ---------------------------------------------------------------- image selection

function setSource(source) {
  state.source = source;
  $("#source-sample").hidden = source !== "sample";
  $("#source-upload").hidden = source !== "upload";
  reset();
}

function selectSample(name) {
  state.sample = name;
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("active", t.dataset.name === name));
  document.querySelector(`.thumb[data-name="${name}"]`)?.scrollIntoView({ block: "nearest" });
  reset();
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) return showError("Please choose an image file.");
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  $("#file-name").textContent = file.name;
  reset();
}

function currentImageUrl() {
  return state.source === "sample" ? state.sample && `/api/samples/${state.sample}` : state.fileUrl;
}

function reset() {
  showError(null);
  state.response = null;
  $("#cards").replaceChildren();
  const url = currentImageUrl();
  $("#preview").hidden = !url;
  $("#preview").replaceChildren(url ? el("img", { src: url, alt: "Selected radiograph" }) : "");
  if (state.source === "sample") {
    $("#stage-title").textContent = state.sample ? `Test image ${stem(state.sample)}` : "No image selected";
  } else {
    $("#stage-title").textContent = state.file ? state.file.name : "No image uploaded";
  }
  $("#stage-sub").textContent = url ? "Press “Run detection” to compare the models." : "";
}

function showError(msg) {
  $("#error").hidden = !msg;
  $("#error").textContent = msg ?? "";
}

// ---------------------------------------------------------------- detection

async function run() {
  const models = [...document.querySelectorAll("#models input:checked")].map((i) => i.value);
  if (!models.length) return showError("Select at least one model.");
  const form = new FormData();
  form.append("models", models.join(","));
  if (state.source === "sample") {
    if (!state.sample) return showError("Select a test image.");
    form.append("sample", state.sample);
  } else {
    if (!state.file) return showError("Upload an image first.");
    form.append("image", state.file);
  }

  const btn = $("#run-btn");
  btn.disabled = true;
  btn.textContent = "Running…";
  showError(null);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    state.response = data;
    state.imageUrl = currentImageUrl();
    renderCards();
  } catch (err) {
    showError(`Detection failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run detection";
  }
}

function buildSvg(result, response, large = false) {
  const [vx, vy, vw, vh] = response.view;
  const svg = svgEl("svg", { viewBox: `${vx} ${vy} ${vw} ${vh}` });
  svg.append(svgEl("image", { href: state.imageUrl, x: 0, y: 0, width: response.width, height: response.height }));
  const stroke = Math.max(2, vw / (large ? 520 : 380));
  const fontSize = Math.max(11, vw / (large ? 85 : 62));
  const truth = response.ground_truth;
  const visible = result.detections.filter((d) => d.score >= threshold());
  const m = truth ? match(visible, truth) : { marked: visible.map((d) => ({ ...d, tp: null })), missed: [] };

  if (truth && $("#show-gt").checked) {
    truth.forEach((g, i) => {
      const [x, y, w, h] = g.bbox;
      const rect = svgEl("rect", {
        x, y, width: w, height: h, fill: "none", stroke: COLORS[g.category_id], "stroke-width": stroke * 0.9,
        "stroke-dasharray": `${stroke * 4} ${stroke * 3}`, opacity: m.missed[i] ? 0.95 : 0.6,
      });
      const title = svgEl("title");
      title.textContent = `Ground truth: ${catName(g.category_id)}${m.missed[i] ? " (missed)" : ""}`;
      rect.append(title);
      svg.append(rect);
    });
  }

  for (const d of m.marked) {
    const [x, y, w, h] = d.bbox;
    const color = COLORS[d.category_id];
    const g = svgEl("g");
    g.append(svgEl("rect", { x, y, width: w, height: h, fill: color, "fill-opacity": 0.08, stroke: color, "stroke-width": stroke }));
    const title = svgEl("title");
    title.textContent = `${catName(d.category_id)} ${d.score.toFixed(3)}${d.tp === false ? " (false positive)" : d.tp ? " (true positive)" : ""}`;
    g.append(title);
    if ($("#show-labels").checked) {
      const text = svgEl("text", {
        x: x + stroke, y: Math.max(vy + fontSize, y - stroke * 1.5), fill: color,
        "font-size": fontSize, "stroke-width": fontSize / 4, class: "box-label",
      });
      text.textContent = `${d.tp === false ? "✗ " : ""}${catName(d.category_id)} ${d.score.toFixed(2)}`;
      g.append(text);
    }
    svg.append(g);
  }
  return { svg, m, visible };
}

function renderCards() {
  const response = state.response;
  if (!response) return;
  $("#preview").hidden = true;
  $("#stage-sub").textContent = response.ground_truth
    ? `${response.ground_truth.length} ground-truth boxes · threshold ${threshold().toFixed(2)}`
    : `no ground truth for uploaded images · threshold ${threshold().toFixed(2)}`;

  $("#cards").replaceChildren(...response.results.map((result) => {
    const model = state.info.models.find((x) => x.id === result.model);
    const { svg, m, visible } = buildSvg(result, response);
    const perClass = {};
    visible.forEach((d) => (perClass[d.category_id] = (perClass[d.category_id] || 0) + 1));

    const counts = response.ground_truth
      ? [["tp", "TP", m.tp], ["fp", "FP", m.fp], ["fn", "FN", m.fn], ["", "Detections", visible.length]]
      : [["", "Detections", visible.length]];
    const device = result.device.startsWith("cuda") ? "GPU" : "CPU";

    return el("article", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: model?.name ?? result.model }),
        el("span", { class: "meta", text: `${result.latency_ms.toFixed(1)} ms · ${device}` }),
      ]),
      el("div", { class: "viewer", title: "Click to enlarge", onclick: () => openLightbox(result, model) }, svg),
      el("div", { class: "card-foot" }, [
        el("div", { class: "counts" }, counts.map(([cls, label, value]) => el("span", { class: cls }, [label, el("b", { text: String(value) })]))),
        el("div", { class: "per-class" }, Object.keys(perClass).length
          ? Object.entries(perClass).map(([id, n]) => el("span", {}, [el("span", { class: "swatch", style: `background:${COLORS[id]}` }), `${catName(+id)} ${n}`]))
          : el("span", { text: "Nothing above the threshold. This does not mean the teeth are healthy." })),
      ]),
    ]);
  }));
}

function openLightbox(result, model) {
  $("#lightbox-title").textContent = `${model?.name ?? result.model} · ${result.latency_ms.toFixed(1)} ms`;
  const { svg } = buildSvg(result, state.response, true);
  const body = $("#lightbox-body");
  body.replaceChildren(svg);
  $("#lightbox").hidden = false;
  // Size the SVG to the image aspect ratio so no padding around the crop is shown.
  const [, , vw, vh] = state.response.view;
  const scale = Math.min(body.clientWidth / vw, body.clientHeight / vh);
  svg.setAttribute("width", Math.floor(vw * scale));
  svg.setAttribute("height", Math.floor(vh * scale));
}

// ---------------------------------------------------------------- results tab

async function loadResults() {
  const rows = await (await fetch("/api/results")).json();
  if (!rows.length) {
    $("#results-meta").textContent = "No evaluation found. Run evaluate.py for each model first (see README).";
    return;
  }
  const first = rows[0];
  $("#results-meta").textContent = `${first.images} test images · ${first.annotations} annotations · evaluated on ${first.device}`;
  $("#results-note").textContent =
    `Precision, recall and F1 at score ≥ ${first.threshold} with IoU ≥ 0.5 and matching class. ` +
    "Latency: batch size 1, after warm-up, without disk I/O. Best value in each row is in bold.";

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const dec = (v) => v.toFixed(3);
  const metrics = [
    ["mAP50–95", "map50_95", dec, "max"],
    ["mAP50", "map50", dec, "max"],
    ["Precision", "precision", pct, "max"],
    ["Recall", "recall", pct, "max"],
    ["F1", "f1", dec, "max"],
    ["Latency per image", "latency_ms", (v) => `${v.toFixed(1)} ms`, "min"],
    ["Training time", "training_min", (v) => `${v.toFixed(0)} min`, "min"],
    ["Parameters", "params_m", (v) => `${v.toFixed(1)} M`, "min"],
  ];
  const head = el("thead", {}, el("tr", {}, [el("th", { text: "Metric" }), ...rows.map((r) => el("th", { text: r.name }))]));
  const body = el("tbody", {}, metrics.filter(([, key]) => rows.every((r) => r[key] != null)).map(([label, key, fmt, better]) => {
    const values = rows.map((r) => r[key]);
    const best = better === "max" ? Math.max(...values) : Math.min(...values);
    return el("tr", {}, [el("td", { text: label }), ...values.map((v) => el("td", { class: v === best ? "best" : "", text: fmt(v) }))]);
  }));
  $("#metrics-table").replaceChildren(head, body);

  const classes = state.info.categories.map((c) => c.name).filter((name) => name in first.per_class);
  const classHead = el("thead", {}, el("tr", {}, [el("th", { text: "Class" }), ...rows.flatMap((r) => [el("th", { text: r.name }), el("th")])]));
  const classBody = el("tbody", {}, classes.map((name, ci) => el("tr", {}, [
    el("td", {}, [el("span", { class: "swatch", style: `background:${COLORS[ci]};margin-right:7px` }), name]),
    ...rows.flatMap((r) => [
      el("td", { text: dec(r.per_class[name]) }),
      el("td", { class: "bar-cell" }, el("div", { class: "bar" }, el("div", { style: `width:${Math.min(100, r.per_class[name] * 100 / 0.6)}%;background:${COLORS[ci]}` }))),
    ]),
  ])));
  $("#class-table").replaceChildren(classHead, classBody);
}

// ---------------------------------------------------------------- events

document.querySelectorAll(".nav a").forEach((link) => link.addEventListener("click", (e) => {
  e.preventDefault();
  showView(link.dataset.view);
}));

function showView(view) {
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
}

document.querySelectorAll("input[name=source]").forEach((r) => r.addEventListener("change", () => setSource(r.value)));
$("#random-btn").addEventListener("click", () => {
  const s = state.info.samples;
  selectSample(s[Math.floor(Math.random() * s.length)]);
});
$("#file-input").addEventListener("change", (e) => selectFile(e.target.files[0]));
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
$("#threshold").addEventListener("input", () => {
  $("#thr-out").textContent = threshold().toFixed(2);
  renderCards();
});
$("#show-gt").addEventListener("change", renderCards);
$("#show-labels").addEventListener("change", renderCards);
$("#run-btn").addEventListener("click", run);
$("#lightbox-close").addEventListener("click", () => ($("#lightbox").hidden = true));
$("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox-body") $("#lightbox").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

// Optional URL parameters: ?sample=370.png&run=1&threshold=0.3 or ?view=results
async function applyUrlParams() {
  const params = new URLSearchParams(location.search);
  if (params.get("view")) showView(params.get("view"));
  if (params.get("threshold")) {
    $("#threshold").value = params.get("threshold");
    $("#thr-out").textContent = threshold().toFixed(2);
  }
  if (state.info.samples.includes(params.get("sample"))) selectSample(params.get("sample"));
  if (params.get("run") === "1") await run();
  const zoom = state.response?.results.find((r) => r.model === params.get("zoom"));
  if (zoom) openLightbox(zoom, state.info.models.find((m) => m.id === zoom.model));
}

init()
  .then(applyUrlParams)
  .catch((err) => showError(`Could not reach the server: ${err.message}`));
