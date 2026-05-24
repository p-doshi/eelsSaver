const API_BASE = "http://localhost:8000";

const state = {
  map: null,
  hotspotLayer: null,
  selectedFeature: null,
  previewOverlay: null,
  fancyOverlay: null,
  pixelLayer: null,
  lastResult: null,
  pixelMode: "decline",
};

function fullAssetUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function clearImageryOverlays() {
  if (state.previewOverlay) {
    state.map.removeLayer(state.previewOverlay);
    state.previewOverlay = null;
  }
  if (state.fancyOverlay) {
    state.map.removeLayer(state.fancyOverlay);
    state.fancyOverlay = null;
  }
}

function showImagery(props, mode = "stress") {
  clearImageryOverlays();
  const imagery = props.imagery;
  if (!imagery) return;

  const bounds = imagery.bounds;

  if (imagery.preview_url) {
    state.previewOverlay = L.imageOverlay(
      fullAssetUrl(imagery.preview_url),
      bounds,
      { opacity: 0.84, interactive: false }
    ).addTo(state.map);
  }

  const fancyUrl =
    mode === "risk" ? imagery.risk_overlay_url :
    mode === "stress" ? imagery.stress_overlay_url :
    null;

  if (fancyUrl) {
    state.fancyOverlay = L.imageOverlay(
      fullAssetUrl(fancyUrl),
      bounds,
      { opacity: 0.58, interactive: false }
    ).addTo(state.map);
  }
}

function $(id) {
  return document.getElementById(id);
}

function setupThemeToggle() {
  const btn = document.querySelector("[data-theme-toggle]");
  btn?.addEventListener("click", () => {
    const html = document.documentElement;
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    btn.textContent = next === "dark" ? "☼" : "☾";
  });
}

function riskColor(level) {
  return {
    low: "#52b788",
    moderate: "#ffd166",
    high: "#ff8c42",
    critical: "#ff4d6d",
  }[level] || "#83f0d7";
}

function styleFeature(feature) {
  const level = feature.properties.risk_level;
  return {
    color: riskColor(level),
    weight: 2,
    fillColor: riskColor(level),
    fillOpacity: 0.3,
    opacity: 0.95,
  };
}

function popupHTML(props) {
  return `
    <div style="min-width:220px">
      <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.7;margin-bottom:6px">Hotspot</div>
      <div style="font-size:18px;font-weight:800;margin-bottom:6px">${props.name}</div>
      <div style="font-size:13px;opacity:.8;margin-bottom:8px">${props.region}, ${props.country}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
        <div><div style="font-size:11px;opacity:.7">Risk</div><strong>${Math.round(props.depletion_risk_90d * 100)}%</strong></div>
        <div><div style="font-size:11px;opacity:.7">Confidence</div><strong>${Math.round(props.confidence * 100)}%</strong></div>
      </div>
      <div style="font-size:13px;line-height:1.45">${props.summary}</div>
    </div>
  `;
}

function fillDrawer(props) {
  state.selectedFeature = props;

  $("drawerTitle").textContent = props.name;
  $("drawerRisk").textContent = props.risk_level;
  $("drawerRisk").style.borderColor = riskColor(props.risk_level);
  $("drawerRisk").style.color = riskColor(props.risk_level);
  $("eelgrassValue").textContent = `${props.eelgrass_pct_estimate}%`;
  $("riskValue").textContent = `${Math.round(props.depletion_risk_90d * 100)}%`;
  $("confidenceValue").textContent = `${Math.round(props.confidence * 100)}%`;
  $("summaryText").textContent = props.summary;

  const driverList = $("driverList");
  driverList.innerHTML = "";
  (props.drivers || []).forEach(d => {
    const li = document.createElement("li");
    li.textContent = d;
    driverList.appendChild(li);
  });

  const reportList = $("reportList");
  reportList.innerHTML = "";
  (props.reports || []).forEach(r => {
    const li = document.createElement("li");
    li.innerHTML = `<a href="${r.url}" target="_blank" rel="noopener noreferrer">${r.title}</a>`;
    reportList.appendChild(li);
  });

  $("imageryCredit").textContent = props.imagery?.credit || "No imagery source provided.";

  // Confidence explanation block (only populated for live inference results).
  const confBlock = $("confidenceBlock");
  if (props.confidence_headline || (props.confidence_factors || []).length) {
    confBlock.hidden = false;
    $("confidenceHeadline").textContent = props.confidence_headline || "";
    const factorList = $("confidenceFactors");
    factorList.innerHTML = "";
    (props.confidence_factors || []).forEach(f => {
      const li = document.createElement("li");
      li.textContent = f;
      factorList.appendChild(li);
    });
  } else {
    confBlock.hidden = true;
  }
}

// Linear interpolation across three stops (green → yellow → red).
function interpRGB(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0, [82, 183, 136]],     // green
    [0.5, [255, 209, 102]],    // yellow
    [1.0, [255, 77, 109]],     // red
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const u = (t - t0) / (t1 - t0);
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * u));
    }
  }
  return stops[stops.length - 1][1];
}

function pixelColor(value, mode) {
  // For decline & confidence: high value = red. For GPI: high value = green (invert).
  const t = mode === "gpi" ? 1.0 - value : value;
  const [r, g, b] = interpRGB(t);
  return `rgb(${r},${g},${b})`;
}

function pixelMetric(pixel, mode) {
  if (mode === "gpi") return pixel.gpi;
  if (mode === "confidence") return pixel.confidence;
  return pixel.decline_prob;
}

function pixelTooltip(pixel) {
  return `
    <div style="font-family:Inter,sans-serif;font-size:12px;min-width:160px">
      <div><strong>Coverage (GPI):</strong> ${(pixel.gpi * 100).toFixed(0)}%</div>
      <div><strong>Depletion risk:</strong> ${(pixel.decline_prob * 100).toFixed(0)}%</div>
      <div><strong>Confidence:</strong> ${(pixel.confidence * 100).toFixed(0)}%</div>
      <div><strong>Stress:</strong> ${pixel.stress}</div>
      <div style="opacity:.7;margin-top:4px">
        ${pixel.valid_obs} valid passes · ${pixel.in_distribution ? "in" : "out of"} training distribution
      </div>
    </div>`;
}

function renderPixelHeatmap(result, mode) {
  if (state.pixelLayer) {
    state.map.removeLayer(state.pixelLayer);
    state.pixelLayer = null;
  }

  const pixels = result.pixels || [];
  if (!pixels.length) return;

  const [dx, dy] = result.pixel_size_deg || [0.001, 0.001];
  const halfDx = dx / 2;
  const halfDy = dy / 2;

  const rectangles = pixels.map(p => {
    const v = pixelMetric(p, mode);
    return L.rectangle(
      [[p.lat - halfDy, p.lon - halfDx], [p.lat + halfDy, p.lon + halfDx]],
      {
        color: pixelColor(v, mode),
        fillColor: pixelColor(v, mode),
        weight: 0,
        fillOpacity: 0.78,
        interactive: true,
      }
    ).bindTooltip(pixelTooltip(p), { sticky: true, opacity: 0.95 });
  });

  state.pixelLayer = L.layerGroup(rectangles).addTo(state.map);
}

function updatePixelLegend(mode) {
  const el = $("pixelLegend");
  if (!el) return;
  const captions = {
    decline:    "Green = low depletion risk · Red = high depletion risk.",
    gpi:        "Green = healthy eelgrass coverage · Red = sparse / stressed.",
    confidence: "Green = high-confidence pixel · Red = low-confidence pixel.",
  };
  el.textContent = captions[mode] || "";
}

function setupPixelModeButtons() {
  document.querySelectorAll("[data-pixel-mode]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-pixel-mode]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.pixelMode = btn.dataset.pixelMode;
      updatePixelLegend(state.pixelMode);
      if (state.lastResult) renderPixelHeatmap(state.lastResult, state.pixelMode);
    });
  });
}

async function loadHotspots() {
  const res = await fetch(`${API_BASE}/api/hotspots`);
  const data = await res.json();
  $("hotspotCount").textContent = data.features.length;

  state.hotspotLayer = L.geoJSON(data, {
    style: styleFeature,
    onEachFeature: (feature, layer) => {
      layer.bindPopup(popupHTML(feature.properties), { className: "custom-popup" });

      layer.on("mouseover", () => {
        layer.setStyle({ weight: 3, fillOpacity: 0.45 });
        fillDrawer(feature.properties);
      });

      layer.on("mouseout", () => {
        state.hotspotLayer.resetStyle(layer);
      });

      layer.on("click", () => {
        fillDrawer(feature.properties);

        state.map.flyToBounds(layer.getBounds(), {
            padding: [50, 50],
            duration: 1.1,
            easeLinearity: 0.2
        });

        showImagery(feature.properties, "stress");
        });
    },
  }).addTo(state.map);

  state.map.fitBounds(state.hotspotLayer.getBounds(), { padding: [40, 40] });
}
function setupLayerButtons() {
  document.querySelectorAll(".layer-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".layer-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      if (!state.selectedFeature) return;

      const mode = btn.dataset.layerMode;
      if (mode === "preview") {
        showImagery(state.selectedFeature, null);
      } else {
        showImagery(state.selectedFeature, mode);
      }
    });
  });
}

function initMap() {
  state.map = L.map("map", {
    zoomControl: false,
    worldCopyJump: true,
    preferCanvas: true,
  }).setView([46.3, -63.2], 6);

  L.control.zoom({ position: "bottomleft" }).addTo(state.map);

  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: "abcd",
      maxZoom: 19,
    }
  ).addTo(state.map);
}

async function startInference(payload) {
  const res = await fetch(`${API_BASE}/api/inference/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

function polygonCenter(feature) {
  const coords = feature.geometry.coordinates[0];
  let sumX = 0, sumY = 0;
  coords.forEach(([lng, lat]) => {
    sumX += lng;
    sumY += lat;
  });
  return {
    lon: sumX / coords.length,
    lat: sumY / coords.length,
  };
}

async function loadAreaPhotos(lat, lon) {
  const gallery = $("photoGallery");
  gallery.innerHTML = `<div class="photo-placeholder">Loading nearby photos…</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/photos?lat=${lat}&lon=${lon}&radius=4000&limit=4`);
    const data = await res.json();

    if (!data.photos || data.photos.length === 0) {
      gallery.innerHTML = `<div class="photo-placeholder">No nearby photos found for this area.</div>`;
      return;
    }

    gallery.innerHTML = data.photos.map(photo => `
      <a class="photo-card" href="${photo.page_url}" target="_blank" rel="noopener noreferrer">
        <img src="${photo.url}" alt="${photo.title}" loading="lazy" />
        <span>${photo.title}</span>
      </a>
    `).join("");
  } catch (err) {
    gallery.innerHTML = `<div class="photo-placeholder">Could not load photos for this area.</div>`;
  }
}

function connectJob(jobId) {
  const ws = new WebSocket(`${API_BASE.replace("http", "ws")}/api/inference/ws/${jobId}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    $("jobStage").textContent = `${msg.stage || "running"} — ${msg.message || ""}`;
    if (msg.fact && msg.progress > 5) {
        $("jobFact").textContent = msg.fact;
    } else {
        $("jobFact").textContent = "Running model pipeline…";
    }    
    $("progressBar").style.width = `${msg.progress || 0}%`;

    if (msg.stage === "completed" && msg.result) {
      const r = msg.result;
      addInferencePolygon(r);
      state.lastResult = r;

      if (r.pixels && r.pixels.length) {
        $("pixelLegendBlock").hidden = false;
        renderPixelHeatmap(r, state.pixelMode);
        updatePixelLegend(state.pixelMode);
      } else {
        $("pixelLegendBlock").hidden = true;
      }

      fillDrawer({
        name: r.area_name,
        risk_level: r.risk_tier || (r.depletion_risk_90d > 0.7 ? "high" : r.depletion_risk_90d > 0.45 ? "moderate" : "low"),
        eelgrass_pct_estimate: r.eelgrass_pct_estimate,
        depletion_risk_90d: r.depletion_risk_90d,
        confidence: r.confidence,
        summary: `Live model result from Sentinel-2 (${r.scene_count} scenes, ${r.pixels_scored || 0} pixels scored, ${r.confidence_tier || "n/a"} confidence).`,
        drivers: r.top_drivers,
        reports: [],
        confidence_headline: r.confidence_headline,
        confidence_factors: r.confidence_factors,
      });
      ws.close();
    }

    if (msg.stage === "failed") {
      $("jobFact").textContent = msg.message || "Inference failed.";
      ws.close();
    }
  };
}

function bboxToPolygon(bbox) {
  const [minx, miny, maxx, maxy] = bbox;
  return [
    [miny, minx],
    [miny, maxx],
    [maxy, maxx],
    [maxy, minx],
    [miny, minx],
  ];
}

function addInferencePolygon(result) {
  const risk = result.depletion_risk_90d;
  const riskLevel = risk > 0.7 ? "high" : risk > 0.45 ? "moderate" : "low";

  const poly = L.polygon(bboxToPolygon(result.bbox), {
    color: riskColor(riskLevel),
    fillColor: riskColor(riskLevel),
    fillOpacity: 0.05,
    weight: 1.5,
    dashArray: "8 6",
  }).addTo(state.map);
  const centerLat = (result.bbox[1] + result.bbox[3]) / 2;
    const centerLon = (result.bbox[0] + result.bbox[2]) / 2;
    loadAreaPhotos(centerLat, centerLon);

    state.map.flyToBounds(poly.getBounds(), {
    padding: [50, 50],
    duration: 1.2,
    easeLinearity: 0.25
    });

  poly.bindPopup(`
    <div style="min-width:220px">
      <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.7;margin-bottom:6px">Live inference</div>
      <div style="font-size:18px;font-weight:800;margin-bottom:6px">${result.area_name}</div>
      <div style="font-size:13px;opacity:.8;margin-bottom:8px">${result.scene_count} Sentinel-2 scenes</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
        <div><div style="font-size:11px;opacity:.7">Risk</div><strong>${Math.round(result.depletion_risk_90d * 100)}%</strong></div>
        <div><div style="font-size:11px;opacity:.7">Confidence</div><strong>${Math.round(result.confidence * 100)}%</strong></div>
      </div>
      <div style="font-size:13px;line-height:1.45">Estimated eelgrass cover: ${result.eelgrass_pct_estimate}%</div>
    </div>
  `, { className: "custom-popup" });

  poly.openPopup();
  state.map.fitBounds(poly.getBounds(), { padding: [40, 40] });
}

function wireForm() {
  const form = $("inferenceForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fd = new FormData(form);
    const bbox = String(fd.get("bbox")).split(",").map(v => parseFloat(v.trim()));

    const payload = {
      name: fd.get("name"),
      bbox,
      datetime_start: fd.get("datetime_start"),
      datetime_end: fd.get("datetime_end"),
      max_cloud_cover: parseFloat(fd.get("max_cloud_cover")),
    };

    $("jobStage").textContent = "Submitting job…";
    $("jobFact").textContent = "Connecting to inference service…";
    $("progressBar").style.width = "5%";

    const out = await startInference(payload);
    connectJob(out.job_id);
  });
}

async function main() {
  setupThemeToggle();
  setupLayerButtons();
  setupPixelModeButtons();
  initMap();
  wireForm();
  await loadHotspots();
}

main();