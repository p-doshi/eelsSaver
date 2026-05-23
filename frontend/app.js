const API_BASE = "http://localhost:8000";

const state = {
  map: null,
  hotspotLayer: null,
  selectedFeature: null,
  previewOverlay: null,
  fancyOverlay: null,
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
      addInferencePolygon(msg.result);
      fillDrawer({
        name: msg.result.area_name,
        risk_level: msg.result.depletion_risk_90d > 0.7 ? "high" : msg.result.depletion_risk_90d > 0.45 ? "moderate" : "low",
        eelgrass_pct_estimate: msg.result.eelgrass_pct_estimate,
        depletion_risk_90d: msg.result.depletion_risk_90d,
        confidence: msg.result.confidence,
        summary: `Live model result from Sentinel-2 query using ${msg.result.scene_count} scenes.`,
        drivers: msg.result.top_drivers,
        reports: [],
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
    fillOpacity: 0.28,
    weight: 2,
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
  initMap();
  wireForm();
  await loadHotspots();
}

main();