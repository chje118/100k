import base64
import json
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import openslide
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)
from openslide.deepzoom import DeepZoomGenerator


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>WSI calibration wizard</title>
  <script src="/static/openseadragon.min.js"></script>
  <script src="/static/html2canvas.min.js"></script>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      display: grid;
      grid-template-columns: 420px 1fr;
      height: 100vh;
    }
    #sidebar {
      padding: 14px;
      border-right: 1px solid #ccc;
      overflow: auto;
      background: #f7f7f7;
    }
    #viewer-wrap {
      position: relative;
      width: 100%;
      height: 100%;
      background: black;
      overflow: hidden;
    }
    #viewer {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      background: black;
      z-index: 1;
    }
    #overlay-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 50;
      pointer-events: none;
    }
    #instruction-box {
      background: #fffbe6;
      border: 1px solid #e6d27a;
      padding: 12px;
      margin-bottom: 16px;
      border-radius: 8px;
    }
    #progress {
      font-weight: bold;
      margin-bottom: 6px;
    }
    .hint {
      color: #555;
      font-size: 13px;
      line-height: 1.4;
    }
    button {
      margin: 4px 4px 4px 0;
      padding: 8px 10px;
      cursor: pointer;
    }
    pre {
      background: white;
      border: 1px solid #ddd;
      padding: 8px;
      overflow: auto;
      font-size: 12px;
      max-height: 240px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .section {
      margin-bottom: 16px;
    }
    .done {
      color: #0a7a22;
      font-weight: bold;
    }
    .active {
      color: #0b57d0;
      font-weight: bold;
    }
    .pending {
      color: #666;
    }
    #confirm-box {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 5000;
      background: white;
      border: 2px solid #333;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
      display: none;
    }
    #confirm-box button {
      min-width: 90px;
    }
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>WSI calibration wizard</h2>

    <div id="instruction-box">
      <div id="progress"></div>
      <div id="instruction"></div>
      <div class="hint" style="margin-top:8px;">
        Pan with mouse drag, zoom with wheel, then double-click to propose a calibration point.
        Accept to save it and generate four screenshots. The view resets after acceptance.
        On Save & close, a global overview screenshot is also captured.
      </div>
    </div>

    <div class="section">
      <button onclick="goBackOneStep()">Undo last step</button>
      <button onclick="resetWorkflow()">Reset all</button>
      <button onclick="saveAndClose()">Save & close</button>
    </div>

    <div class="section">
      <h3>Workflow</h3>
      <div id="workflow-status"></div>
    </div>

    <div class="section">
      <h3>Recorded coordinates</h3>
      <pre id="coords-output"></pre>
    </div>

    <div class="section">
      <h3>Saved screenshots</h3>
      <pre id="shots-output"></pre>
    </div>

    <div class="section">
      <h3>Polygon debug</h3>
      <pre id="polygon-debug"></pre>
    </div>

    <div class="section">
      <div id="status" class="hint"></div>
    </div>
  </div>

  <div id="viewer-wrap">
    <div id="viewer"></div>
    <canvas id="overlay-canvas"></canvas>
  </div>

  <div id="confirm-box">
    <div id="confirm-text" style="margin-bottom:10px;font-weight:bold;"></div>
    <button onclick="acceptPendingPoint()">Accept</button>
    <button onclick="rejectPendingPoint()">Reject</button>
  </div>

<script>
const initialTop = {{ top_polygons | safe }};
const initialBottom = {{ bottom_polygons | safe }};

const steps = [
  {key: "calibration_point_1", label: "Calibration point 1"},
  {key: "calibration_point_2", label: "Calibration point 2"},
  {key: "calibration_point_3", label: "Calibration point 3"}
];

let currentStepIndex = 0;
let records = {
  calibration_point_1: null,
  calibration_point_2: null,
  calibration_point_3: null
};

let screenshots = {
  calibration_point_1: null,
  calibration_point_2: null,
  calibration_point_3: null,
  global_overview: null
};

let pendingPoint = null;
let busySaving = false;

const viewer = OpenSeadragon({
  id: "viewer",
  prefixUrl: "",
  tileSources: "/dzi",
  showNavigator: false,
  animationTime: 0.8,
  blendTime: 0.1,
  minZoomImageRatio: 0.2,
  maxZoomPixelRatio: 8,
  visibilityRatio: 1.0,
  constrainDuringPan: true,
  clickToZoom: false,
  dblClickToZoom: false
});

const canvas = document.getElementById("overlay-canvas");
const ctx = canvas.getContext("2d");
const viewerWrap = document.getElementById("viewer-wrap");

viewer.addHandler("open", function() {
  resizeCanvas();
  fitToAllPolygons();
  redraw();
  updateInstruction();
  updatePolygonDebug();
  setStatus("Viewer loaded and focused on tile outlines.");
});

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function resizeCanvas() {
  const rect = viewerWrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function imageToScreen(x, y) {
  const tiled = viewer.world.getItemAt(0);
  const point = tiled.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  return { x: point.x, y: point.y };
}

function clearCanvas() {
  const rect = viewerWrap.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
}

function polygonBBox(poly) {
  if (!poly || poly.length === 0) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of poly) {
    const x = p[0];
    const y = p[1];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return {minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY};
}

function drawPolygonOutline(poly, color) {
  if (!poly || poly.length < 2) return;

  ctx.beginPath();
  const p0 = imageToScreen(poly[0][0], poly[0][1]);
  ctx.moveTo(p0.x, p0.y);

  for (let i = 1; i < poly.length; i++) {
    const p = imageToScreen(poly[i][0], poly[i][1]);
    ctx.lineTo(p.x, p.y);
  }

  ctx.closePath();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  ctx.strokeStyle = "black";
  ctx.lineWidth = 2.2;
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.0;
  ctx.stroke();
}

function drawPolygons(polygons, color) {
  for (const poly of polygons) {
    drawPolygonOutline(poly, color);
  }
}

function drawMarker(point, label, isPending=false) {
  const p = imageToScreen(point.x, point.y);

  ctx.beginPath();
  ctx.arc(p.x, p.y, isPending ? 11 : 8, 0, Math.PI * 2);
  ctx.strokeStyle = "black";
  ctx.lineWidth = isPending ? 2.2 : 1.8;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(p.x, p.y, isPending ? 11 : 8, 0, Math.PI * 2);
  ctx.strokeStyle = "#00ff66";
  ctx.lineWidth = isPending ? 1.2 : 1.0;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = "#00ff66";
  ctx.fill();

  if (label) {
    ctx.font = "bold 16px Arial";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";
    ctx.strokeText(label, p.x + 10, p.y - 10);
    ctx.fillStyle = "#00ff66";
    ctx.fillText(label, p.x + 10, p.y - 10);
  }
}

function fitToAllPolygons() {
  const allPolys = [...initialTop, ...initialBottom];
  if (!allPolys.length) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const poly of allPolys) {
    for (const p of poly) {
      const x = p[0];
      const y = p[1];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  const pad = 2000;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = maxX + pad;
  maxY = maxY + pad;

  const tiled = viewer.world.getItemAt(0);
  const rect = tiled.imageToViewportRectangle(minX, minY, maxX - minX, maxY - minY);
  viewer.viewport.fitBounds(rect, true);
  redraw();
}

function fitToAllCalibrationPoints() {
  const pts = [];
  if (records.calibration_point_1) pts.push(records.calibration_point_1);
  if (records.calibration_point_2) pts.push(records.calibration_point_2);
  if (records.calibration_point_3) pts.push(records.calibration_point_3);

  if (!pts.length) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pt of pts) {
    if (pt.x < minX) minX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y > maxY) maxY = pt.y;
  }

  const pad = 6000;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = maxX + pad;
  maxY = maxY + pad;

  const tiled = viewer.world.getItemAt(0);
  const rect = tiled.imageToViewportRectangle(minX, minY, maxX - minX, maxY - minY);
  viewer.viewport.fitBounds(rect, true);
}

function redraw() {
  resizeCanvas();
  clearCanvas();

  drawPolygons(initialTop, "#ff2d2d");
  drawPolygons(initialBottom, "#2d6cff");

  if (records.calibration_point_1) drawMarker(records.calibration_point_1, "P1", false);
  if (records.calibration_point_2) drawMarker(records.calibration_point_2, "P2", false);
  if (records.calibration_point_3) drawMarker(records.calibration_point_3, "P3", false);

  if (pendingPoint) drawMarker(pendingPoint, "?", true);

  document.getElementById("coords-output").textContent = JSON.stringify(records, null, 2);

  const shotSummary = {};
  for (const k of Object.keys(screenshots)) {
    const val = screenshots[k];
    if (!val) {
      shotSummary[k] = null;
    } else if (k === "global_overview") {
      shotSummary[k] = val.filename;
    } else {
      shotSummary[k] = Object.fromEntries(
        Object.entries(val).map(([name, meta]) => [name, meta.filename])
      );
    }
  }
  document.getElementById("shots-output").textContent = JSON.stringify(shotSummary, null, 2);

  renderWorkflowStatus();
}

function updatePolygonDebug() {
  function bbox(poly) {
    if (!poly || !poly.length) return null;
    const xs = poly.map(p => p[0]);
    const ys = poly.map(p => p[1]);
    return {
      min_x: Math.min(...xs),
      max_x: Math.max(...xs),
      min_y: Math.min(...ys),
      max_y: Math.max(...ys)
    };
  }

  const summary = {
    top_polygon_count: initialTop.length,
    bottom_polygon_count: initialBottom.length,
    first_top_polygon_first_5_points: initialTop.length ? initialTop[0].slice(0, 5) : null,
    first_bottom_polygon_first_5_points: initialBottom.length ? initialBottom[0].slice(0, 5) : null,
    first_top_polygon_bbox: initialTop.length ? bbox(initialTop[0]) : null,
    first_bottom_polygon_bbox: initialBottom.length ? bbox(initialBottom[0]) : null
  };
  document.getElementById("polygon-debug").textContent = JSON.stringify(summary, null, 2);
}

function renderWorkflowStatus() {
  const el = document.getElementById("workflow-status");
  let html = "";
  steps.forEach((step, idx) => {
    let cls = "pending";
    let prefix = "○";
    if (records[step.key]) {
      cls = "done";
      prefix = "✔";
    } else if (idx === currentStepIndex) {
      cls = "active";
      prefix = "➜";
    }
    html += `<div class="${cls}">${prefix} ${idx + 1}. ${step.label}</div>`;
  });
  el.innerHTML = html;
}

function updateInstruction() {
  const progress = document.getElementById("progress");
  const instruction = document.getElementById("instruction");

  if (currentStepIndex >= steps.length) {
    progress.textContent = "All steps completed";
    instruction.innerHTML = "All 3 calibration points have been recorded. Review if needed, then click <b>Save & close</b>.";
    return;
  }

  const step = steps[currentStepIndex];
  progress.textContent = `Step ${currentStepIndex + 1} of ${steps.length}`;
  instruction.textContent = `Navigate to the correct location, then double-click to propose: ${step.label}`;
}

function showConfirmBox(text) {
  document.getElementById("confirm-text").textContent = text;
  document.getElementById("confirm-box").style.display = "block";
}

function hideConfirmBox() {
  document.getElementById("confirm-box").style.display = "none";
}

async function captureCurrentCanvasAsDataUrl() {
  redraw();
  await sleep(250);
  const node = document.getElementById("viewer-wrap");
  const canvasShot = await html2canvas(node, {
    backgroundColor: null,
    useCORS: true,
    logging: false
  });
  return canvasShot.toDataURL("image/png");
}

async function captureFourZoomScreenshots(stepKey, point) {
  const originalCenter = viewer.viewport.getCenter(true);
  const originalZoom = viewer.viewport.getZoom(true);
  const minZoom = viewer.viewport.getMinZoom();

  const tiled = viewer.world.getItemAt(0);
  const targetVpPoint = tiled.imageToViewportCoordinates(point.x, point.y, true);

  const screenshotsForStep = {};

  const zoom1 = originalZoom;
  const zoom2 = Math.max(zoom1 / 4.0, minZoom);
  const zoom3 = Math.max(zoom1 / 12.0, minZoom);
  const zoom4 = Math.max(zoom1 / 36.0, minZoom);

  const levels = [
    {name: "1", zoom: zoom1},
    {name: "2", zoom: zoom2},
    {name: "3", zoom: zoom3},
    {name: "4", zoom: zoom4},
  ];

  for (const level of levels) {
    viewer.viewport.panTo(targetVpPoint, true);
    viewer.viewport.zoomTo(level.zoom, targetVpPoint, true);
    viewer.viewport.applyConstraints(true);
    redraw();
    await sleep(450);

    const dataUrl = await captureCurrentCanvasAsDataUrl();
    screenshotsForStep[level.name] = {
      filename: `${stepKey}_${level.name}.png`,
      data_url: dataUrl,
      zoom: level.zoom
    };
  }

  viewer.viewport.panTo(originalCenter, true);
  viewer.viewport.zoomTo(originalZoom, originalCenter, true);
  viewer.viewport.applyConstraints(true);
  redraw();
  await sleep(250);

  screenshots[stepKey] = screenshotsForStep;
}

async function acceptPendingPoint() {
  if (busySaving || !pendingPoint || currentStepIndex >= steps.length) return;

  busySaving = true;
  hideConfirmBox();

  const step = steps[currentStepIndex];
  records[step.key] = { x: pendingPoint.x, y: pendingPoint.y };

  redraw();
  setStatus(`Accepted ${step.label} at (${pendingPoint.x}, ${pendingPoint.y}), saving 4 screenshots...`);

  try {
    await captureFourZoomScreenshots(step.key, pendingPoint);

    pendingPoint = null;
    currentStepIndex += 1;
    updateInstruction();
    redraw();
    viewer.viewport.goHome();

    if (currentStepIndex < steps.length) {
      setStatus(`Saved ${step.label} with screenshots _1, _2, _3, _4. View reset. Next: ${steps[currentStepIndex].label}`);
    } else {
      setStatus("All steps recorded with screenshots _1, _2, _3, _4. View reset. Click Save & close.");
    }
  } catch (err) {
    setStatus("Failed to capture screenshots: " + err);
  } finally {
    busySaving = false;
  }
}

function rejectPendingPoint() {
  pendingPoint = null;
  hideConfirmBox();
  redraw();
  setStatus("Point rejected. Double-click again to choose another location.");
}

function goBackOneStep() {
  if (busySaving) return;
  pendingPoint = null;
  hideConfirmBox();

  if (currentStepIndex === 0 && !records[steps[0].key]) {
    return;
  }

  if (currentStepIndex >= steps.length) {
    currentStepIndex = steps.length - 1;
  } else if (!records[steps[currentStepIndex].key] && currentStepIndex > 0) {
    currentStepIndex -= 1;
  }

  const step = steps[currentStepIndex];
  records[step.key] = null;
  screenshots[step.key] = null;
  updateInstruction();
  redraw();
  viewer.viewport.goHome();

  setStatus(`Cleared ${step.label} and reset view.`);
}

function resetWorkflow() {
  pendingPoint = null;
  hideConfirmBox();

  for (const step of steps) {
    records[step.key] = null;
    screenshots[step.key] = null;
  }
  currentStepIndex = 0;
  updateInstruction();
  redraw();
  viewer.viewport.goHome();

  setStatus("Workflow reset.");
}

viewer.addHandler("animation", redraw);
viewer.addHandler("pan", redraw);
viewer.addHandler("zoom", redraw);
viewer.addHandler("resize", redraw);

viewer.addHandler("canvas-double-click", function(event) {
  if (busySaving) return;
  if (currentStepIndex >= steps.length) return;

  event.preventDefaultAction = true;

  const viewportPoint = viewer.viewport.pointFromPixel(event.position);
  const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

  pendingPoint = {
    x: Math.round(imagePoint.x),
    y: Math.round(imagePoint.y)
  };

  redraw();
  showConfirmBox(`Accept ${steps[currentStepIndex].label} at (${pendingPoint.x}, ${pendingPoint.y})?`);
  setStatus("Point proposed. Accept or reject.");
});

async function saveAndClose() {
  if (busySaving) return;

  busySaving = true;
  setStatus("Returning to whole-slide view and capturing overview...");

  try {
    // Return to exactly the same view used when opening the viewer.
    viewer.viewport.goHome(true);
    viewer.viewport.applyConstraints(true);

    // Ensure the calibration points are drawn on top of the home view.
    redraw();

    // Give OpenSeadragon time to render the whole-slide view.
    await sleep(700);

    // Capture the whole-slide view with calibration points visible.
    const globalDataUrl = await captureCurrentCanvasAsDataUrl();

    screenshots.global_overview = {
      filename: "global_overview.png",
      data_url: globalDataUrl,
      zoom: viewer.viewport.getZoom(true)
    };

    const payload = {
      calibration_points: [
        records.calibration_point_1,
        records.calibration_point_2,
        records.calibration_point_3
      ],
      calibration_point_1: records.calibration_point_1,
      calibration_point_2: records.calibration_point_2,
      calibration_point_3: records.calibration_point_3,
      top_polygons: initialTop,
      bottom_polygons: initialBottom,
      screenshots: screenshots
    };

    const resp = await fetch("/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error(`Save request failed with status ${resp.status}`);
    }

    const data = await resp.json();
    setStatus(data.message + " Attempting to close tab...");

    setTimeout(() => {
      window.close();
      setStatus(data.message + " You may need to close this tab manually.");
    }, 300);

  } catch (err) {
    console.error(err);
    setStatus("Save failed: " + err.message);
  } finally {
    busySaving = false;
  }
}

updateInstruction();
</script>
</body>
</html>
"""


def _normalize_polygons(polygons, input_order="xy", close_polygon=True):
    out = []
    for poly in polygons:
        arr = np.asarray(poly, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Each polygon must have shape (N, 2)")

        if input_order.lower() == "yx":
            arr = arr[:, [1, 0]]
        elif input_order.lower() != "xy":
            raise ValueError("input_order must be 'xy' or 'yx'")

        if close_polygon and len(arr) >= 3:
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])

        out.append(arr.tolist())
    return out


class NotebookWSIViewer:
    def __init__(
        self,
        slide_path,
        top_polygons,
        bottom_polygons,
        save_json="annotations.json",
        screenshot_dir="annotation_screenshots",
        polygon_input_order="xy",
        host="127.0.0.1",
        port=5000,
    ):
        self.slide_path = str(Path(slide_path).expanduser().resolve())
        self.top_polygons = _normalize_polygons(top_polygons, input_order=polygon_input_order)
        self.bottom_polygons = _normalize_polygons(bottom_polygons, input_order=polygon_input_order)
        self.save_json = str(Path(save_json).expanduser().resolve())
        self.screenshot_dir = str(Path(screenshot_dir).expanduser().resolve())
        self.polygon_input_order = polygon_input_order
        self.host = host
        self.port = port

        self.slide = openslide.OpenSlide(self.slide_path)
        self.level0_w, self.level0_h = self.slide.dimensions
        # limit_bounds=False so DeepZoom level-0 matches full slide coords
        self.dz = DeepZoomGenerator(self.slide, tile_size=254, overlap=1, limit_bounds=False)

        self.app = Flask(__name__ + f"_{port}")
        self._thread = None
        self._setup_routes()

    def _polygon_summary(self, polygons):
        if not polygons:
            return {"count": 0}
        arr = np.vstack([np.asarray(p) for p in polygons if len(p) > 0])
        return {
            "count": len(polygons),
            "min_x": float(arr[:, 0].min()),
            "max_x": float(arr[:, 0].max()),
            "min_y": float(arr[:, 1].min()),
            "max_y": float(arr[:, 1].max()),
        }

    def _setup_routes(self):
        app = self.app
        dz = self.dz
        save_json = self.save_json
        screenshot_dir = Path(self.screenshot_dir)
        top_polygons = self.top_polygons
        bottom_polygons = self.bottom_polygons
        img_w = self.level0_w
        img_h = self.level0_h
        top_summary = self._polygon_summary(top_polygons)
        bottom_summary = self._polygon_summary(bottom_polygons)

        @app.route("/static/<path:filename>")
        def static_files(filename):
            return send_from_directory(Path.cwd() / "static", filename)

        @app.route("/")
        def index():
            return render_template_string(
                HTML,
                top_polygons=json.dumps(top_polygons),
                bottom_polygons=json.dumps(bottom_polygons),
            )

        @app.route("/debug_polygons")
        def debug_polygons():
            return jsonify({
                "image_width": img_w,
                "image_height": img_h,
                "top_summary": top_summary,
                "bottom_summary": bottom_summary,
                "first_top_polygon": top_polygons[0][:10] if top_polygons else None,
                "first_bottom_polygon": bottom_polygons[0][:10] if bottom_polygons else None,
            })

        @app.route("/dzi")
        def dzi():
            return Response(dz.get_dzi("jpeg"), mimetype="application/xml")

        @app.route("/dzi_files/<int:level>/<int:col>_<int:row>.jpeg")
        def tile(level, col, row):
            tile = dz.get_tile(level, (col, row))
            buf = BytesIO()
            tile.save(buf, format="JPEG", quality=90)
            return Response(buf.getvalue(), mimetype="image/jpeg")

        @app.route("/save", methods=["POST"])
        def save():
            data = request.get_json(force=True)

            screenshot_dir.mkdir(parents=True, exist_ok=True)

            screenshots = data.get("screenshots", {})
            saved_screens = {}

            for step_name, shot_group in screenshots.items():
                if not shot_group:
                    continue

                saved_screens[step_name] = {}

                # global_overview is a single screenshot dict, others are dicts of levels
                if step_name == "global_overview":
                    shot = shot_group
                    if shot and shot.get("data_url"):
                        _, encoded = shot["data_url"].split(",", 1)
                        img_bytes = base64.b64decode(encoded)
                        filename = shot.get("filename", "global_overview.png")
                        out_path = screenshot_dir / filename
                        out_path.write_bytes(img_bytes)
                        saved_screens[step_name] = str(out_path)
                        shot.pop("data_url", None)
                    continue

                for level_name, shot in shot_group.items():
                    if shot and shot.get("data_url"):
                        _, encoded = shot["data_url"].split(",", 1)
                        img_bytes = base64.b64decode(encoded)
                        filename = shot.get("filename", f"{step_name}_{level_name}.png")
                        out_path = screenshot_dir / filename
                        out_path.write_bytes(img_bytes)
                        saved_screens[step_name][level_name] = str(out_path)
                        shot.pop("data_url", None)

            data["saved_screenshot_paths"] = saved_screens

            Path(save_json).write_text(json.dumps(data, indent=2))
            return jsonify({"message": f"Saved annotations to {save_json} and screenshots to {self.screenshot_dir}"})

        @app.route("/state")
        def state():
            p = Path(save_json)
            if p.exists():
                return jsonify(json.loads(p.read_text()))
            return jsonify({})

    def start(self, open_browser=False, wait_seconds=1.0):
        if self._thread is not None and self._thread.is_alive():
            return f"http://{self.host}:{self.port}"

        def run():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(wait_seconds)

        url = f"http://{self.host}:{self.port}"
        print("Viewer URL:", url)
        print("Polygon debug URL:", f"http://{self.host}:{self.port}/debug_polygons")
        return url

    def load_annotations(self):
        p = Path(self.save_json)
        if not p.exists():
            raise FileNotFoundError(f"No annotation file found at {self.save_json}")
        return json.loads(p.read_text())


def launch_calibration_viewer(
    slide_path,
    top_polygons,
    bottom_polygons,
    save_json="annotations.json",
    screenshot_dir="annotation_screenshots",
    polygon_input_order="xy",
    host="127.0.0.1",
    port=5000,
    open_browser=False,
):
    viewer = NotebookWSIViewer(
        slide_path=slide_path,
        top_polygons=top_polygons,
        bottom_polygons=bottom_polygons,
        save_json=save_json,
        screenshot_dir=screenshot_dir,
        polygon_input_order=polygon_input_order,
        host=host,
        port=port,
    )
    url = viewer.start(open_browser=open_browser)
    return viewer, url