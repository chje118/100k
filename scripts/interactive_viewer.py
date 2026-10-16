import base64
import json
import math
import socket
import threading
import time
import uuid
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from PIL import Image, ImageDraw, ImageFont
from werkzeug.serving import make_server


DEFAULT_HOST = "127.0.0.1"

_ACTIVE_VIEWERS: Dict[Tuple[str, int], "NotebookWSIViewer"] = {}


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>WSI calibration wizard</title>

  <script src="{{ openseadragon_js }}"></script>
  <script src="{{ html2canvas_js }}"></script>

  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      display: grid;
      grid-template-columns: 430px 1fr;
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
        Pan with mouse drag, zoom with wheel, then double-click to propose a
        calibration point. Accept to save it and generate four screenshots.
        The view resets after acceptance.
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
const initialDisease = {{ disease_polygons | safe }};
const initialHealthy = {{ healthy_polygons | safe }};
const initialSessionId = {{ session_id | tojson }};

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
  calibration_point_3: null
};

let pendingPoint = null;
let busySaving = false;

const viewer = OpenSeadragon({
  id: "viewer",
  tileSources: "/dzi",
  showNavigator: true,
  showNavigationControl: false,
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
  setStatus("Viewer loaded.");
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

  const point = tiled.imageToViewerElementCoordinates(
    new OpenSeadragon.Point(x, y)
  );

  return {
    x: point.x,
    y: point.y
  };
}

function clearCanvas() {
  const rect = viewerWrap.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
}

function drawPolygonOutline(poly, color) {
  if (!poly || poly.length < 2) return;

  ctx.beginPath();

  const p0 = imageToScreen(
    poly[0][0],
    poly[0][1]
  );

  ctx.moveTo(p0.x, p0.y);

  for (let i = 1; i < poly.length; i++) {
    const p = imageToScreen(
      poly[i][0],
      poly[i][1]
    );

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
  const p = imageToScreen(
    point.x,
    point.y
  );

  ctx.beginPath();
  ctx.arc(
    p.x,
    p.y,
    isPending ? 11 : 8,
    0,
    Math.PI * 2
  );

  ctx.strokeStyle = "black";
  ctx.lineWidth = isPending ? 2.2 : 1.8;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(
    p.x,
    p.y,
    isPending ? 11 : 8,
    0,
    Math.PI * 2
  );

  ctx.strokeStyle = "#00ff66";
  ctx.lineWidth = isPending ? 1.2 : 1.0;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(
    p.x,
    p.y,
    1.5,
    0,
    Math.PI * 2
  );

  ctx.fillStyle = "#00ff66";
  ctx.fill();

  if (label) {
    ctx.font = "bold 16px Arial";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";

    ctx.strokeText(
      label,
      p.x + 10,
      p.y - 10
    );

    ctx.fillStyle = "#00ff66";

    ctx.fillText(
      label,
      p.x + 10,
      p.y - 10
    );
  }
}

function fitToAllPolygons() {
  const allPolys = [
    ...initialDisease,
    ...initialHealthy
  ];

  if (!allPolys.length) return;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

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

  minX = Math.max(
    0,
    minX - pad
  );

  minY = Math.max(
    0,
    minY - pad
  );

  maxX += pad;
  maxY += pad;

  const tiled = viewer.world.getItemAt(0);

  const rect = tiled.imageToViewportRectangle(
    minX,
    minY,
    maxX - minX,
    maxY - minY
  );

  viewer.viewport.fitBounds(
    rect,
    true
  );

  redraw();
}

function redraw() {
  resizeCanvas();
  clearCanvas();

  drawPolygons(
    initialDisease,
    "#ff2d2d"
  );

  drawPolygons(
    initialHealthy,
    "#2d6cff"
  );

  if (records.calibration_point_1) {
    drawMarker(
      records.calibration_point_1,
      "P1"
    );
  }

  if (records.calibration_point_2) {
    drawMarker(
      records.calibration_point_2,
      "P2"
    );
  }

  if (records.calibration_point_3) {
    drawMarker(
      records.calibration_point_3,
      "P3"
    );
  }

  if (pendingPoint) {
    drawMarker(
      pendingPoint,
      "?",
      true
    );
  }

  document.getElementById(
    "coords-output"
  ).textContent = JSON.stringify(
    records,
    null,
    2
  );

  const shotSummary = {};

  for (const k of Object.keys(screenshots)) {
    shotSummary[k] = screenshots[k]
      ? Object.fromEntries(
          Object.entries(
            screenshots[k]
          ).map(
            ([name, meta]) => [
              name,
              meta.filename
            ]
          )
        )
      : null;
  }

  document.getElementById(
    "shots-output"
  ).textContent = JSON.stringify(
    shotSummary,
    null,
    2
  );

  renderWorkflowStatus();
}

function renderWorkflowStatus() {
  const el = document.getElementById(
    "workflow-status"
  );

  let html = "";

  steps.forEach(
    (step, idx) => {
      let cls = "pending";
      let prefix = "○";

      if (records[step.key]) {
        cls = "done";
        prefix = "✔";
      } else if (idx === currentStepIndex) {
        cls = "active";
        prefix = "➜";
      }

      html += `
        <div class="${cls}">
          ${prefix} ${idx + 1}. ${step.label}
        </div>
      `;
    }
  );

  el.innerHTML = html;
}

function updateInstruction() {
  const progress =
    document.getElementById("progress");

  const instruction =
    document.getElementById("instruction");

  if (currentStepIndex >= steps.length) {
    progress.textContent =
      "All steps completed";

    instruction.innerHTML =
      "All 3 calibration points have been recorded. " +
      "Review if needed, then click <b>Save & close</b>.";

    return;
  }

  const step =
    steps[currentStepIndex];

  progress.textContent =
    `Step ${currentStepIndex + 1} of ${steps.length}`;

  instruction.textContent =
    `Navigate to the correct location, then double-click to propose: ${step.label}`;
}

function showConfirmBox(text) {
  document.getElementById(
    "confirm-text"
  ).textContent = text;

  document.getElementById(
    "confirm-box"
  ).style.display = "block";
}

function hideConfirmBox() {
  document.getElementById(
    "confirm-box"
  ).style.display = "none";
}

async function captureCurrentCanvasAsDataUrl() {
  redraw();

  await sleep(250);

  const node =
    document.getElementById(
      "viewer-wrap"
    );

  const canvasShot =
    await html2canvas(
      node,
      {
        backgroundColor: null,
        useCORS: true,
        logging: false
      }
    );

  return canvasShot.toDataURL(
    "image/png"
  );
}

async function captureFourZoomScreenshots(
  stepKey,
  point
) {
  const originalCenter =
    viewer.viewport.getCenter(true);

  const originalZoom =
    viewer.viewport.getZoom(true);

  const minZoom =
    viewer.viewport.getMinZoom();

  const tiled =
    viewer.world.getItemAt(0);

  const targetVpPoint =
    tiled.imageToViewportCoordinates(
      point.x,
      point.y,
      true
    );

  const zooms = [
    {
      name: "1",
      zoom: originalZoom
    },
    {
      name: "2",
      zoom: Math.max(
        originalZoom / 4.0,
        minZoom
      )
    },
    {
      name: "3",
      zoom: Math.max(
        originalZoom / 12.0,
        minZoom
      )
    },
    {
      name: "4",
      zoom: Math.max(
        originalZoom / 36.0,
        minZoom
      )
    }
  ];

  const screenshotsForStep = {};

  for (const level of zooms) {
    viewer.viewport.panTo(
      targetVpPoint,
      true
    );

    viewer.viewport.zoomTo(
      level.zoom,
      targetVpPoint,
      true
    );

    viewer.viewport.applyConstraints(
      true
    );

    redraw();

    await sleep(450);

    const dataUrl =
      await captureCurrentCanvasAsDataUrl();

    screenshotsForStep[
      level.name
    ] = {
      filename:
        `${stepKey}_${level.name}.png`,
      data_url:
        dataUrl
    };
  }

  viewer.viewport.panTo(
    originalCenter,
    true
  );

  viewer.viewport.zoomTo(
    originalZoom,
    originalCenter,
    true
  );

  viewer.viewport.applyConstraints(
    true
  );

  redraw();

  await sleep(250);

  screenshots[stepKey] =
    screenshotsForStep;
}

async function acceptPendingPoint() {
  if (
    busySaving ||
    !pendingPoint ||
    currentStepIndex >= steps.length
  ) {
    return;
  }

  busySaving = true;

  hideConfirmBox();

  const step =
    steps[currentStepIndex];

  records[step.key] = {
    x: pendingPoint.x,
    y: pendingPoint.y
  };

  redraw();

  setStatus(
    `Accepted ${step.label} at ` +
    `(${pendingPoint.x}, ${pendingPoint.y}), ` +
    `saving screenshots...`
  );

  try {
    await captureFourZoomScreenshots(
      step.key,
      pendingPoint
    );

    pendingPoint = null;

    currentStepIndex += 1;

    updateInstruction();
    redraw();

    viewer.viewport.goHome();

    if (currentStepIndex < steps.length) {
      setStatus(
        `Saved ${step.label}. View reset. Next: ` +
        `${steps[currentStepIndex].label}`
      );
    } else {
      setStatus(
        "All steps recorded. Click Save & close."
      );
    }

  } catch (err) {
    setStatus(
      "Failed to capture screenshots: " +
      err
    );
  } finally {
    busySaving = false;
  }
}

function rejectPendingPoint() {
  pendingPoint = null;

  hideConfirmBox();

  redraw();

  setStatus(
    "Point rejected."
  );
}

function goBackOneStep() {
  if (busySaving) return;

  pendingPoint = null;

  hideConfirmBox();

  if (
    currentStepIndex === 0 &&
    !records[steps[0].key]
  ) {
    return;
  }

  if (
    currentStepIndex >= steps.length
  ) {
    currentStepIndex =
      steps.length - 1;
  } else if (
    !records[
      steps[currentStepIndex].key
    ] &&
    currentStepIndex > 0
  ) {
    currentStepIndex -= 1;
  }

  const step =
    steps[currentStepIndex];

  records[step.key] = null;
  screenshots[step.key] = null;

  updateInstruction();
  redraw();

  viewer.viewport.goHome();

  setStatus(
    `Cleared ${step.label}.`
  );
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

  setStatus(
    "Workflow reset."
  );
}

viewer.addHandler(
  "animation",
  redraw
);

viewer.addHandler(
  "pan",
  redraw
);

viewer.addHandler(
  "zoom",
  redraw
);

viewer.addHandler(
  "resize",
  redraw
);

viewer.addHandler(
  "canvas-double-click",
  function(event) {
    if (busySaving) return;
    if (currentStepIndex >= steps.length) {
      return;
    }

    event.preventDefaultAction = true;

    const viewportPoint =
      viewer.viewport.pointFromPixel(
        event.position
      );

    const imagePoint =
      viewer.viewport.viewportToImageCoordinates(
        viewportPoint
      );

    pendingPoint = {
      x: Math.round(
        imagePoint.x
      ),
      y: Math.round(
        imagePoint.y
      )
    };

    redraw();

    showConfirmBox(
      `Accept ${steps[currentStepIndex].label} ` +
      `at (${pendingPoint.x}, ${pendingPoint.y})?`
    );

    setStatus(
      "Point proposed."
    );
  }
);

async function saveAndClose() {
  const payload = {
    calibration_points: [
      records.calibration_point_1,
      records.calibration_point_2,
      records.calibration_point_3
    ],

    calibration_point_1:
      records.calibration_point_1,

    calibration_point_2:
      records.calibration_point_2,

    calibration_point_3:
      records.calibration_point_3,

    disease_polygons:
      initialDisease,

    healthy_polygons:
      initialHealthy,

    screenshots:
      screenshots,

    session_id:
      initialSessionId
  };

  try {
    const resp =
      await fetch(
        "/calibration_save",
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json"
          },
          body:
            JSON.stringify(payload)
        }
      );

    const data =
      await resp.json();

    if (!resp.ok) {
      setStatus(
        "Save failed: " +
        data.message
      );

      return;
    }

    setStatus(
      data.message +
      " Closing..."
    );

    setTimeout(
      () => {
        window.close();
      },
      500
    );

  } catch (err) {
    setStatus(
      "Save failed: " +
      err
    );
  }
}

updateInstruction();
</script>
</body>
</html>
"""


class ViewerServer(threading.Thread):
    def __init__(
        self,
        app: Flask,
        host: str,
        port: int
    ):
        super().__init__(
            daemon=True
        )

        self.server = make_server(
            host,
            port,
            app
        )

        self.ctx = app.app_context()
        self.ctx.push()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


class NotebookWSIViewer:
    def __init__(
        self,
        slide_path: str,
        disease_polygons: Sequence[
            Sequence[Sequence[float]]
        ],
        healthy_polygons: Sequence[
            Sequence[Sequence[float]]
        ],
        save_json: str = "annotations.json",
        screenshot_dir: str = "annotation_screenshots",
        polygon_input_order: str = "xy",
        host: str = DEFAULT_HOST,
        port: Optional[int] = None,
        strict: bool = True,
        internet: bool = True,
        tile_inspection: bool = True,
        tile_inspection_dir: Optional[str] = None,
        overview_max_px: Optional[int] = None,
        tile_max_px: Optional[int] = None,
    ):
        self.tile_inspection = bool(tile_inspection)

        if tile_inspection_dir is None:
            # Follow the plate layout:
            #   <plate>/annotations/<slide>_annotations.json
            #   -> <plate>/tile_inspection/<slide>/
            ann_parent = Path(save_json).expanduser().resolve().parent
            plate_dir = (
                ann_parent.parent
                if ann_parent.name == "annotations"
                else ann_parent
            )
            tile_inspection_dir = (
                plate_dir / "tile_inspection" / Path(slide_path).stem
            )

        self.tile_inspection_dir = str(
            Path(tile_inspection_dir)
            .expanduser()
            .resolve()
        )

        # None = infer from tile size (see generate_tile_inspection)
        self.overview_max_px = (
            int(overview_max_px) if overview_max_px is not None else None
        )
        self.tile_max_px = (
            int(tile_max_px) if tile_max_px is not None else None
        )

        self.slide_path = str(
            Path(slide_path)
            .expanduser()
            .resolve()
        )

        self.save_json = str(
            Path(save_json)
            .expanduser()
            .resolve()
        )

        self.screenshot_dir = str(
            Path(screenshot_dir)
            .expanduser()
            .resolve()
        )

        self.host = host

        self.port = (
            int(port)
            if port is not None
            else 0
        )

        self.strict = bool(strict)

        self.internet = bool(
            internet
        )

        self.polygon_input_order = (
            polygon_input_order
            .lower()
            .strip()
        )

        self.session_id = (
            uuid.uuid4().hex
        )

        self.created_at = time.time()

        self.script_dir = (
            Path(__file__)
            .resolve()
            .parent
        )

        self.static_dir = (
            self.script_dir / "static"
        )

        self._validate_paths_and_environment()

        self.disease_polygons = (
            self._normalize_polygons(
                disease_polygons,
                self.polygon_input_order,
            )
        )

        self.healthy_polygons = (
            self._normalize_polygons(
                healthy_polygons,
                self.polygon_input_order,
            )
        )

        if (
            self.strict
            and not self.disease_polygons
            and not self.healthy_polygons
        ):
            raise ValueError(
                "Both disease_polygons and healthy_polygons "
                "are empty. Refusing to launch in strict mode."
            )

        self.slide = (
            openslide.OpenSlide(
                self.slide_path
            )
        )

        self.slide_dimensions = tuple(
            int(v)
            for v in self.slide.dimensions
        )

        self.dz = (
            DeepZoomGenerator(
                self.slide,
                tile_size=254,
                overlap=1,
                limit_bounds=False
            )
        )

        self.app = Flask(
            __name__ +
            f"_{self.session_id}"
        )

        self.app.config[
            "SEND_FILE_MAX_AGE_DEFAULT"
        ] = 0

        self.server_thread: Optional[
            ViewerServer
        ] = None

        self._configure_app()
        self._setup_routes()

    def _validate_paths_and_environment(
        self
    ) -> None:
        slide_path = Path(
            self.slide_path
        )

        if not slide_path.exists():
            raise FileNotFoundError(
                f"Slide file does not exist: "
                f"{slide_path}"
            )

        if not slide_path.is_file():
            raise ValueError(
                f"Slide path is not a file: "
                f"{slide_path}"
            )

        # Only require local JS files when
        # internet mode is disabled.
        if not self.internet:
            required_static = [
                self.static_dir /
                "openseadragon.min.js",

                self.static_dir /
                "html2canvas.min.js",
            ]

            missing = [
                str(p)
                for p in required_static
                if not p.exists()
            ]

            if missing:
                raise FileNotFoundError(
                    "Missing required static files: "
                    + ", ".join(missing)
                )

        Path(
            self.save_json
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Path(
            self.screenshot_dir
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            self.polygon_input_order
            not in {"xy", "yx"}
        ):
            raise ValueError(
                "polygon_input_order must be "
                "'xy' or 'yx'"
            )

    def _configure_app(self) -> None:
        @self.app.after_request
        def add_no_cache_headers(
            response: Response
        ) -> Response:
            response.headers[
                "Cache-Control"
            ] = (
                "no-store, no-cache, "
                "must-revalidate, public, "
                "max-age=0"
            )

            response.headers[
                "Pragma"
            ] = "no-cache"

            response.headers[
                "Expires"
            ] = "0"

            return response

    @staticmethod
    def _normalize_polygons(
        polygons: Sequence[
            Sequence[Sequence[float]]
        ],
        input_order: str,
        close_polygon: bool = True,
    ) -> List[
        List[List[float]]
    ]:
        normalized: List[
            List[List[float]]
        ] = []

        for poly in polygons:
            arr = np.asarray(
                poly,
                dtype=float
            )

            if (
                arr.ndim != 2
                or arr.shape[1] != 2
            ):
                raise ValueError(
                    "Each polygon must have "
                    "shape (N, 2)"
                )

            if input_order == "yx":
                arr = arr[:, [1, 0]]

            if (
                close_polygon
                and len(arr) >= 3
                and not np.allclose(
                    arr[0],
                    arr[-1]
                )
            ):
                arr = np.vstack(
                    [arr, arr[0]]
                )

            normalized.append(
                arr.tolist()
            )

        return normalized

    # ------------------------------------------------------------------
    # Tile inspection images (rendered server-side from the slide itself)
    # ------------------------------------------------------------------
    # Auto-sizing for tile inspection images (used when *_max_px is None)
    AUTO_OVERVIEW_PX_PER_TILE = 64
    AUTO_OVERVIEW_MIN = 1024
    AUTO_OVERVIEW_MAX = 8192
    AUTO_TILE_CAP = 4096

    _TILE_COLORS = {
        "top": (255, 45, 45),
        "bottom": (45, 108, 255),
    }

    def _tile_list(self) -> List[Dict]:
        """One entry per polygon (top first, then bottom) with its bbox."""
        tiles: List[Dict] = []

        for group, polys in (
            ("top", self.top_polygons),
            ("bottom", self.bottom_polygons),
        ):
            for i, poly in enumerate(polys, start=1):
                arr = np.asarray(poly, dtype=float)
                x0, y0 = arr.min(axis=0)
                x1, y1 = arr.max(axis=0)

                tiles.append(
                    {
                        # top = disease (d1, d2, ...), bottom = healthy (h1, h2, ...)
                        "id": f"{'d' if group == 'top' else 'h'}{i}",
                        "group": group,
                        "polygon": poly,
                        "bbox": [
                            float(x0), float(y0),
                            float(x1), float(y1),
                        ],
                    }
                )

        return tiles

    @staticmethod
    def _font(size: int):
        size = max(10, int(size))
        for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            return ImageFont.load_default()

    def _read_region_scaled(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        max_px: int,
    ) -> Tuple[Image.Image, float, int, int]:
        """
        Render the level-0 box [x0, x1) x [y0, y1) so that its longest side
        is at most `max_px` pixels. Never upsamples beyond level-0
        resolution. Anything outside the slide is filled white, so a tile
        near the slide edge still ends up centred.

        Reads in chunks, so even a huge overview on a slide without a
        pyramid stays within memory.

        Returns (image, downsample, int_x0, int_y0).
        """
        ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
        ix1, iy1 = int(math.ceil(x1)), int(math.ceil(y1))
        w0, h0 = ix1 - ix0, iy1 - iy0

        ds = max(1.0, max(w0, h0) / float(max_px))
        out_w = max(1, int(round(w0 / ds)))
        out_h = max(1, int(round(h0 / ds)))

        out = Image.new("RGB", (out_w, out_h), "white")

        # Only read the part that actually lies on the slide.
        sw, sh = self.slide_dimensions
        rx0, ry0 = max(ix0, 0), max(iy0, 0)
        rx1, ry1 = min(ix1, sw), min(iy1, sh)
        if rx1 <= rx0 or ry1 <= ry0:
            return out, ds, ix0, iy0

        level = self.slide.get_best_level_for_downsample(ds)
        lds = float(self.slide.level_downsamples[level])

        chunk0 = max(1, int(4096 * lds))  # chunk edge in level-0 px

        yy = ry0
        while yy < ry1:
            ch = min(chunk0, ry1 - yy)
            xx = rx0
            while xx < rx1:
                cw = min(chunk0, rx1 - xx)

                rw = max(1, int(math.ceil(cw / lds)))
                rh = max(1, int(math.ceil(ch / lds)))

                region = self.slide.read_region(
                    (int(xx), int(yy)), level, (rw, rh)
                ).convert("RGBA")

                rgb = Image.new("RGB", region.size, "white")
                rgb.paste(region, mask=region.split()[3])

                px0 = int(round((xx - ix0) / ds))
                py0 = int(round((yy - iy0) / ds))
                px1 = int(round((xx + cw - ix0) / ds))
                py1 = int(round((yy + ch - iy0) / ds))

                if px1 > px0 and py1 > py0:
                    out.paste(
                        rgb.resize(
                            (px1 - px0, py1 - py0),
                            Image.LANCZOS,
                        ),
                        (px0, py0),
                    )

                xx += cw
            yy += ch

        return out, ds, ix0, iy0

    def _draw_tile_overlay(
        self,
        img: Image.Image,
        x0: int,
        y0: int,
        ds: float,
        tiles: List[Dict],
        highlight_id: Optional[str] = None,
        label_all: bool = True,
        calibration_points: Optional[List[Optional[Dict]]] = None,
    ) -> None:
        draw = ImageDraw.Draw(img)

        scale = max(1.0, min(img.size) / 700.0)
        base_w = max(1, int(round(1.5 * scale)))
        font = self._font(14 * scale)

        def to_px(x: float, y: float) -> Tuple[float, float]:
            return ((x - x0) / ds, (y - y0) / ds)

        for t in tiles:
            pts = [to_px(x, y) for x, y in t["polygon"]]
            if len(pts) < 2:
                continue
            if pts[0] != pts[-1]:
                pts.append(pts[0])

            is_hl = t["id"] == highlight_id
            w = base_w * (2 if is_hl else 1)
            color = self._TILE_COLORS[t["group"]]

            draw.line(pts, fill=(0, 0, 0), width=w + 2 * base_w, joint="curve")
            draw.line(pts, fill=color, width=w, joint="curve")

            if label_all or is_hl:
                bx0, by0 = to_px(t["bbox"][0], t["bbox"][1])
                draw.text(
                    (bx0, by0 - 4 * scale),
                    t["id"],
                    fill=color,
                    font=font,
                    anchor="ld",
                    stroke_width=max(1, int(2 * scale)),
                    stroke_fill=(0, 0, 0),
                )

        for idx, p in enumerate(calibration_points or [], start=1):
            if not p:
                continue
            cx, cy = to_px(float(p["x"]), float(p["y"]))
            r = 8 * scale
            green = (0, 255, 102)
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=(0, 0, 0),
                width=base_w + 2,
            )
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=green,
                width=base_w,
            )
            draw.text(
                (cx + r + 2, cy - r - 2),
                f"P{idx}",
                fill=green,
                font=font,
                anchor="ld",
                stroke_width=max(1, int(2 * scale)),
                stroke_fill=(0, 0, 0),
            )

    def generate_tile_inspection(
        self,
        out_dir: Optional[str] = None,
        calibration_points: Optional[List[Optional[Dict]]] = None,
    ) -> Dict:
        """
        Write tile-inspection images:

        - overview.png: all tiles visible, with a margin of 2 tile sizes
          around their joint bounding box.
        - d<n>.png / h<n>.png (disease / healthy): one per tile, the tile centred
          with a margin of one tile size on every side (3x3 tile field of
          view), at native (level-0) resolution.

        Output sizes are inferred from the tile size unless set explicitly:
        - per tile: the 3x3 field at native resolution, i.e.
          ceil(3 * tile edge) px (capped at AUTO_TILE_CAP px for very large
          tiles);
        - overview: each tile drawn ~AUTO_OVERVIEW_PX_PER_TILE px wide,
          clamped to [AUTO_OVERVIEW_MIN, AUTO_OVERVIEW_MAX] px.

        Can also be called directly from the notebook, without the GUI.
        """
        out = Path(out_dir or self.tile_inspection_dir)
        out.mkdir(parents=True, exist_ok=True)

        tiles = self._tile_list()
        if not tiles:
            return {}

        manifest: Dict = {
            "slide_path": self.slide_path,
            "slide_dimensions": list(self.slide_dimensions),
            "mpp_x": self.slide.properties.get("openslide.mpp-x"),
            "mpp_y": self.slide.properties.get("openslide.mpp-y"),
            "tiles": [],
        }

        # --- overview ---------------------------------------------------
        bboxes = np.asarray([t["bbox"] for t in tiles], dtype=float)
        tile_w = float(np.median(bboxes[:, 2] - bboxes[:, 0]))
        tile_h = float(np.median(bboxes[:, 3] - bboxes[:, 1]))

        ox0 = bboxes[:, 0].min() - 2 * tile_w
        oy0 = bboxes[:, 1].min() - 2 * tile_h
        ox1 = bboxes[:, 2].max() + 2 * tile_w
        oy1 = bboxes[:, 3].max() + 2 * tile_h

        overview_px = self.overview_max_px
        if overview_px is None:
            tiles_across = max(
                (ox1 - ox0) / max(tile_w, 1.0),
                (oy1 - oy0) / max(tile_h, 1.0),
            )
            overview_px = int(np.clip(
                math.ceil(tiles_across * self.AUTO_OVERVIEW_PX_PER_TILE),
                self.AUTO_OVERVIEW_MIN,
                self.AUTO_OVERVIEW_MAX,
            ))

        img, ds, ix0, iy0 = self._read_region_scaled(
            ox0, oy0, ox1, oy1, overview_px
        )
        self._draw_tile_overlay(
            img, ix0, iy0, ds, tiles,
            label_all=True,
            calibration_points=calibration_points,
        )
        overview_path = out / "overview.png"
        img.save(overview_path)

        manifest["overview"] = {
            "path": str(overview_path),
            "region_level0": [ix0, iy0, ix0 + round(img.width * ds), iy0 + round(img.height * ds)],
            "downsample": ds,
            "margin_tile_size": [tile_w, tile_h],
        }

        # --- one image per tile -----------------------------------------
        for t in tiles:
            bx0, by0, bx1, by1 = t["bbox"]
            w, h = bx1 - bx0, by1 - by0
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0

            # tile + one tile size on each side = 3x the tile extent
            rx0, ry0 = cx - 1.5 * w, cy - 1.5 * h
            rx1, ry1 = cx + 1.5 * w, cy + 1.5 * h

            tile_px = self.tile_max_px
            if tile_px is None:
                tile_px = min(
                    int(math.ceil(3 * max(w, h))),
                    self.AUTO_TILE_CAP,
                )

            img, ds, ix0, iy0 = self._read_region_scaled(
                rx0, ry0, rx1, ry1, tile_px
            )
            # draw neighbours too (clipped), but highlight this tile
            self._draw_tile_overlay(
                img, ix0, iy0, ds, tiles,
                highlight_id=t["id"],
                label_all=False,
                calibration_points=calibration_points,
            )
            tile_path = out / f"{t['id']}.png"
            img.save(tile_path)

            manifest["tiles"].append(
                {
                    "id": t["id"],
                    "group": t["group"],
                    "bbox_level0": t["bbox"],
                    "path": str(tile_path),
                    "region_level0": [ix0, iy0, ix0 + round(img.width * ds), iy0 + round(img.height * ds)],
                    "downsample": ds,
                }
            )

        return manifest

    def _setup_routes(self) -> None:
        app = self.app
        outer = self
        dz = self.dz

        @app.route("/")
        def index() -> Response:
            if outer.internet:
                openseadragon_js = (
                    "https://cdn.jsdelivr.net/npm/"
                    "openseadragon@5.0.1/"
                    "build/openseadragon/"
                    "openseadragon.min.js"
                )

                html2canvas_js = (
                    "https://cdn.jsdelivr.net/npm/"
                    "html2canvas@1.4.1/"
                    "dist/html2canvas.min.js"
                )

            else:
                openseadragon_js = (
                    "/static/openseadragon.min.js"
                )

                html2canvas_js = (
                    "/static/html2canvas.min.js"
                )

            return Response(
                render_template_string(
                    HTML,
                    disease_polygons=json.dumps(
                        outer.disease_polygons
                    ),
                    healthy_polygons=json.dumps(
                        outer.healthy_polygons
                    ),
                    session_id=outer.session_id,
                    openseadragon_js=openseadragon_js,
                    html2canvas_js=html2canvas_js,
                ),
                mimetype="text/html"
            )

        @app.route(
            "/static/<path:filename>"
        )
        def local_static(
            filename: str
        ):
            return send_from_directory(
                outer.static_dir,
                filename
            )

        @app.route("/slide_info")
        def slide_info():
            return jsonify(
                {
                    "slide_path":
                        outer.slide_path,

                    "dimensions":
                        list(
                            outer.slide_dimensions
                        ),

                    "properties": {
                        "vendor":
                            outer.slide.properties.get(
                                "openslide.vendor"
                            ),

                        "mpp_x":
                            outer.slide.properties.get(
                                "openslide.mpp-x"
                            ),

                        "mpp_y":
                            outer.slide.properties.get(
                                "openslide.mpp-y"
                            ),
                    },
                }
            )

        @app.route("/dzi")
        def dzi() -> Response:
            return Response(
                dz.get_dzi("jpeg"),
                mimetype="application/xml"
            )

        @app.route(
            "/dzi_files/"
            "<int:level>/"
            "<int:col>_<int:row>.jpeg"
        )
        def tile(
            level: int,
            col: int,
            row: int
        ) -> Response:
            tile_image = dz.get_tile(
                level,
                (col, row)
            )

            buf = BytesIO()

            tile_image.save(
                buf,
                format="JPEG",
                quality=90
            )

            return Response(
                buf.getvalue(),
                mimetype="image/jpeg"
            )

        @app.route(
            "/calibration_save",
            methods=["POST"]
        )
        def calibration_save():
            data = request.get_json(
                force=True
            )

            if (
                data.get("session_id")
                != outer.session_id
            ):
                return jsonify(
                    {
                        "message":
                            "Session mismatch. "
                            "Refusing save."
                    }
                ), 409

            save_json_path = Path(
                outer.save_json
            )

            screenshot_dir_path = Path(
                outer.screenshot_dir
            )

            save_json_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            screenshot_dir_path.mkdir(
                parents=True,
                exist_ok=True
            )

            saved_screens: Dict[
                str,
                Dict[str, str]
            ] = {}

            for (
                step_name,
                shot_group
            ) in data.get(
                "screenshots",
                {}
            ).items():

                if not shot_group:
                    continue

                saved_screens[
                    step_name
                ] = {}

                for (
                    level_name,
                    shot
                ) in shot_group.items():

                    if (
                        not shot
                        or not shot.get(
                            "data_url"
                        )
                    ):
                        continue

                    _, encoded = shot[
                        "data_url"
                    ].split(",", 1)

                    out_path = (
                        screenshot_dir_path
                        / shot.get(
                            "filename",
                            f"{step_name}_"
                            f"{level_name}.png"
                        )
                    )

                    out_path.write_bytes(
                        base64.b64decode(
                            encoded
                        )
                    )

                    saved_screens[
                        step_name
                    ][level_name] = str(
                        out_path
                    )

            data[
                "saved_screenshot_paths"
            ] = saved_screens

            tile_msg = ""
            if outer.tile_inspection:
                try:
                    manifest = outer.generate_tile_inspection(
                        calibration_points=data.get(
                            "calibration_points"
                        ),
                    )
                    data["tile_inspection"] = {
                        "dir": outer.tile_inspection_dir,
                        "overview": manifest.get(
                            "overview", {}
                        ).get("path"),
                        "tiles": {
                            t["id"]: t["path"]
                            for t in manifest.get("tiles", [])
                        },
                    }
                    tile_msg = (
                        f" Tile inspection images "
                        f"({len(manifest.get('tiles', []))} tiles "
                        f"+ overview) in "
                        f"{outer.tile_inspection_dir}."
                    )
                except Exception as exc:  # never lose the calibration
                    data["tile_inspection_error"] = repr(exc)
                    tile_msg = (
                        f" Tile inspection failed: {exc!r}"
                    )

            save_json_path.write_text(
                json.dumps(
                    data,
                    indent=2
                )
            )

            def delayed_shutdown() -> None:
                time.sleep(0.5)
                outer.stop()

            threading.Thread(
                target=delayed_shutdown,
                daemon=True
            ).start()

            return jsonify(
                {
                    "message":
                        f"Saved annotations to "
                        f"{save_json_path} and "
                        f"screenshots to "
                        f"{screenshot_dir_path}."
                        f"{tile_msg}"
                }
            )

    def start(
        self,
        open_browser: bool = False,
        wait_seconds: float = 1.0
    ) -> str:
        self.port = _find_free_port(
            self.host
        )

        self.server_thread = ViewerServer(
            self.app,
            self.host,
            self.port
        )

        self.server_thread.start()

        active_key = (
            self.host,
            self.port
        )

        _ACTIVE_VIEWERS[
            active_key
        ] = self

        time.sleep(
            wait_seconds
        )

        url = (
            f"http://{self.host}:{self.port}/"
            f"?session={self.session_id}"
            f"&slide="
            f"{Path(self.slide_path).name}"
        )

        print(
            "=== WSI calibration viewer "
            "session ==="
        )

        print(
            f"slide_path: {self.slide_path}"
        )

        print(
            f"slide_dimensions: "
            f"{self.slide_dimensions}"
        )

        print(
          f"disease_polygons: "
          f"{len(self.disease_polygons)}"
        )

        print(
          f"healthy_polygons: "
          f"{len(self.healthy_polygons)}"
        )

        print(
            f"save_json: "
            f"{self.save_json}"
        )

        print(
            f"screenshot_dir: "
            f"{self.screenshot_dir}"
        )

        print(
            f"internet: "
            f"{self.internet}"
        )

        print(
            f"tile_inspection_dir: "
            f"{self.tile_inspection_dir if self.tile_inspection else 'disabled'}"
        )

        print(
            f"port: "
            f"{self.port}"
        )

        print(
            "Viewer URL:",
            url
        )

        if open_browser:
            webbrowser.open_new(
                url
            )

        return url

    def stop(self) -> None:
        key = (
            self.host,
            self.port
        )

        if self.server_thread is not None:
            try:
                self.server_thread.shutdown()
            finally:
                self.server_thread.join(
                    timeout=3
                )

                self.server_thread = None

        try:
            self.slide.close()
        except Exception:
            pass

        _ACTIVE_VIEWERS.pop(
            key,
            None
        )


def _find_free_port(
    host: str = DEFAULT_HOST
) -> int:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as sock:
        sock.bind(
            (host, 0)
        )

        return sock.getsockname()[1]


def launch_calibration_viewer(
  slide_path: str,
  disease_polygons: Sequence[
    Sequence[Sequence[float]]
  ],
  healthy_polygons: Sequence[
    Sequence[Sequence[float]]
  ],
    save_json: str = "annotations.json",
    screenshot_dir: str = "annotation_screenshots",
    polygon_input_order: str = "xy",
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    open_browser: bool = False,
    strict: bool = True,
    internet: bool = True,
    tile_inspection: bool = True,
    tile_inspection_dir: Optional[str] = None,
    overview_max_px: Optional[int] = None,
    tile_max_px: Optional[int] = None,
):
    viewer = NotebookWSIViewer(
        slide_path=slide_path,
      disease_polygons=disease_polygons,
      healthy_polygons=healthy_polygons,
        save_json=save_json,
        screenshot_dir=screenshot_dir,
        polygon_input_order=polygon_input_order,
        host=host,
        port=port,
        strict=strict,
        internet=internet,
        tile_inspection=tile_inspection,
        tile_inspection_dir=tile_inspection_dir,
        overview_max_px=overview_max_px,
        tile_max_px=tile_max_px,
    )

    url = viewer.start(
        open_browser=open_browser
    )

    return viewer, url