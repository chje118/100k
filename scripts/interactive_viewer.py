import base64
import json
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
const initialTop = {{ top_polygons | safe }};
const initialBottom = {{ bottom_polygons | safe }};
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
    ...initialTop,
    ...initialBottom
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
    initialTop,
    "#ff2d2d"
  );

  drawPolygons(
    initialBottom,
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

    top_polygons:
      initialTop,

    bottom_polygons:
      initialBottom,

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
        top_polygons: Sequence[
            Sequence[Sequence[float]]
        ],
        bottom_polygons: Sequence[
            Sequence[Sequence[float]]
        ],
        save_json: str = "annotations.json",
        screenshot_dir: str = "annotation_screenshots",
        polygon_input_order: str = "xy",
        host: str = DEFAULT_HOST,
        port: Optional[int] = None,
        strict: bool = True,
        internet: bool = True,
    ):
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

        self.top_polygons = (
            self._normalize_polygons(
                top_polygons,
                self.polygon_input_order
            )
        )

        self.bottom_polygons = (
            self._normalize_polygons(
                bottom_polygons,
                self.polygon_input_order
            )
        )

        if (
            self.strict
            and not self.top_polygons
            and not self.bottom_polygons
        ):
            raise ValueError(
                "Both top_polygons and bottom_polygons "
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
                    top_polygons=json.dumps(
                        outer.top_polygons
                    ),
                    bottom_polygons=json.dumps(
                        outer.bottom_polygons
                    ),
                    session_id=outer.session_id,
                    openseadragon_js=openseadragon_js,
                    html2canvas_js=html2canvas_js,
                ),
                mimetype="text/html",
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
                        f"{screenshot_dir_path}"
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
            f"top_polygons: "
            f"{len(self.top_polygons)}"
        )

        print(
            f"bottom_polygons: "
            f"{len(self.bottom_polygons)}"
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
    top_polygons: Sequence[
        Sequence[Sequence[float]]
    ],
    bottom_polygons: Sequence[
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
        strict=strict,
        internet=internet,
    )

    url = viewer.start(
        open_browser=open_browser
    )

    return viewer, url
