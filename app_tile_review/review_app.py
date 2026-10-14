"""
Pathologist review app: whole-slide viewer + notes.

The pathologist logs in with a shared password, picks a plate, then a slide.
The slide opens in a zoomable whole-slide viewer (OpenSeadragon + OpenSlide
DeepZoom, like interactive_viewer.py), with the disease (red, d1, d2, ...) and
healthy (blue, h1, h2, ...) tiles drawn on top. Next to it she writes
disease_tile_notes, healthy_tile_notes and diagnosis, then submits. One CSV row
per slide; submitted slides are marked done.

Expected layout (as written by interactive_viewer.py):

    <DATA_ROOT>/<plate>/annotations/<slide>_annotations.json   (disease/healthy polygons)

The slide file itself (<slide>.mrxs) is looked up by file name under 
--slide-root (default: DATA_ROOT), or given explicitly in a
--slide-map CSV (columns slide_id,slide_path).

Run:
    pip install flask pillow waitress openslide-python openslide-bin
    python review_app.py --data-root /path/to/plates --slide-root /path/to/wsis
"""

import argparse
import csv
import getpass
import json
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openslide
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from openslide.deepzoom import DeepZoomGenerator
from werkzeug.security import check_password_hash, generate_password_hash

# Only accept .mrxs slide files for this review app
WSI_EXTENSIONS = {".mrxs"}

CSV_FIELDS = [
    "plate",
    "slide_id",
    "status",
    "suggested_diagnosis",
    "disease_tile_notes",
    "healthy_tile_notes",
    "diagnosis",
    "updated_at",
    "completed_at",
]
NOTE_FIELDS = ["disease_tile_notes", "healthy_tile_notes", "diagnosis"]

ANN_SUFFIX = "_annotations.json"


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
LAYOUT = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{{ title }} · Slide review</title>
<style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#1d2330;--muted:#667085;--line:#e2e5ea;
--accent:#2f5bd3;--done:#127a3a;--draft:#a35f00;--dis:#ff2d2d;--hea:#2d6cff;--err:#c4262e}
*{box-sizing:border-box}
body{margin:0;font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;align-items:center;gap:14px;padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--line);min-height:52px}
.crumbs a{color:var(--accent);text-decoration:none}.crumbs span{color:var(--muted);margin:0 6px}
.spacer{flex:1}a.logout{color:var(--muted);font-size:13px}
main{padding:24px 20px;max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
table{width:100%;border-collapse:collapse}
td,th{padding:11px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600;background:#fafbfc}
tr:last-child td{border-bottom:0}
tr.row{cursor:pointer}tr.row:hover{background:#f0f4ff}
tr.row a{color:inherit;text-decoration:none;font-weight:600}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.b-done{background:#e2f4e8;color:var(--done)}.b-draft{background:#fff0d9;color:var(--draft)}.b-todo{background:#eef0f3;color:var(--muted)}
.b-err{background:#fde8e9;color:var(--err)}
.bar{height:6px;background:#eef0f3;border-radius:3px;overflow:hidden;width:140px;display:inline-block;vertical-align:middle;margin-right:8px}
.bar>i{display:block;height:100%;background:var(--done)}
button,.btn{font:inherit;font-size:14px;padding:7px 13px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer;text-decoration:none;display:inline-block}
button:hover,.btn:hover{border-color:#c5cad3}
.primary{background:var(--accent);border-color:var(--accent);color:#fff}.primary:hover{background:#2649b0}
.muted{color:var(--muted)}
.top{display:flex;align-items:end;gap:12px;margin-bottom:16px}
</style>
</head><body>{{ body|safe }}</body></html>"""


LOGIN = r"""
<main style="max-width:380px;margin-top:12vh">
  <div class="card" style="padding:28px">
    <h1>Slide review</h1>
    <p class="muted" style="margin-top:0">Please enter the access password.</p>
    <form method="post">
      <input type="password" name="password" autofocus autocomplete="current-password"
        style="font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;width:100%">
      {% if error %}<p style="color:var(--err);margin:10px 0 0">{{ error }}</p>{% endif %}
      <button class="primary" style="width:100%;margin-top:14px">Log in</button>
    </form>
  </div>
</main>
"""


PLATES = r"""
<header>
  <div class="crumbs"><b>Plates</b></div><div class="spacer"></div>
  <a class="logout" href="{{ url_for('logout') }}">Log out</a>
</header>
<main>
  <div class="top"><div><h1>Plates</h1><div class="muted">Choose a plate to see its slides.</div></div></div>
  <div class="card"><table>
    <tr><th>Plate</th><th>Slides</th><th>Progress</th></tr>
    {% for p in plates %}
    <tr class="row" onclick="location.href='{{ url_for('plate', plate=p.name) }}'">
      <td><a href="{{ url_for('plate', plate=p.name) }}">{{ p.name }}</a></td>
      <td>{{ p.total }}</td>
      <td><span class="bar"><i style="width:{{ (100 * p.done / p.total) if p.total else 0 }}%"></i></span>
          <span class="muted">{{ p.done }} / {{ p.total }} done</span></td>
    </tr>
    {% else %}
    <tr><td colspan="3" class="muted">No plates found.</td></tr>
    {% endfor %}
  </table></div>
</main>
"""


PLATE = r"""
<header>
  <div class="crumbs"><a href="{{ url_for('index') }}">Plates</a><span>›</span><b>{{ plate }}</b></div>
  <div class="spacer"></div>
  <a class="logout" href="{{ url_for('logout') }}">Log out</a>
</header>
<main>
  <div class="top">
    <div><h1>{{ plate }}</h1><div class="muted">{{ done }} of {{ slides|length }} slides done</div></div>
    <div class="spacer"></div>
    {% if next_todo %}<a class="btn primary" href="{{ url_for('slide', plate=plate, slide=next_todo) }}">Next unreviewed slide →</a>{% endif %}
  </div>
  <div class="card"><table>
    <tr><th>Slide</th><th>Disease tiles</th><th>Healthy tiles</th><th>Status</th><th>Last saved</th></tr>
    {% for s in slides %}
    <tr class="row" onclick="location.href='{{ url_for('slide', plate=plate, slide=s.name) }}'">
      <td><a href="{{ url_for('slide', plate=plate, slide=s.name) }}">{{ s.name }}</a>
          {% if not s.has_wsi %} <span class="badge b-err">slide file not found</span>{% endif %}</td>
      <td>{{ s.n_d }}</td><td>{{ s.n_h }}</td>
      <td>{% if s.status == 'done' %}<span class="badge b-done">✓ Done</span>
          {% elif s.status == 'draft' %}<span class="badge b-draft">In progress</span>
          {% else %}<span class="badge b-todo">Not started</span>{% endif %}</td>
      <td class="muted">{{ s.updated_at or '' }}</td>
    </tr>
    {% endfor %}
  </table></div>
</main>
"""


SLIDE = r"""
<style>
body{overflow:hidden}
.review{display:grid;grid-template-columns:minmax(0,1fr) 380px;height:calc(100vh - 53px)}
.viewer-col{display:flex;flex-direction:column;min-width:0;background:#000}
.tools{display:flex;align-items:center;gap:6px;padding:7px 10px;background:#1b1f27;color:#d7dbe3;flex-wrap:wrap;font-size:13px}
.tools button{background:#2a303b;border-color:#3a4150;color:#d7dbe3;padding:4px 10px;font-size:13px}
.tools button:hover{border-color:#6b7484}
.tools label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none;margin-left:6px}
.tools .sep{width:1px;height:20px;background:#3a4150;margin:0 4px}
.chips{display:flex;gap:4px;overflow-x:auto;padding:6px 10px;background:#151920;min-height:38px;align-items:center}
.chip{flex:0 0 auto;font:600 12px/1 system-ui,sans-serif;padding:5px 8px;border-radius:6px;cursor:pointer;
      background:#232833;border:1px solid #3a4150;color:#d7dbe3}
.chip.d{border-color:#7a2a2a}.chip.h{border-color:#2a3f7a}
.chip.on{background:#fff;color:#111;border-color:#fff}
.chip .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
#wrap{position:relative;flex:1;min-height:0}
#osd{position:absolute;inset:0}
#overlay{position:absolute;inset:0;pointer-events:none;z-index:5}
#mag{position:absolute;left:10px;top:10px;z-index:6;background:rgba(0,0,0,.6);color:#fff;font:600 13px system-ui,sans-serif;padding:4px 8px;border-radius:6px;pointer-events:none}
#missing{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#ddd;text-align:center;padding:30px}
.form-col{overflow:auto;padding:18px;background:var(--panel);border-left:1px solid var(--line)}
.sec{border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.sec h3{margin:0 0 8px;font-size:15px;display:flex;align-items:center;gap:8px}
.sec .dot{width:10px;height:10px;border-radius:50%}
.sec .muted{font-size:13px;margin:0 0 8px}
textarea{width:100%;min-height:110px;font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;resize:vertical}
.suggest{background:#f5f6f8;border-radius:8px;padding:8px 10px;margin-bottom:8px;font-size:14px}
#msg{color:var(--err);font-size:14px;min-height:1em}
#save-state{font-size:13px}
kbd{font:600 11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;background:#fafbfc}
@media (max-width:900px){body{overflow:auto}.review{grid-template-columns:1fr;height:auto}#wrap{height:65vh;flex:none}}
</style>

<header>
  <div class="crumbs"><a href="{{ url_for('index') }}">Plates</a><span>›</span>
    <a href="{{ url_for('plate', plate=plate) }}">{{ plate }}</a><span>›</span><b>{{ slide }}</b></div>
  <span id="status-badge"></span>
  <div class="spacer"></div>
  <span id="save-state" class="muted"></span>
  <a class="logout" href="{{ url_for('logout') }}">Log out</a>
</header>

<div class="review">
  <section class="viewer-col">
    <div class="tools">
      <button type="button" id="btn-tiles" title="Zoom to all marked tiles">All tiles</button>
      <button type="button" id="btn-home" title="Show the whole slide">Whole slide</button>
      <span class="sep"></span>
      <label><input type="checkbox" id="show-d" checked> <span style="color:var(--dis)">■</span> Disease outlines</label>
      <label><input type="checkbox" id="show-h" checked> <span style="color:var(--hea)">■</span> Healthy outlines</label>
      <label><input type="checkbox" id="show-l" checked> Labels</label>
      <div class="spacer"></div>
      <span class="muted" style="color:#9aa3b2">scroll = zoom · drag = pan · <b>O</b> outlines on/off · <b>N</b>/<b>P</b> next/prev tile</span>
    </div>
    <div class="chips" id="chips"></div>
    <div id="wrap">
      {% if has_wsi %}
      <div id="osd"></div>
      <canvas id="overlay"></canvas>
      <div id="mag"></div>
      {% else %}
      <div id="missing"><div><b>The slide file for {{ slide }} was not found on the server.</b><br>
        You can still write notes; please let the researcher know.</div></div>
      {% endif %}
    </div>
  </section>

  <aside class="form-col">
    <form id="f" autocomplete="off">
      {% if n_d %}
      <div class="sec">
        <h3><span class="dot" style="background:var(--dis)"></span>Disease tiles ({{ n_d }})</h3>
        <p class="muted">What do you see? Do you agree they are disease? Refer to tiles by label, e.g. “d3, d7: mostly stroma”.</p>
        <textarea name="disease_tile_notes"></textarea>
      </div>
      {% endif %}
      {% if n_h %}
      <div class="sec">
        <h3><span class="dot" style="background:var(--hea)"></span>Healthy tiles ({{ n_h }})</h3>
        <p class="muted">What do you see? Do you agree they are healthy? Anything else?</p>
        <textarea name="healthy_tile_notes"></textarea>
      </div>
      {% endif %}
      <div class="sec">
        <h3>Diagnosis</h3>
        {% if suggested %}<div class="suggest"><span class="muted">Suggested:</span> <b>{{ suggested }}</b></div>{% endif %}
        <p class="muted">{{ 'Do you agree with the suggested diagnosis? Your diagnosis / comments.' if suggested else 'Your (suspected) diagnosis and comments.' }}</p>
        <textarea name="diagnosis"></textarea>
      </div>
    </form>
    <p id="msg"></p>
    <button type="button" id="submit" class="primary" style="width:100%">Submit &amp; mark slide as done</button>
    <p class="muted" style="font-size:13px">Notes are saved automatically as you type. You can come back and edit a submitted slide.</p>
  </aside>
</div>

{% if has_wsi %}<script src="{{ osd_js }}"></script>{% endif %}
<script>
const CFG = {{ cfg|tojson }};
const form = document.getElementById('f');
let timer = null, dirty = false;

// ---------------- form + saving ----------------
for (const k of CFG.note_fields) { const el = form.elements[k]; if (el) el.value = CFG.review[k] || ''; }

function setState(t) { document.getElementById('save-state').textContent = t; }
function setBadge(status) {
  const el = document.getElementById('status-badge');
  const m = {done: ['b-done', '✓ Done'], draft: ['b-draft', 'In progress']}[status] || ['b-todo', 'Not started'];
  el.className = 'badge ' + m[0]; el.textContent = m[1];
}
function payload(action) {
  const o = {action};
  for (const k of CFG.note_fields) { const el = form.elements[k]; o[k] = el ? el.value : ''; }
  return o;
}
async function post(action) {
  setState('Saving…');
  let r;
  try {
    r = await fetch(CFG.save_url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify(payload(action))});
  } catch (e) { setState('⚠ Not saved – connection problem'); return null; }
  if (r.status === 401) { location.href = CFG.login_url + '?next=' + encodeURIComponent(location.pathname); return null; }
  if (!r.ok) { setState('⚠ Not saved (error ' + r.status + ')'); return null; }
  const data = await r.json();
  dirty = false;
  setState('Saved ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}));
  setBadge(data.status);
  return data;
}
function schedule() {
  dirty = true; setState('Unsaved changes');
  clearTimeout(timer); timer = setTimeout(() => post('save'), 1200);
}
form.addEventListener('input', schedule);
form.addEventListener('submit', e => e.preventDefault());
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

document.getElementById('submit').onclick = async () => {
  const msg = document.getElementById('msg'), missing = [];
  const v = k => (form.elements[k] ? form.elements[k].value.trim() : 'n/a');
  if (!v('disease_tile_notes')) missing.push('disease tile notes');
  if (!v('healthy_tile_notes')) missing.push('healthy tile notes');
  if (!v('diagnosis')) missing.push('diagnosis');
  if (missing.length) { msg.textContent = 'Please fill in: ' + missing.join(', ') + '.'; return; }
  msg.textContent = '';
  clearTimeout(timer);
  const data = await post('done');
  if (data) location.href = data.next_url || CFG.plate_url;
};
setBadge(CFG.review.status);

// ---------------- tile chips ----------------
const tiles = CFG.tiles;            // [{id, group:'d'|'h', poly:[[x,y],...], bbox:[x0,y0,x1,y1]}]
let current = -1;                   // index of the tile last jumped to
const chips = document.getElementById('chips');
if (!tiles.length) chips.innerHTML = '<span style="color:#9aa3b2;font-size:13px">No marked tiles for this slide.</span>';
tiles.forEach((t, i) => {
  const c = document.createElement('span');
  c.className = 'chip ' + t.group;
  c.innerHTML = `<span class="dot" style="background:${t.group === 'd' ? 'var(--dis)' : 'var(--hea)'}"></span>${t.id}`;
  c.title = 'Zoom to ' + t.id;
  c.onclick = () => focusTile(i);
  chips.appendChild(c);
});
function markChip() {
  [...chips.querySelectorAll('.chip')].forEach((c, i) => c.classList.toggle('on', i === current));
  const a = chips.children[current];
  if (a && a.scrollIntoView) a.scrollIntoView({block: 'nearest', inline: 'nearest'});
}

// ---------------- viewer ----------------
if (CFG.has_wsi && typeof OpenSeadragon !== 'undefined') {
  const viewer = OpenSeadragon({
    id: 'osd',
    tileSources: CFG.dzi_url,
    showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT',
    showNavigationControl: false,
    animationTime: 0.6, blendTime: 0.1,
    minZoomImageRatio: 0.5, maxZoomPixelRatio: 4, visibilityRatio: 0.5,
    gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true},
  });
  const wrap = document.getElementById('wrap');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const show = {d: true, h: true, l: true};
  const COLORS = {d: '#ff2d2d', h: '#2d6cff'};
  const item = () => viewer.world.getItemAt(0);

  function rectAround(x0, y0, x1, y1, pad) {
    const w = x1 - x0, h = y1 - y0;
    return item().imageToViewportRectangle(x0 - pad * w, y0 - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad));
  }
  function fitAllTiles(immediate) {
    if (!tiles.length) { viewer.viewport.goHome(immediate); return; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const t of tiles) { x0 = Math.min(x0, t.bbox[0]); y0 = Math.min(y0, t.bbox[1]);
                             x1 = Math.max(x1, t.bbox[2]); y1 = Math.max(y1, t.bbox[3]); }
    const tw = Math.max(...tiles.map(t => t.bbox[2] - t.bbox[0]));
    const r = item().imageToViewportRectangle(x0 - 2 * tw, y0 - 2 * tw, (x1 - x0) + 4 * tw, (y1 - y0) + 4 * tw);
    viewer.viewport.fitBounds(r, immediate);
    current = -1; markChip();
  }
  window.focusTile = function (i) {
    if (!item()) return;
    current = i; markChip();
    const b = tiles[i].bbox;
    viewer.viewport.fitBounds(rectAround(b[0], b[1], b[2], b[3], 1.0));   // tile + 1 tile margin (3x3 field)
  };

  function draw() {
    const r = wrap.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
      canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    const it = item(); if (!it) return;
    const toScreen = (x, y) => it.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));

    tiles.forEach((t, i) => {
      if (!show[t.group]) return;
      ctx.beginPath();
      t.poly.forEach((p, j) => { const s = toScreen(p[0], p[1]); j ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.closePath();
      const hl = i === current;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'black'; ctx.lineWidth = hl ? 4 : 2.6; ctx.stroke();
      ctx.strokeStyle = COLORS[t.group]; ctx.lineWidth = hl ? 2.2 : 1.2; ctx.stroke();
      if (show.l) {
        const s = toScreen(t.bbox[0], t.bbox[1]);
        ctx.font = 'bold 14px system-ui, Arial, sans-serif';
        ctx.lineWidth = 3; ctx.strokeStyle = 'black'; ctx.strokeText(t.id, s.x, s.y - 5);
        ctx.fillStyle = COLORS[t.group]; ctx.fillText(t.id, s.x, s.y - 5);
      }
    });

    // magnification + scale bar
    const imgZoom = it.viewportToImageZoom(viewer.viewport.getZoom(true));   // screen px per level-0 px
    const mag = document.getElementById('mag');
    if (CFG.mpp) {
      const umPerPx = CFG.mpp / imgZoom;
      const nice = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
      const um = nice.find(n => n / umPerPx >= 70) || nice[nice.length - 1];
      const px = um / umPerPx, x = 12, y = r.height - 16;
      ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(x - 6, y - 22, px + 12, 32);
      ctx.fillStyle = '#fff'; ctx.fillRect(x, y, px, 4);
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.fillText(um >= 1000 ? (um / 1000) + ' mm' : um + ' µm', x, y - 6);
      const m = (10 / CFG.mpp) * imgZoom;   // ~ objective magnification (0.25 µm/px ≈ 40×)
      mag.textContent = '≈ ' + (m >= 10 ? Math.round(m) : m.toFixed(1)) + '×';
    } else {
      mag.textContent = Math.round(imgZoom * 100) + '%';
    }
  }

  viewer.addHandler('open', () => { fitAllTiles(true); draw(); });
  viewer.addHandler('update-viewport', draw);
  viewer.addHandler('resize', draw);
  viewer.addHandler('open-failed', () => {
    document.getElementById('mag').textContent = 'Could not load slide';
  });

  document.getElementById('btn-tiles').onclick = () => fitAllTiles(false);
  document.getElementById('btn-home').onclick = () => { current = -1; markChip(); viewer.viewport.goHome(); };
  for (const g of ['d', 'h', 'l']) {
    document.getElementById('show-' + g).onchange = e => { show[g] = e.target.checked; draw(); };
  }
  document.addEventListener('keydown', e => {
    if (['TEXTAREA', 'INPUT'].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'o') {
      const on = !(show.d || show.h);
      show.d = show.h = on;
      document.getElementById('show-d').checked = document.getElementById('show-h').checked = on;
      draw();
    } else if ((k === 'n' || k === 'p') && tiles.length) {
      const n = tiles.length;
      focusTile(k === 'n' ? (current + 1) % n : (current - 1 + n) % n);
    }
  });
} else {
  window.focusTile = () => {};
  document.querySelector('.tools').style.display = 'none';
}
</script>
"""


# ----------------------------------------------------------------------
# Slide access
# ----------------------------------------------------------------------
class SlideStore:
    """Finds slide files by name and keeps a few DeepZoom generators open."""

    def __init__(self, slide_root: Path, slide_map: Dict[str, str], max_open: int = 8):
        self.slide_root = slide_root
        self.slide_map = {k: Path(v).expanduser().resolve() for k, v in slide_map.items()}
        self.max_open = max_open
        self._index: Dict[str, List[Path]] = {}
        self._indexed_at = 0.0
        self._lock = threading.Lock()
        self._open: "OrderedDict[str, Tuple[DeepZoomGenerator, Optional[float]]]" = OrderedDict()

    def _build_index(self) -> None:
        index: Dict[str, List[Path]] = {}
        for p in self.slide_root.rglob("*"):
            if p.suffix.lower() in WSI_EXTENSIONS and p.is_file() and not any(
                part.startswith(".") for part in p.relative_to(self.slide_root).parts
            ):
                index.setdefault(p.stem, []).append(p)
        self._index = index
        self._indexed_at = time.time()

    def find(self, slide: str, plate_dir: Optional[Path] = None) -> Optional[Path]:
        if slide in self.slide_map:
            p = self.slide_map[slide]
            return p if p.is_file() else None
        with self._lock:
            if not self._indexed_at or (slide not in self._index and time.time() - self._indexed_at > 60):
                self._build_index()   # re-scan for newly added slides, at most once a minute
            cands = self._index.get(slide, [])
        if not cands:
            return None
        if plate_dir is not None:
            for c in cands:  # prefer a file inside the plate's own folder
                try:
                    c.relative_to(plate_dir)
                    return c
                except ValueError:
                    pass
        return cands[0]

    def deepzoom(self, path: Path) -> Tuple[DeepZoomGenerator, Optional[float]]:
        key = str(path)
        with self._lock:
            if key in self._open:
                self._open.move_to_end(key)
                return self._open[key]
            slide = openslide.OpenSlide(key)
            # limit_bounds=False keeps level-0 coordinates identical to the
            # polygons saved by interactive_viewer.py.
            dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)
            try:
                mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
            except (TypeError, ValueError):
                mpp = None
            self._open[key] = (dz, mpp)
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)   # closed when garbage-collected
            return dz, mpp


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
def create_app(
    data_root: str,
    csv_path: str,
    password_hash: str,
    slide_root: Optional[str] = None,
    slide_map_csv: Optional[str] = None,
    diagnoses_csv: Optional[str] = None,
    secure_cookies: bool = False,
    secret_key: Optional[str] = None,
) -> Flask:
    root = Path(data_root).expanduser().resolve()
    csv_file = Path(csv_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    wsi_root = Path(slide_root).expanduser().resolve() if slide_root else root
    if not wsi_root.is_dir():
        raise FileNotFoundError(f"Slide root does not exist: {wsi_root}")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    slides_store = SlideStore(
        wsi_root, _load_two_col(slide_map_csv, "slide_path") if slide_map_csv else {}
    )
    suggested = _load_two_col(diagnoses_csv, "diagnosis") if diagnoses_csv else {}
    lock = threading.Lock()
    failed_logins: Dict[str, List[float]] = {}
    ann_cache: Dict[str, Tuple[float, List[Dict]]] = {}

    # Local copy of OpenSeadragon (static/openseadragon.min.js next to this
    # script, same as interactive_viewer.py) if present, otherwise the CDN.
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=str(static_dir))
    osd_js = (
        "/static/openseadragon.min.js"
        if (static_dir / "openseadragon.min.js").is_file()
        else "https://cdn.jsdelivr.net/npm/openseadragon@5.0.1/build/openseadragon/openseadragon.min.js"
    )
    app.config.update(
        SECRET_KEY=secret_key or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=secure_cookies,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    )

    # ---------------- data helpers ----------------
    def list_plates() -> List[str]:
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and (p / "annotations").is_dir()
            and any((p / "annotations").glob("*" + ANN_SUFFIX))
        )

    def list_slides(plate: str) -> List[str]:
        d = root / plate / "annotations"
        return sorted(f.name[: -len(ANN_SUFFIX)] for f in d.glob("*" + ANN_SUFFIX) if not f.name.startswith("."))

    def check(plate: str, slide: str) -> None:
        # Only names discovered on disk are accepted -> no path traversal.
        if plate not in list_plates() or slide not in list_slides(plate):
            abort(404)

    def load_tiles(plate: str, slide: str) -> List[Dict]:
        f = root / plate / "annotations" / f"{slide}{ANN_SUFFIX}"
        mtime = f.stat().st_mtime
        hit = ann_cache.get(str(f))
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            data = {}
        tiles = []
        # annotations use `disease_polygons` / `healthy_polygons` keys
        for prefix, key in (("d", "disease_polygons"), ("h", "healthy_polygons")):
            for i, poly in enumerate(data.get(key) or [], start=1):
                pts = [[float(p[0]), float(p[1])] for p in poly if len(p) >= 2]
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                tiles.append({"id": f"{prefix}{i}", "group": prefix, "poly": pts,
                              "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        ann_cache[str(f)] = (mtime, tiles)
        return tiles

    def wsi_path(plate: str, slide: str) -> Optional[Path]:
        return slides_store.find(slide, root / plate)

    def load_reviews() -> Dict[Tuple[str, str], Dict[str, str]]:
        if not csv_file.exists():
            return {}
        with csv_file.open(newline="", encoding="utf-8-sig") as fh:
            return {(r["plate"], r["slide_id"]): r for r in csv.DictReader(fh)}

    def write_reviews(rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
        tmp = csv_file.with_suffix(csv_file.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel-friendly (æøå)
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for key in sorted(rows):
                w.writerow({k: rows[key].get(k, "") for k in CSV_FIELDS})
        os.replace(tmp, csv_file)  # atomic: the CSV is never half-written

    def next_todo(plate: str, after: Optional[str], reviews) -> Optional[str]:
        slides = list_slides(plate)
        start = slides.index(after) + 1 if after in slides else 0
        for s in slides[start:] + slides[:start]:
            if s != after and reviews.get((plate, s), {}).get("status") != "done":
                return s
        return None

    def page(title: str, tpl: str, **ctx) -> str:
        return render_template_string(LAYOUT, title=title, body=render_template_string(tpl, **ctx))

    def now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- auth ----------------
    def login_required(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if not session.get("auth"):
                if request.path.startswith(("/api/", "/dzi/")):
                    return jsonify({"error": "not logged in"}), 401
                return redirect(url_for("login", next=request.path))
            return view(*a, **kw)
        return wrapped

    @app.after_request
    def headers(resp):
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        if request.path.startswith("/dzi/") and resp.status_code == 200:
            resp.headers["Cache-Control"] = "private, max-age=3600"
        elif not request.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            ip = request.remote_addr or "?"
            t = time.time()
            recent = [x for x in failed_logins.get(ip, []) if t - x < 900]
            if len(recent) >= 8:
                error = "Too many attempts. Please wait 15 minutes."
            elif check_password_hash(password_hash, request.form.get("password", "")):
                failed_logins.pop(ip, None)
                session.clear()
                session["auth"] = True
                session.permanent = True
                nxt = request.args.get("next", "")
                if not nxt.startswith("/") or nxt.startswith("//"):
                    nxt = url_for("index")
                return redirect(nxt)
            else:
                recent.append(t)
                time.sleep(1.0)
                error = "Wrong password."
            failed_logins[ip] = recent
        return page("Log in", LOGIN, error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ---------------- pages ----------------
    @app.route("/")
    @login_required
    def index():
        reviews = load_reviews()
        plates = []
        for p in list_plates():
            slides = list_slides(p)
            done = sum(reviews.get((p, s), {}).get("status") == "done" for s in slides)
            plates.append({"name": p, "total": len(slides), "done": done})
        return page("Plates", PLATES, plates=plates)

    @app.route("/plate/<plate>")
    @login_required
    def plate(plate):
        if plate not in list_plates():
            abort(404)
        reviews = load_reviews()
        slides = []
        for s in list_slides(plate):
            tiles = load_tiles(plate, s)
            r = reviews.get((plate, s), {})
            slides.append({
                "name": s,
                "n_d": sum(t["group"] == "d" for t in tiles),
                "n_h": sum(t["group"] == "h" for t in tiles),
                "has_wsi": wsi_path(plate, s) is not None,
                "status": r.get("status", ""),
                "updated_at": r.get("updated_at", ""),
            })
        done = sum(s["status"] == "done" for s in slides)
        return page(plate, PLATE, plate=plate, slides=slides, done=done,
                    next_todo=next_todo(plate, None, reviews))

    @app.route("/slide/<plate>/<slide>")
    @login_required
    def slide(plate, slide):
        check(plate, slide)
        tiles = load_tiles(plate, slide)
        path = wsi_path(plate, slide)
        mpp = None
        if path is not None:
            try:
                _, mpp = slides_store.deepzoom(path)
            except openslide.OpenSlideError:
                path = None
        review = load_reviews().get((plate, slide), {})
        sugg = suggested.get(slide, "")
        cfg = {
            "tiles": tiles,
            "has_wsi": path is not None,
            "mpp": mpp,
            "dzi_url": url_for("dzi", plate=plate, slide=slide),
            "review": {k: review.get(k, "") for k in CSV_FIELDS},
            "note_fields": NOTE_FIELDS,
            "save_url": url_for("save_review", plate=plate, slide=slide),
            "plate_url": url_for("plate", plate=plate),
            "login_url": url_for("login"),
        }
        return page(slide, SLIDE, plate=plate, slide=slide, cfg=cfg, osd_js=osd_js,
                    has_wsi=path is not None, suggested=sugg,
                    n_d=sum(t["group"] == "d" for t in tiles),
                    n_h=sum(t["group"] == "h" for t in tiles))

    # ---------------- deep zoom ----------------
    def get_dz(plate: str, slide: str) -> DeepZoomGenerator:
        check(plate, slide)
        path = wsi_path(plate, slide)
        if path is None:
            abort(404)
        return slides_store.deepzoom(path)[0]

    @app.route("/dzi/<plate>/<slide>.dzi")
    @login_required
    def dzi(plate, slide):
        return Response(get_dz(plate, slide).get_dzi("jpeg"), mimetype="application/xml")

    @app.route("/dzi/<plate>/<slide>_files/<int:level>/<int:col>_<int:row>.jpeg")
    @login_required
    def dzi_tile(plate, slide, level, col, row):
        dz = get_dz(plate, slide)
        try:
            img = dz.get_tile(level, (col, row))
        except ValueError:
            abort(404)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        return Response(buf.getvalue(), mimetype="image/jpeg")

    # ---------------- saving ----------------
    @app.route("/api/review/<plate>/<slide>", methods=["POST"])
    @login_required
    def save_review(plate, slide):
        check(plate, slide)
        if not request.is_json:
            abort(400)
        data = request.get_json(silent=True) or {}
        action = data.get("action")
        if action not in ("save", "done"):
            abort(400)

        with lock:
            rows = load_reviews()
            prev = rows.get((plate, slide), {})
            status = "done" if action == "done" else (prev.get("status") or "draft")
            ts = now()
            row = {
                "plate": plate,
                "slide_id": slide,
                "status": status,
                "suggested_diagnosis": suggested.get(slide, ""),
                "updated_at": ts,
                "completed_at": prev.get("completed_at") or (ts if status == "done" else ""),
            }
            for k in NOTE_FIELDS:
                row[k] = str(data.get(k, "")).strip()[:20000]
            rows[(plate, slide)] = row
            write_reviews(rows)

        nxt = next_todo(plate, slide, rows) if action == "done" else None
        return jsonify({
            "status": status,
            "next_url": url_for("slide", plate=plate, slide=nxt) if nxt else None,
        })

    return app


def _load_two_col(path: str, value_col: str) -> Dict[str, str]:
    """CSV with columns slide_id,<value_col> -> {slide_id: value}."""
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("slide_id") or r.get("slide") or "").strip()
            if sid:
                out[sid] = (r.get(value_col) or "").strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pathologist whole-slide review app")
    ap.add_argument("--data-root", help="Folder containing the plate folders (<plate>/annotations/...)")
    ap.add_argument("--slide-root", default=None,
                    help="Folder searched (recursively) for the slide files, matched by file name. Default: --data-root")
    ap.add_argument("--slide-map", default=None, help="Optional CSV: slide_id,slide_path (overrides the search)")
    ap.add_argument("--csv", default="pathologist_reviews.csv", help="Where reviews are stored")
    ap.add_argument("--diagnoses", default=None, help="Optional CSV: slide_id,diagnosis (suggested diagnosis)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--https", action="store_true",
                    help="Mark cookies Secure. Use whenever the app is reached over HTTPS (tunnel/reverse proxy).")
    ap.add_argument("--behind-proxy", action="store_true",
                    help="Trust X-Forwarded-* from one proxy (tunnel/reverse proxy) for client IPs.")
    ap.add_argument("--hash-password", action="store_true",
                    help="Print a password hash for REVIEW_PASSWORD_HASH and exit.")
    args = ap.parse_args()

    if args.hash_password:
        pw = getpass.getpass("New password: ")
        if pw != getpass.getpass("Repeat: "):
            sys.exit("Passwords do not match.")
        print(generate_password_hash(pw))
        return

    if not args.data_root:
        ap.error("--data-root is required")

    pw_hash = os.environ.get("REVIEW_PASSWORD_HASH")
    if not pw_hash:
        pw = getpass.getpass("Set access password for this session: ")
        if len(pw) < 10:
            sys.exit("Use at least 10 characters – this protects patient data.")
        pw_hash = generate_password_hash(pw)

    app = create_app(
        data_root=args.data_root,
        csv_path=args.csv,
        password_hash=pw_hash,
        slide_root=args.slide_root,
        slide_map_csv=args.slide_map,
        diagnoses_csv=args.diagnoses,
        secure_cookies=args.https,
        secret_key=os.environ.get("REVIEW_SECRET_KEY"),
    )
    if args.behind_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    print(f"Reviews CSV: {Path(args.csv).resolve()}")
    print(f"Serving on http://{args.host}:{args.port}/")
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        print("waitress not installed – using Flask's development server (pip install waitress).")
        app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
