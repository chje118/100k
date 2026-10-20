r"""
Pathologist review app: whole-slide viewer + notes.

FOR THE PATHOLOGIST
    Double-click the "Slide review" icon on the desktop. The app starts in the
    background (no black terminal window) and opens in the web browser.
    Choose a project (e.g. Lymphoma / Endometrium), then a plate, then a slide.
    When finished, click "Close app" (top right). Notes are saved automatically.
    If the browser tab is just closed instead, the app stops by itself after a
    while (see "auto_close_minutes"). Double-clicking the icon while the app is
    already running simply opens it again.

ONE-TIME SET-UP (researcher)
    1. Edit review_projects.json next to this file: one entry per project.
    2. From the Python environment that has openslide / flask / waitress:
           python make_shortcut.py
       -> creates the "Slide review" shortcut on the desktop.
    Errors are written to review_app.log next to this file (and shown in a
    pop-up if the app cannot start at all).

Expected project folder (one per project):

    <folder>/<plate>/annotations/<slide>_annotations.json   (from interactive_viewer.py)
    <folder>/slides/...                   slide files, found by file name (searched recursively)
    <folder>/pathologist_reviews.csv      one row per slide, created / updated by the app

Developer use from a terminal still works:
    python review_app.py                                 # projects from review_projects.json
    python review_app.py E:\Christine\100k_lymphoma      # one ad-hoc project folder
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_ID = "slide-review"
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "review_app.log"
CONFIG_FILE = HERE / "review_projects.json"


# ----------------------------------------------------------------------
# Running without a terminal (pythonw.exe)
# ----------------------------------------------------------------------
def _setup_logging() -> None:
    """pythonw has no console (sys.stdout/stderr are None): send output to a log file."""
    if sys.stdout is None or sys.stderr is None:
        try:
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5_000_000:
                LOG_FILE.replace(LOG_FILE.with_name(LOG_FILE.name + ".old"))
            fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        except OSError:
            fh = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = fh
        if sys.stderr is None:
            sys.stderr = fh
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _show_error(msg: str) -> None:
    """Log the error and, since there may be no terminal, also show a pop-up."""
    logging.error(msg)
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Slide review", msg)
        root.destroy()
    except Exception:
        pass


_setup_logging()

# A conda environment started without "conda activate" (as a desktop shortcut
# does) may not find the OpenSlide DLLs: add the environment's Library\bin.
if sys.platform == "win32":
    _lib_bin = Path(sys.prefix) / "Library" / "bin"
    if _lib_bin.is_dir():
        os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(_lib_bin))
        except (AttributeError, OSError):
            pass

try:
    import openslide
    from flask import (
        Flask,
        Response,
        abort,
        jsonify,
        redirect,
        render_template_string,
        request,
        url_for,
    )
    from openslide.deepzoom import DeepZoomGenerator
except (ImportError, OSError) as _exc:
    _show_error(
        f"Slide review could not start:\n\n{_exc}\n\n"
        "The shortcut must use the Python environment that has openslide and "
        "flask installed. Re-run make_shortcut.py from that environment."
    )
    sys.exit(1)


WSI_EXTENSIONS = {
    ".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn",
    ".vms", ".vmu", ".bif", ".svslide", ".dcm",
}

CSV_FIELDS = [
    "plate",
    "slide_id",
    "status",
    "disease_tile_notes",
    "healthy_tile_notes",
    "suggested_diagnosis",
    "updated_at",
    "completed_at",
]
NOTE_FIELDS = ["disease_tile_notes", "healthy_tile_notes", "suggested_diagnosis"]

# Quick-pick buttons above the suggested-diagnosis field (override per project
# with "diagnoses" in review_projects.json).
DEFAULT_DIAGNOSES = ["FL", "DLBCL", "cHL", "Nodular hyperplasia"]

ANN_SUFFIX = "_annotations.json"

QUIT_BTN = '<button type="button" class="quit" onclick="quitApp()" title="Stop the review app">Close app</button>'


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
.spacer{flex:1}
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
.quit{border-color:#e4b4b7;color:var(--err)}.quit:hover{background:#fde8e9;border-color:var(--err)}
.muted{color:var(--muted)}
.top{display:flex;align-items:end;gap:12px;margin-bottom:16px}
.closed{text-align:center;padding-top:20vh}
</style>
</head><body>{{ body|safe }}
<script>
// Keep-alive: the app stops by itself some minutes after the last page is closed.
const APP_PING = {{ ping_url|tojson }}, APP_QUIT = {{ quit_url|tojson }};
function appPing() { fetch(APP_PING, {method: 'POST'}).catch(() => {}); }
appPing();
const appPingTimer = setInterval(appPing, 60000);

async function quitApp() {
  // the slide page saves any unsaved notes first
  if (window.beforeQuit && !(await window.beforeQuit())
      && !confirm('Your latest notes could not be saved. Close the app anyway?')) return;
  try {
    await fetch(APP_QUIT, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) {}
  clearInterval(appPingTimer);
  window.beforeQuit = null;
  document.body.innerHTML = '<main class="closed"><h1>Slide review is closed</h1>' +
    '<p class="muted">Your notes are saved. You can close this browser tab.</p>' +
    '<p class="muted">To review again, double-click the Slide review icon.</p></main>';
  try { window.close(); } catch (e) {}
}
</script>
</body></html>"""


PROJECTS = r"""
<header>
  <div class="crumbs"><b>Projects</b></div><div class="spacer"></div>{{ quit_btn|safe }}
</header>
<main>
  <div class="top"><div><h1>Slide review</h1><div class="muted">Choose a project.</div></div></div>
  <div class="card"><table>
    <tr><th>Project</th><th>Plates</th><th>Slides</th><th>Progress</th></tr>
    {% for p in projects %}
    {% if p.ok %}
    <tr class="row" onclick="location.href='{{ url_for('plates_page', proj=p.key) }}'">
      <td><a href="{{ url_for('plates_page', proj=p.key) }}">{{ p.name }}</a></td>
      <td>{{ p.plates }}</td>
      <td>{{ p.total }}</td>
      <td><span class="bar"><i style="width:{{ (100 * p.done / p.total) if p.total else 0 }}%"></i></span>
          <span class="muted">{{ p.done }} / {{ p.total }} done</span></td>
    </tr>
    {% else %}
    <tr>
      <td><b>{{ p.name }}</b> <span class="badge b-err">{{ p.error }}</span></td>
      <td colspan="3" class="muted">{{ p.folder }} – please let the researcher know.</td>
    </tr>
    {% endif %}
    {% else %}
    <tr><td colspan="4" class="muted">No projects configured.</td></tr>
    {% endfor %}
  </table></div>
</main>
"""


PLATES = r"""
<header>
  <div class="crumbs"><a href="{{ url_for('home') }}">Projects</a><span>›</span><b>{{ project.name }}</b></div>
  <div class="spacer"></div>{{ quit_btn|safe }}
</header>
<main>
  <div class="top"><div><h1>{{ project.name }}</h1><div class="muted">Choose a plate to see its slides.</div></div></div>
  <div class="card"><table>
    <tr><th>Plate</th><th>Slides</th><th>Progress</th></tr>
    {% for p in plates %}
    <tr class="row" onclick="location.href='{{ url_for('plate_page', proj=project.key, plate=p.name) }}'">
      <td><a href="{{ url_for('plate_page', proj=project.key, plate=p.name) }}">{{ p.name }}</a></td>
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
  <div class="crumbs"><a href="{{ url_for('home') }}">Projects</a><span>›</span>
    <a href="{{ url_for('plates_page', proj=project.key) }}">{{ project.name }}</a><span>›</span><b>{{ plate }}</b></div>
  <div class="spacer"></div>{{ quit_btn|safe }}
</header>
<main>
  <div class="top">
    <div><h1>{{ plate }}</h1><div class="muted">{{ done }} of {{ slides|length }} slides done</div></div>
    <div class="spacer"></div>
    {% if next_todo %}<a class="btn primary" href="{{ url_for('slide_page', proj=project.key, plate=plate, slide=next_todo) }}">Next unreviewed slide →</a>{% endif %}
  </div>
  <div class="card"><table>
    <tr><th>Slide</th><th>Disease tiles</th><th>Healthy tiles</th><th>Status</th><th>Last saved</th></tr>
    {% for s in slides %}
    <tr class="row" onclick="location.href='{{ url_for('slide_page', proj=project.key, plate=plate, slide=s.name) }}'">
      <td><a href="{{ url_for('slide_page', proj=project.key, plate=plate, slide=s.name) }}">{{ s.name }}</a>
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
.picks{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.picks button{font-size:13px;padding:4px 11px;border-radius:999px}
.picks button.on{background:#e8eefc;border-color:var(--accent);color:var(--accent);font-weight:600}
#dx{width:100%;font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin-bottom:8px}
#msg{color:var(--err);font-size:14px;min-height:1em}
#save-state{font-size:13px}
kbd{font:600 11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;background:#fafbfc}
@media (max-width:900px){body{overflow:auto}.review{grid-template-columns:1fr;height:auto}#wrap{height:65vh;flex:none}}
</style>

<header>
  <div class="crumbs"><a href="{{ url_for('home') }}">Projects</a><span>›</span>
    <a href="{{ url_for('plates_page', proj=project.key) }}">{{ project.name }}</a><span>›</span>
    <a href="{{ url_for('plate_page', proj=project.key, plate=plate) }}">{{ plate }}</a><span>›</span><b>{{ slide }}</b></div>
  <span id="status-badge"></span>
  <div class="spacer"></div>
  <span id="save-state" class="muted"></span>
  {{ quit_btn|safe }}
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
        <p class="muted">What do you see? Do you agree they are disease?</p>
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
        <h3>Suggested diagnosis</h3>
        <p class="muted">Pick one or type your own.</p>
        {% if diagnoses %}<div class="picks" id="picks">
          {% for d in diagnoses %}<button type="button" data-dx="{{ d }}">{{ d }}</button>{% endfor %}
        </div>{% endif %}
        <input id="dx" name="suggested_diagnosis" type="text" placeholder="Suggested diagnosis">
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

// "Close app": save pending notes first (true = nothing left unsaved)
window.beforeQuit = async () => {
  clearTimeout(timer);
  if (!dirty) return true;
  return (await post('save')) !== null;
};

document.getElementById('submit').onclick = async () => {
  const msg = document.getElementById('msg'), missing = [];
  const v = k => (form.elements[k] ? form.elements[k].value.trim() : 'n/a');
  if (!v('disease_tile_notes')) missing.push('disease tile notes');
  if (!v('healthy_tile_notes')) missing.push('healthy tile notes');
  if (!v('suggested_diagnosis')) missing.push('suggested diagnosis');
  if (missing.length) { msg.textContent = 'Please fill in: ' + missing.join(', ') + '.'; return; }
  msg.textContent = '';
  clearTimeout(timer);
  const data = await post('done');
  if (data) location.href = data.next_url || CFG.plate_url;
};
setBadge(CFG.review.status);

// quick-pick diagnosis buttons fill the text field
const dx = form.elements['suggested_diagnosis'];
const picks = [...document.querySelectorAll('#picks [data-dx]')];
function markPicks() { picks.forEach(b => b.classList.toggle('on', b.dataset.dx === dx.value.trim())); }
picks.forEach(b => b.onclick = () => { dx.value = b.dataset.dx; markPicks(); schedule(); });
dx.addEventListener('input', markPicks);
markPicks();

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
        if self.slide_root.is_dir():
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
# One review project (e.g. lymphoma, endometrium)
# ----------------------------------------------------------------------
class ReviewProject:
    def __init__(
        self,
        key: str,
        name: str,
        folder: str,
        slide_root: Optional[str] = None,
        csv_path: Optional[str] = None,
        slide_map_csv: Optional[str] = None,
        diagnoses: Optional[List[str]] = None,
    ):
        self.key = key
        self.name = name
        self.root = Path(folder).expanduser().resolve()
        self.csv_file = (Path(csv_path).expanduser().resolve() if csv_path
                         else self.root / "pathologist_reviews.csv")
        if slide_root:
            wsi_root = Path(slide_root).expanduser().resolve()
        else:
            wsi_root = self.root / "slides" if (self.root / "slides").is_dir() else self.root
        self.wsi_root = wsi_root
        slide_map = _load_two_col(slide_map_csv, "slide_path") if slide_map_csv else {}
        self.slides = SlideStore(wsi_root, slide_map)
        self.diagnoses = list(DEFAULT_DIAGNOSES if diagnoses is None else diagnoses)
        self.lock = threading.Lock()
        self._ann_cache: Dict[str, Tuple[float, List[Dict]]] = {}

    @property
    def ok(self) -> bool:
        # checked live, so a drive that is connected later is picked up
        return self.root.is_dir()

    # ---------------- data helpers ----------------
    def list_plates(self) -> List[str]:
        if not self.ok:
            return []
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and (p / "annotations").is_dir()
            and any((p / "annotations").glob("*" + ANN_SUFFIX))
        )

    def list_slides(self, plate: str) -> List[str]:
        d = self.root / plate / "annotations"
        return sorted(f.name[: -len(ANN_SUFFIX)] for f in d.glob("*" + ANN_SUFFIX) if not f.name.startswith("."))

    def check(self, plate: str, slide: Optional[str] = None) -> None:
        # Only names discovered on disk are accepted -> no path traversal.
        if plate not in self.list_plates() or (slide is not None and slide not in self.list_slides(plate)):
            abort(404)

    def load_tiles(self, plate: str, slide: str) -> List[Dict]:
        f = self.root / plate / "annotations" / f"{slide}{ANN_SUFFIX}"
        mtime = f.stat().st_mtime
        hit = self._ann_cache.get(str(f))
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logging.warning("Could not read %s", f)
            data = {}
        tiles = []
        for prefix, key in (("d", "disease_polygons"), ("h", "healthy_polygons")):
            for i, poly in enumerate(data.get(key) or [], start=1):
                pts = [[float(p[0]), float(p[1])] for p in poly if len(p) >= 2]
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                tiles.append({"id": f"{prefix}{i}", "group": prefix, "poly": pts,
                              "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        self._ann_cache[str(f)] = (mtime, tiles)
        return tiles

    def wsi_path(self, plate: str, slide: str) -> Optional[Path]:
        return self.slides.find(slide, self.root / plate)

    def load_reviews(self) -> Dict[Tuple[str, str], Dict[str, str]]:
        if not self.csv_file.exists():
            return {}
        rows: Dict[Tuple[str, str], Dict[str, str]] = {}
        with self.csv_file.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
                sid = r.get("slide_id", "")
                if not sid:
                    continue
                key = (r.get("plate", ""), sid)
                if not key[0]:
                    # Row added by hand without a plate: attach it to the plate that has this slide.
                    for p in self.list_plates():
                        if sid in self.list_slides(p):
                            key = (p, sid)
                            r["plate"] = p
                            break
                rows[key] = {**rows.get(key, {}), **{k: v for k, v in r.items() if v}}
        return rows

    def write_reviews(self, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_file.with_suffix(self.csv_file.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel-friendly (æøå)
            # keep any extra columns added by hand, after the standard ones
            extra = []
            for r in rows.values():
                extra += [k for k in r if k not in CSV_FIELDS and k not in extra]
            fields = CSV_FIELDS + extra
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for key in sorted(rows):
                w.writerow({k: rows[key].get(k, "") for k in fields})
        os.replace(tmp, self.csv_file)  # atomic: the CSV is never half-written

    def next_todo(self, plate: str, after: Optional[str], reviews) -> Optional[str]:
        slides = self.list_slides(plate)
        start = slides.index(after) + 1 if after in slides else 0
        for s in slides[start:] + slides[:start]:
            if s != after and reviews.get((plate, s), {}).get("status") != "done":
                return s
        return None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
def create_app(projects: "OrderedDict[str, ReviewProject]", auto_close_minutes: float = 15) -> Flask:
    # Local copy of OpenSeadragon (static/openseadragon.min.js next to this
    # script, same as interactive_viewer.py) if present, otherwise the CDN.
    static_dir = HERE / "static"
    app = Flask(__name__, static_folder=str(static_dir))
    osd_js = (
        "/static/openseadragon.min.js"
        if (static_dir / "openseadragon.min.js").is_file()
        else "https://cdn.jsdelivr.net/npm/openseadragon@5.0.1/build/openseadragon/openseadragon.min.js"
    )
    activity = {"last": time.time()}

    def get_project(proj: str) -> ReviewProject:
        p = projects.get(proj)
        if p is None or not p.ok:
            abort(404)
        return p

    def page(title: str, tpl: str, **ctx) -> str:
        body = render_template_string(tpl, quit_btn=QUIT_BTN, **ctx)
        return render_template_string(LAYOUT, title=title, body=body,
                                      ping_url=url_for("api_ping"), quit_url=url_for("api_quit"))

    def now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- headers ----------------
    @app.before_request
    def touch():
        activity["last"] = time.time()

    @app.after_request
    def headers(resp):
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        if "/dzi/" in request.path and resp.status_code == 200:
            resp.headers["Cache-Control"] = "private, max-age=3600"
        elif not request.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---------------- pages ----------------
    @app.route("/")
    def home():
        if len(projects) == 1:
            only = next(iter(projects.values()))
            if only.ok:
                return redirect(url_for("plates_page", proj=only.key))
        rows = []
        for p in projects.values():
            if not p.ok:
                rows.append({"key": p.key, "name": p.name, "ok": False,
                             "error": "folder not found", "folder": str(p.root)})
                continue
            reviews = p.load_reviews()
            plates = p.list_plates()
            total = done = 0
            for pl in plates:
                slides = p.list_slides(pl)
                total += len(slides)
                done += sum(reviews.get((pl, s), {}).get("status") == "done" for s in slides)
            rows.append({"key": p.key, "name": p.name, "ok": True,
                         "plates": len(plates), "total": total, "done": done})
        return page("Projects", PROJECTS, projects=rows)

    @app.route("/p/<proj>/")
    def plates_page(proj):
        project = get_project(proj)
        reviews = project.load_reviews()
        plates = []
        for p in project.list_plates():
            slides = project.list_slides(p)
            done = sum(reviews.get((p, s), {}).get("status") == "done" for s in slides)
            plates.append({"name": p, "total": len(slides), "done": done})
        return page(project.name, PLATES, project=project, plates=plates)

    @app.route("/p/<proj>/plate/<plate>")
    def plate_page(proj, plate):
        project = get_project(proj)
        project.check(plate)
        reviews = project.load_reviews()
        slides = []
        for s in project.list_slides(plate):
            tiles = project.load_tiles(plate, s)
            r = reviews.get((plate, s), {})
            slides.append({
                "name": s,
                "n_d": sum(t["group"] == "d" for t in tiles),
                "n_h": sum(t["group"] == "h" for t in tiles),
                "has_wsi": project.wsi_path(plate, s) is not None,
                "status": r.get("status", ""),
                "updated_at": r.get("updated_at", ""),
            })
        done = sum(s["status"] == "done" for s in slides)
        return page(plate, PLATE, project=project, plate=plate, slides=slides, done=done,
                    next_todo=project.next_todo(plate, None, reviews))

    @app.route("/p/<proj>/slide/<plate>/<slide>")
    def slide_page(proj, plate, slide):
        project = get_project(proj)
        project.check(plate, slide)
        tiles = project.load_tiles(plate, slide)
        path = project.wsi_path(plate, slide)
        mpp = None
        if path is not None:
            try:
                _, mpp = project.slides.deepzoom(path)
            except openslide.OpenSlideError:
                logging.exception("Could not open slide %s", path)
                path = None
        review = project.load_reviews().get((plate, slide), {})
        cfg = {
            "tiles": tiles,
            "has_wsi": path is not None,
            "mpp": mpp,
            "dzi_url": url_for("dzi", proj=proj, plate=plate, slide=slide),
            "review": {k: review.get(k, "") for k in CSV_FIELDS},
            "note_fields": NOTE_FIELDS,
            "save_url": url_for("save_review", proj=proj, plate=plate, slide=slide),
            "plate_url": url_for("plate_page", proj=proj, plate=plate),
        }
        return page(slide, SLIDE, project=project, plate=plate, slide=slide, cfg=cfg, osd_js=osd_js,
                    has_wsi=path is not None, diagnoses=project.diagnoses,
                    n_d=sum(t["group"] == "d" for t in tiles),
                    n_h=sum(t["group"] == "h" for t in tiles))

    # ---------------- deep zoom ----------------
    def get_dz(proj: str, plate: str, slide: str) -> DeepZoomGenerator:
        project = get_project(proj)
        project.check(plate, slide)
        path = project.wsi_path(plate, slide)
        if path is None:
            abort(404)
        return project.slides.deepzoom(path)[0]

    @app.route("/p/<proj>/dzi/<plate>/<slide>.dzi")
    def dzi(proj, plate, slide):
        return Response(get_dz(proj, plate, slide).get_dzi("jpeg"), mimetype="application/xml")

    @app.route("/p/<proj>/dzi/<plate>/<slide>_files/<int:level>/<int:col>_<int:row>.jpeg")
    def dzi_tile(proj, plate, slide, level, col, row):
        dz = get_dz(proj, plate, slide)
        try:
            img = dz.get_tile(level, (col, row))
        except ValueError:
            abort(404)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        return Response(buf.getvalue(), mimetype="image/jpeg")

    # ---------------- saving ----------------
    @app.route("/p/<proj>/api/review/<plate>/<slide>", methods=["POST"])
    def save_review(proj, plate, slide):
        project = get_project(proj)
        project.check(plate, slide)
        if not request.is_json:
            abort(400)
        data = request.get_json(silent=True) or {}
        action = data.get("action")
        if action not in ("save", "done"):
            abort(400)

        with project.lock:
            rows = project.load_reviews()
            prev = rows.get((plate, slide), {})
            status = "done" if action == "done" else (prev.get("status") or "draft")
            ts = now()
            row = {
                **prev,
                "plate": plate,
                "slide_id": slide,
                "status": status,
                "updated_at": ts,
                "completed_at": prev.get("completed_at") or (ts if status == "done" else ""),
            }
            for k in NOTE_FIELDS:
                row[k] = str(data.get(k, "")).strip()[:20000]
            rows[(plate, slide)] = row
            project.write_reviews(rows)

        nxt = project.next_todo(plate, slide, rows) if action == "done" else None
        return jsonify({
            "status": status,
            "next_url": url_for("slide_page", proj=proj, plate=plate, slide=nxt) if nxt else None,
        })

    # ---------------- app lifetime ----------------
    @app.route("/api/ping", methods=["GET", "POST"])
    def api_ping():
        return jsonify({"app": APP_ID})

    @app.route("/api/quit", methods=["POST"])
    def api_quit():
        # JSON-only, so another website open in the browser cannot stop the app
        if not request.is_json:
            abort(400)
        logging.info("Close requested from the browser.")
        _shutdown_soon(projects)
        return jsonify({"ok": True})

    _start_idle_watchdog(activity, projects, auto_close_minutes)
    return app


def _shutdown_soon(projects: "OrderedDict[str, ReviewProject]", delay: float = 0.5) -> None:
    """Stop the whole process shortly (after the HTTP response has gone out)."""
    def run():
        time.sleep(delay)
        for p in projects.values():
            p.lock.acquire()          # wait for any CSV write in progress
        logging.info("Slide review stopped.")
        logging.shutdown()
        os._exit(0)
    threading.Thread(target=run, daemon=True).start()


def _start_idle_watchdog(activity: Dict[str, float], projects, minutes: float) -> None:
    """Every open page pings once a minute. With no page open for `minutes`, stop the app."""
    if not minutes or minutes <= 0:
        return

    def run():
        prev = time.time()
        while True:
            time.sleep(30)
            t = time.time()
            if t - prev > 120:
                # the computer was asleep: give the open pages time to ping again
                activity["last"] = t
            prev = t
            if t - activity["last"] > minutes * 60:
                logging.info("No review page open for %s minutes - stopping.", minutes)
                _shutdown_soon(projects, 0)
                return
    threading.Thread(target=run, daemon=True).start()


def _load_two_col(path: str, value_col: str) -> Dict[str, str]:
    """CSV with columns slide_id,<value_col> -> {slide_id: value}."""
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("slide_id") or r.get("slide") or "").strip()
            if sid:
                out[sid] = (r.get(value_col) or "").strip()
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


def load_projects(config_path: Path, extra_dirs: List[str]):
    """Projects from review_projects.json (+ folders given on the command line)."""
    settings: Dict = {}
    entries: List[Dict] = []
    if config_path.is_file():
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            raise ValueError(
                f"{config_path} is not valid JSON ({e}).\n"
                "Tip: write Windows paths with / or \\\\ (e.g. E:/Christine/100k_lymphoma)."
            ) from e
        entries += settings.get("projects", [])
    entries += [{"name": Path(d).name, "folder": d} for d in extra_dirs]

    projects: "OrderedDict[str, ReviewProject]" = OrderedDict()
    for e in entries:
        if not e.get("folder"):
            raise ValueError(f"Project entry without a \"folder\": {e}")
        name = str(e.get("name") or Path(e["folder"]).name)
        key = base = _slug(name)
        n = 2
        while key in projects:
            key, n = f"{base}-{n}", n + 1
        projects[key] = ReviewProject(
            key, name, e["folder"],
            slide_root=e.get("slide_root"),
            csv_path=e.get("csv"),
            slide_map_csv=e.get("slide_map"),
            diagnoses=e.get("diagnoses"),
        )
    return projects, settings


def _already_running(url: str) -> bool:
    """Is our app already answering on this address?"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "api/ping", timeout=1.5) as r:
            return json.loads(r.read().decode("utf-8")).get("app") == APP_ID
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pathologist whole-slide review app",
        epilog="Example: python review_app.py   (projects from review_projects.json)",
    )
    ap.add_argument("project_dir", nargs="*",
                    help="Optional extra project folder(s), in addition to review_projects.json")
    ap.add_argument("--config", default=str(CONFIG_FILE), help="Projects file (default: review_projects.json next to this script)")
    ap.add_argument("--host", default=None,
                    help="127.0.0.1 = only this machine (default); 0.0.0.0 = reachable from the network/VPN")
    ap.add_argument("--port", type=int, default=None, help="default 8050")
    ap.add_argument("--no-browser", action="store_true", help="Don't open the page in a browser on start")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    try:
        projects, settings = load_projects(config_path, args.project_dir)
    except Exception as e:
        _show_error(f"Could not read the project list:\n\n{e}")
        sys.exit(1)
    if not projects:
        _show_error(f"No projects configured.\n\nAdd them to {config_path}.")
        sys.exit(1)

    host = args.host or settings.get("host", "127.0.0.1")
    port = args.port or int(settings.get("port", 8050))
    auto_close = float(settings.get("auto_close_minutes", 15))
    url_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{url_host}:{port}/"

    # Icon double-clicked while the app is already running: just show it again.
    if _already_running(url):
        logging.info("Already running - opening %s", url)
        if not args.no_browser:
            webbrowser.open(url)
        return

    app = create_app(projects, auto_close)
    logging.info("Slide review starting on %s", url)
    for p in projects.values():
        logging.info("  %-14s %s%s", p.name, p.root, "" if p.ok else "   (FOLDER NOT FOUND)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        try:
            from waitress import serve
        except ImportError:
            logging.warning("waitress not installed - using Flask's development server (pip install waitress).")
            app.run(host=host, port=port, threaded=True)
        else:
            serve(app, host=host, port=port, threads=8)
    except OSError as e:
        _show_error(
            f"Slide review could not start on port {port}:\n\n{e}\n\n"
            f"Another program is probably using this port. Change \"port\" in {config_path}."
        )
        os._exit(1)


if __name__ == "__main__":
    main()