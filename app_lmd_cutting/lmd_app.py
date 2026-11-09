r"""
LMD cutting app: whole-slide viewer + cut log.

Used at the LMD while cutting. Pick a plate, then a slide. The slide opens in
the same zoomable viewer as the tile review app, with the disease (red) and
healthy (blue) tiles and the three calibration points drawn on top. For every
tile you mark "cut", "difficult to cut" or write your own note; it is written
straight into the `notes` column of the plate's well assignment CSV (the CSV
filled by ROISelector.write_lmd_and_metadata()).

Tiles are named like the shapes in the LMD XML: d0..d4 / h0..h4
(= lmd_shape_index, the cut order), so they match what the LMD software shows.

Expected project folder:

    <PROJECT_DIR>/<plate>/*_well_assignment_<plate>.csv          (from plate_randomization.py)
    <PROJECT_DIR>/<plate>/annotations/<slide>_annotations.json    (from interactive_viewer.py)
    <PROJECT_DIR>/slides/...                   slide files, found by file name (searched recursively)

Run:
    python lmd_app.py E:\Christine\100k_lymphoma
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import openslide
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    url_for,
)
from openslide.deepzoom import DeepZoomGenerator

# Slide access and page layout are shared with the tile review app.
REVIEW_APP_DIR = Path(__file__).resolve().parent.parent / "app_tile_review"
sys.path.insert(0, str(REVIEW_APP_DIR))
from review_app import ANN_SUFFIX, LAYOUT as REVIEW_LAYOUT, SlideStore, _load_two_col  # noqa: E402

LAYOUT = REVIEW_LAYOUT.replace("· Slide review", "· LMD cutting")

WA_GLOB = "*_well_assignment_*.csv"
NOTE_CUT = "cut"
NOTE_DIFFICULT = "difficult to cut"


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
EXTRA_CSS = r"""
<style>
.b-diff{background:#fff0d9;color:var(--draft)}
.tid{font-size:22px;font-weight:700;letter-spacing:.01em}
.tid small{font-size:13px;font-weight:400;color:var(--muted);margin-left:8px}
.prog{display:flex;align-items:center;gap:8px;white-space:nowrap}
.prog .bar{width:160px;height:8px}
.prog b{font-size:18px}
</style>
"""

PLATES = EXTRA_CSS + r"""
<header>
  <div class="crumbs"><b>Plates</b></div><div class="spacer"></div>
</header>
<main>
  <div class="top"><div><h1>LMD cutting</h1><div class="muted">Choose a plate.</div></div></div>
  <div class="card"><table>
    <tr><th>Plate</th><th>Slides</th><th>Cut progress</th></tr>
    {% for p in plates %}
    <tr class="row" onclick="location.href='{{ url_for('plate', plate=p.name) }}'">
      <td><a href="{{ url_for('plate', plate=p.name) }}">{{ p.name }}</a>
          {% if p.error %} <span class="badge b-err">{{ p.error }}</span>{% endif %}</td>
      <td>{{ p.n_slides }}</td>
      <td><span class="bar"><i style="width:{{ p.pct }}%"></i></span>
          <span class="muted">{{ p.pct }}% · {{ p.noted }} / {{ p.total }} tiles</span></td>
    </tr>
    {% else %}
    <tr><td colspan="3" class="muted">No plates found (a plate is a folder with a *_well_assignment_*.csv).</td></tr>
    {% endfor %}
  </table></div>
</main>
"""


PLATE = EXTRA_CSS + r"""
<header>
  <div class="crumbs"><a href="{{ url_for('index') }}">Plates</a><span>›</span><b>{{ plate }}</b></div>
  <div class="spacer"></div>
  <div class="prog"><span class="bar"><i style="width:{{ prog.pct }}%"></i></span><b>{{ prog.pct }}%</b>
    <span class="muted">cut · {{ prog.noted }} / {{ prog.total }} tiles</span></div>
</header>
<main>
  <div class="top">
    <div><h1>{{ plate }}</h1><div class="muted">{{ csv_name }}</div></div>
    <div class="spacer"></div>
    {% if next_todo %}<a class="btn primary" href="{{ url_for('slide', plate=plate, table_id=next_todo) }}">Continue cutting →</a>{% endif %}
  </div>
  <div class="card"><table>
    <tr><th>#</th><th>Table ID</th><th>Slide</th><th>Tiles</th><th>Status</th></tr>
    {% for s in slides %}
    <tr class="row" onclick="location.href='{{ url_for('slide', plate=plate, table_id=s.table_id) }}'">
      <td class="muted">{{ loop.index }}</td>
      <td><a href="{{ url_for('slide', plate=plate, table_id=s.table_id) }}">{{ s.table_id }}</a></td>
      <td>{{ s.slide or '' }}
          {% if not s.has_ann %} <span class="badge b-err">no annotations</span>{% endif %}
          {% if not s.has_wsi %} <span class="badge b-err">slide file not found</span>{% endif %}</td>
      <td>{{ s.n }}</td>
      <td>{% if s.noted == s.n %}<span class="badge b-done">✓ Done</span>
          {% elif s.noted %}<span class="badge b-draft">{{ s.noted }} / {{ s.n }}</span>
          {% else %}<span class="badge b-todo">Not started</span>{% endif %}
          {% if s.difficult %} <span class="badge b-diff">{{ s.difficult }} difficult</span>{% endif %}
          {% if s.other %} <span class="badge b-todo">{{ s.other }} other note</span>{% endif %}</td>
    </tr>
    {% endfor %}
  </table></div>
</main>
"""


SLIDE = EXTRA_CSS + r"""
<style>
body{overflow:hidden}
.review{display:grid;grid-template-columns:minmax(0,1fr) 400px;height:calc(100vh - 61px)}
.viewer-col{display:flex;flex-direction:column;min-width:0;background:#000}
.tools{display:flex;align-items:center;gap:6px;padding:7px 10px;background:#1b1f27;color:#d7dbe3;flex-wrap:wrap;font-size:13px}
.tools button{background:#2a303b;border-color:#3a4150;color:#d7dbe3;padding:4px 10px;font-size:13px}
.tools button:hover{border-color:#6b7484}
.tools label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none;margin-left:6px}
.tools .sep{width:1px;height:20px;background:#3a4150;margin:0 4px}
.chips{display:flex;gap:4px;overflow-x:auto;padding:6px 10px;background:#151920;min-height:38px;align-items:center}
.chip{flex:0 0 auto;font:600 12px/1 system-ui,sans-serif;padding:5px 8px;border-radius:6px;cursor:pointer;
      background:#232833;border:1px solid #3a4150;color:#d7dbe3}
.chip.d{border-color:#7a2a2a}.chip.h{border-color:#2a3f7a}.chip.c{border-color:#7a6a10}
.chip.on{background:#fff;color:#111;border-color:#fff}
.chip .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
#wrap{position:relative;flex:1;min-height:0}
#osd{position:absolute;inset:0}
#overlay{position:absolute;inset:0;pointer-events:none;z-index:5}
#mag{position:absolute;left:10px;top:10px;z-index:6;background:rgba(0,0,0,.6);color:#fff;font:600 13px system-ui,sans-serif;padding:4px 8px;border-radius:6px;pointer-events:none}
#missing{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#ddd;text-align:center;padding:30px}
.form-col{overflow:auto;padding:16px;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column}
.sec{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.sec h3{margin:0 0 8px;font-size:15px;display:flex;align-items:center;gap:8px}
.sec .dot{width:10px;height:10px;border-radius:50%}
.tile{border:1px solid var(--line);border-radius:8px;padding:8px;margin-top:6px;cursor:pointer}
.tile.on{border-color:var(--accent);box-shadow:0 0 0 2px #dfe7fb}
.tile.k-cut{background:#f1faf4}.tile.k-difficult{background:#fff8ec}.tile.k-other{background:#f3f5fb}
.tile .line{display:flex;align-items:center;gap:6px}
.tile .name{font-weight:700;min-width:30px}
.tile .well{font-size:13px;color:var(--muted)}
.tile .acts{display:flex;gap:4px;margin-left:auto}
.tile .acts button{font-size:13px;padding:3px 9px}
.tile .acts button.on.cut{background:var(--done);border-color:var(--done);color:#fff}
.tile .acts button.on.difficult{background:var(--draft);border-color:var(--draft);color:#fff}
.tile input{width:100%;margin-top:6px;font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:6px}
.tile input.on{border-color:var(--accent);background:#fff}
.nav{display:flex;gap:8px;margin-top:auto;padding-top:8px}
.nav .btn{flex:1;text-align:center}
#save-state{font-size:13px}
.muted.small{font-size:12.5px}
kbd{font:600 11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;background:#fafbfc}
@media (max-width:900px){body{overflow:auto}.review{grid-template-columns:1fr;height:auto}#wrap{height:65vh;flex:none}}
</style>

<header>
  <div class="crumbs"><a href="{{ url_for('index') }}">Plates</a><span>›</span>
    <a href="{{ url_for('plate', plate=plate) }}">{{ plate }}</a></div>
  <div class="tid">{{ table_id }}<small>{{ slide or 'no slide file' }} · slide {{ pos }} of {{ n_slides }}</small></div>
  <div class="spacer"></div>
  <span id="save-state" class="muted"></span>
  <div class="prog" title="Tiles on this plate with a note">
    <span class="bar"><i id="pbar"></i></span><b id="ppct"></b>
    <span class="muted" id="pdetail"></span></div>
</header>

<div class="review">
  <section class="viewer-col">
    <div class="tools">
      <button type="button" id="btn-tiles" title="Zoom to all marked tiles">All tiles</button>
      <button type="button" id="btn-cal" title="Zoom to all calibration points">All calibration points</button>
      <button type="button" id="btn-home" title="Show the whole slide">Whole slide</button>
      <span class="sep"></span>
      <label><input type="checkbox" id="show-d" checked> <span style="color:var(--dis)">■</span> Disease</label>
      <label><input type="checkbox" id="show-h" checked> <span style="color:var(--hea)">■</span> Healthy</label>
      <label><input type="checkbox" id="show-c" checked> <span style="color:#ffd400">✚</span> Calibration</label>
      <label><input type="checkbox" id="show-l" checked> Labels</label>
      <div class="spacer"></div>
      <span style="color:#9aa3b2">scroll = zoom · drag = pan · <b>N</b>/<b>P</b> next/prev tile · <b>1</b>–<b>3</b> calibration point · <b>O</b> outlines</span>
    </div>
    <div class="chips" id="chips"></div>
    <div id="wrap">
      {% if has_wsi %}
      <div id="osd"></div>
      <canvas id="overlay"></canvas>
      <div id="mag"></div>
      {% else %}
      <div id="missing"><div><b>The slide file for {{ table_id }} was not found.</b><br>
        You can still mark the tiles on the right.</div></div>
      {% endif %}
    </div>
  </section>

  <aside class="form-col">
    {% if not cal_n %}<p class="muted small" style="margin-top:0">No calibration points saved for this slide.</p>{% endif %}
    <div id="groups"></div>
    <button type="button" id="all-cut" class="btn">Mark all remaining tiles as cut</button>
    <p class="muted small">Click a tile to zoom to it. <kbd>C</kbd> cut · <kbd>D</kbd> difficult to cut
      (for the selected tile, then jumps to the next). Click an active button again to clear it.
      Every click is saved straight into the plate CSV.</p>
    <div class="nav">
      {% if prev_url %}<a class="btn" href="{{ prev_url }}">← Previous slide</a>{% endif %}
      <a class="btn primary" href="{{ next_url or url_for('plate', plate=plate) }}">{{ 'Next slide →' if next_url else 'Back to plate' }}</a>
    </div>
  </aside>
</div>

{% if has_wsi %}<script src="{{ osd_js }}"></script>{% endif %}
<script>
const CFG = {{ cfg|tojson }};
const tiles = CFG.tiles;   // [{run_id, id, group:'d'|'h', poly, bbox, lmd_well, plate_well, note}]
const cals = CFG.cal;      // [{id:'C1', x, y}]
const PRESETS = {cut: CFG.note_cut, difficult: CFG.note_difficult};
let current = -1;          // index into tiles, or -1
let draw = () => {};

// ---------------- notes ----------------
function kind(note) {
  const n = (note || '').trim().toLowerCase();
  if (!n) return '';
  if (n === PRESETS.cut) return 'cut';
  if (n === PRESETS.difficult) return 'difficult';
  return 'other';
}
function setState(t) { document.getElementById('save-state').textContent = t; }
function showProgress(p) {
  document.getElementById('pbar').style.width = p.pct + '%';
  document.getElementById('ppct').textContent = p.pct + '%';
  document.getElementById('pdetail').textContent = `of ${CFG.plate} cut · ${p.noted} / ${p.total} tiles`;
}
showProgress(CFG.progress);

async function save(notes) {           // notes: {run_id: note}
  setState('Saving…');
  let r;
  try {
    r = await fetch(CFG.save_url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify({notes})});
  } catch (e) { setState('⚠ Not saved – connection problem'); return false; }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { setState('⚠ Not saved: ' + (data.error || 'error ' + r.status)); return false; }
  setState('Saved ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}));
  showProgress(data.progress);
  return true;
}
async function setNote(i, note) {
  const old = tiles[i].note;
  tiles[i].note = note; render(i); draw();
  if (!await save({[tiles[i].run_id]: note})) { tiles[i].note = old; render(i); draw(); }
}

// ---------------- tile list ----------------
const groups = document.getElementById('groups');
const rows = [];
for (const [g, title, color] of [['d', 'Disease tiles', 'var(--dis)'], ['h', 'Healthy tiles', 'var(--hea)']]) {
  const idx = tiles.map((t, i) => i).filter(i => tiles[i].group === g);
  if (!idx.length) continue;
  const sec = document.createElement('div');
  sec.className = 'sec';
  sec.innerHTML = `<h3><span class="dot" style="background:${color}"></span>${title} (${idx.length})</h3>`;
  for (const i of idx) {
    const t = tiles[i];
    const el = document.createElement('div');
    el.className = 'tile';
    el.innerHTML = `<div class="line"><span class="name"></span><span class="well"></span>
        <span class="acts"><button type="button" class="cut">Cut</button>
        <button type="button" class="difficult">Difficult to cut</button></span></div>
      <input type="text" placeholder="Other note…" maxlength="500">`;
    el.querySelector('.name').textContent = t.id;
    el.querySelector('.well').textContent = 'LMD well ' + (t.lmd_well || '?') + (t.plate_well ? ' · plate ' + t.plate_well : '');
    el.title = t.run_id;
    for (const k of ['cut', 'difficult']) {
      el.querySelector('button.' + k).onclick = e => {
        e.stopPropagation();
        setNote(i, kind(tiles[i].note) === k ? '' : PRESETS[k]);
      };
    }
    const inp = el.querySelector('input');
    let timer = null;
    const commit = () => {
      clearTimeout(timer);
      const v = inp.value.trim();
      if (v === (kind(tiles[i].note) === 'other' ? tiles[i].note : '')) return;
      if (!v && kind(tiles[i].note) !== 'other') return;
      setNote(i, v);
    };
    inp.oninput = () => { setState('Unsaved changes'); clearTimeout(timer); timer = setTimeout(commit, 1000); };
    inp.onblur = commit;
    inp.onkeydown = e => { if (e.key === 'Enter') { commit(); inp.blur(); } };
    inp.onclick = e => e.stopPropagation();
    inp.onfocus = () => focusTile(i, true);
    el.onclick = () => focusTile(i);
    sec.appendChild(el);
    rows[i] = el;
  }
  groups.appendChild(sec);
}
function render(i) {
  const el = rows[i], k = kind(tiles[i].note);
  el.className = 'tile' + (k ? ' k-' + k : '') + (i === current ? ' on' : '');
  el.querySelector('button.cut').classList.toggle('on', k === 'cut');
  el.querySelector('button.difficult').classList.toggle('on', k === 'difficult');
  const inp = el.querySelector('input');
  if (document.activeElement !== inp) inp.value = k === 'other' ? tiles[i].note : '';
  inp.classList.toggle('on', k === 'other');
  chipEls[i] && (chipEls[i].querySelector('.st').textContent = k === 'cut' ? ' ✓' : k === 'difficult' ? ' !' : k === 'other' ? ' •' : '');
}

document.getElementById('all-cut').onclick = async () => {
  const todo = tiles.map((t, i) => i).filter(i => !kind(tiles[i].note));
  if (!todo.length) return;
  if (!confirm(`Mark ${todo.length} remaining tile(s) as "${PRESETS.cut}"?`)) return;
  const notes = {};
  todo.forEach(i => { notes[tiles[i].run_id] = PRESETS.cut; });
  if (await save(notes)) { todo.forEach(i => { tiles[i].note = PRESETS.cut; render(i); }); draw(); }
};

// ---------------- chips ----------------
const chips = document.getElementById('chips');
const chipEls = [], calChipEls = [];
const COLORS = {d: '#ff2d2d', h: '#2d6cff', c: '#ffd400'};
function chip(cls, label, color, onclick, title) {
  const c = document.createElement('span');
  c.className = 'chip ' + cls;
  c.innerHTML = `<span class="dot" style="background:${color}"></span><span class="lb"></span><span class="st"></span>`;
  c.querySelector('.lb').textContent = label;
  c.title = title; c.onclick = onclick;
  chips.appendChild(c);
  return c;
}
cals.forEach((p, j) => calChipEls.push(chip('c', p.id, COLORS.c, () => focusCal(j), 'Zoom to calibration point ' + p.id)));
tiles.forEach((t, i) => chipEls.push(chip(t.group, t.id, COLORS[t.group], () => focusTile(i), 'Zoom to ' + t.id)));
if (!tiles.length && !cals.length) chips.innerHTML = '<span style="color:#9aa3b2;font-size:13px">No tiles or calibration points for this slide.</span>';
tiles.forEach((t, i) => render(i));

let currentCal = -1;
function markSelection() {
  chipEls.forEach((c, i) => c.classList.toggle('on', i === current));
  calChipEls.forEach((c, j) => c.classList.toggle('on', j === currentCal));
  rows.forEach((el, i) => el && el.classList.toggle('on', i === current));
  const a = current >= 0 ? chipEls[current] : calChipEls[currentCal];
  if (a && a.scrollIntoView) a.scrollIntoView({block: 'nearest', inline: 'nearest'});
  if (current >= 0 && rows[current]) rows[current].scrollIntoView({block: 'nearest'});
}
let zoomTile = () => {}, zoomCal = () => {};
function focusTile(i, noZoom) { current = i; currentCal = -1; markSelection(); if (!noZoom) zoomTile(i); draw(); }
function focusCal(j) { currentCal = j; current = -1; markSelection(); zoomCal(j); draw(); }

document.addEventListener('keydown', e => {
  if (['TEXTAREA', 'INPUT'].includes(e.target.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase(), n = tiles.length;
  if ((k === 'n' || k === 'p') && n) {
    focusTile(k === 'n' ? (current + 1) % n : (current - 1 + n) % n);
  } else if ((k === 'c' || k === 'd') && current >= 0) {
    const want = k === 'c' ? 'cut' : 'difficult';
    const i = current;
    setNote(i, kind(tiles[i].note) === want ? '' : PRESETS[want]);
    if (kind(tiles[i].note) && i + 1 < n) focusTile(i + 1);
  } else if (['1', '2', '3', '4', '5'].includes(k) && cals[+k - 1]) {
    focusCal(+k - 1);
  } else if (k === 'o') {
    toggleOutlines();
  }
});
let toggleOutlines = () => {};

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
  const show = {d: true, h: true, c: true, l: true};
  const item = () => viewer.world.getItemAt(0);
  // calibration zoom: ~600 µm field (fallback 2400 level-0 px)
  const CAL_HALF = CFG.mpp ? 300 / CFG.mpp : 1200;

  function fitBox(x0, y0, x1, y1, immediate) {
    viewer.viewport.fitBounds(item().imageToViewportRectangle(x0, y0, x1 - x0, y1 - y0), immediate);
  }
  function fitPoints(pts, pad, immediate) {
    if (!pts.length) { viewer.viewport.goHome(immediate); return; }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    fitBox(Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad, immediate);
  }
  const boxed = tiles.filter(t => t.bbox);
  const tileW = boxed.length ? Math.max(...boxed.map(t => t.bbox[2] - t.bbox[0])) : 0;
  function fitAllTiles(immediate) {
    current = -1; currentCal = -1; markSelection();
    fitPoints(boxed.flatMap(t => [[t.bbox[0], t.bbox[1]], [t.bbox[2], t.bbox[3]]]), 2 * tileW, immediate);
  }
  function fitAllCals() {
    current = -1; currentCal = -1; markSelection();
    fitPoints(cals.map(p => [p.x, p.y]), CAL_HALF, false);
  }
  zoomTile = i => {
    const b = tiles[i].bbox; if (!b || !item()) return;
    const w = b[2] - b[0], h = b[3] - b[1];
    fitBox(b[0] - w, b[1] - h, b[2] + w, b[3] + h);   // tile + 1 tile margin (3x3 field)
  };
  zoomCal = j => {
    const p = cals[j]; if (!item()) return;
    fitBox(p.x - CAL_HALF, p.y - CAL_HALF, p.x + CAL_HALF, p.y + CAL_HALF);
  };
  toggleOutlines = () => {
    const on = !(show.d || show.h);
    show.d = show.h = on;
    document.getElementById('show-d').checked = document.getElementById('show-h').checked = on;
    draw();
  };

  draw = function () {
    const r = wrap.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
      canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    const it = item(); if (!it) return;
    const toScreen = (x, y) => it.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
    const label = (text, x, y, color) => {
      ctx.font = 'bold 14px system-ui, Arial, sans-serif';
      ctx.lineWidth = 3; ctx.strokeStyle = 'black'; ctx.strokeText(text, x, y);
      ctx.fillStyle = color; ctx.fillText(text, x, y);
    };

    tiles.forEach((t, i) => {
      if (!show[t.group] || !t.poly) return;
      ctx.beginPath();
      t.poly.forEach((p, j) => { const s = toScreen(p[0], p[1]); j ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.closePath();
      const k = kind(t.note);
      if (k === 'cut' || k === 'difficult') {   // already cut: light fill
        ctx.fillStyle = k === 'cut' ? 'rgba(40,200,90,.18)' : 'rgba(255,160,0,.22)'; ctx.fill();
      }
      const hl = i === current;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'black'; ctx.lineWidth = hl ? 4 : 2.6; ctx.stroke();
      ctx.strokeStyle = COLORS[t.group]; ctx.lineWidth = hl ? 2.2 : 1.2; ctx.stroke();
      if (show.l) {
        const s = toScreen(t.bbox[0], t.bbox[1]);
        label(t.id + (k === 'cut' ? ' ✓' : k === 'difficult' ? ' !' : k === 'other' ? ' •' : ''), s.x, s.y - 5, COLORS[t.group]);
      }
    });

    if (show.c) {
      cals.forEach((p, j) => {
        const s = toScreen(p.x, p.y), hl = j === currentCal, R = hl ? 14 : 10;
        ctx.lineCap = 'round';
        for (const [w, c] of [[4, 'black'], [hl ? 2.2 : 1.6, COLORS.c]]) {
          ctx.strokeStyle = c; ctx.lineWidth = w;
          ctx.beginPath(); ctx.arc(s.x, s.y, R, 0, 2 * Math.PI);
          ctx.moveTo(s.x - 2 * R, s.y); ctx.lineTo(s.x - 4, s.y); ctx.moveTo(s.x + 4, s.y); ctx.lineTo(s.x + 2 * R, s.y);
          ctx.moveTo(s.x, s.y - 2 * R); ctx.lineTo(s.x, s.y - 4); ctx.moveTo(s.x, s.y + 4); ctx.lineTo(s.x, s.y + 2 * R);
          ctx.stroke();
        }
        if (show.l) label(p.id, s.x + R + 4, s.y - R - 2, COLORS.c);
      });
    }

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
  };

  viewer.addHandler('open', () => { fitAllTiles(true); draw(); });
  viewer.addHandler('update-viewport', draw);
  viewer.addHandler('resize', draw);
  viewer.addHandler('open-failed', () => {
    document.getElementById('mag').textContent = 'Could not load slide';
  });

  document.getElementById('btn-tiles').onclick = () => fitAllTiles(false);
  document.getElementById('btn-cal').onclick = fitAllCals;
  document.getElementById('btn-cal').disabled = !cals.length;
  document.getElementById('btn-home').onclick = () => { current = -1; currentCal = -1; markSelection(); viewer.viewport.goHome(); };
  for (const g of ['d', 'h', 'c', 'l']) {
    document.getElementById('show-' + g).onchange = e => { show[g] = e.target.checked; draw(); };
  }
} else {
  document.querySelector('.tools').style.display = 'none';
}
</script>
"""


# ----------------------------------------------------------------------
# Well assignment CSV
# ----------------------------------------------------------------------
class PlateCSVError(Exception):
    pass


class WellAssignment:
    """
    The plate's well assignment CSV, read and written with the csv module so
    every cell except the edited notes stays byte-for-byte as pandas wrote it.
    """

    def __init__(self, path: Path):
        self.path = path
        raw = path.read_bytes()
        self.bom = raw.startswith(b"\xef\xbb\xbf")
        self.newline = "\r\n" if b"\r\n" in raw[:4096] else "\n"
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        self.fields = list(reader.fieldnames or [])
        self.rows = list(reader)
        missing = {"run_id", "table_id", "arm", "tile_id", "lmd_shape_index", "notes"} - set(self.fields)
        if missing:
            raise PlateCSVError(f"{path.name} has no column(s): {', '.join(sorted(missing))}")

    def tile_rows(self) -> List[Dict[str, str]]:
        """Wells that got a tile from write_lmd_and_metadata(), in CSV order."""
        return [r for r in self.rows if (r.get("tile_id") or "").strip() and (r.get("table_id") or "").strip()]

    def write(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig" if self.bom else "utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields, lineterminator=self.newline)
            w.writeheader()
            w.writerows(self.rows)
        try:
            os.replace(tmp, self.path)  # atomic: the CSV is never half-written
        except PermissionError:
            tmp.unlink(missing_ok=True)
            raise PlateCSVError(f"{self.path.name} is locked – close it in Excel and click again.")


def note_kind(note: str) -> str:
    n = (note or "").strip().lower()
    if not n:
        return ""
    return {NOTE_CUT: "cut", NOTE_DIFFICULT: "difficult"}.get(n, "other")


def progress(tile_rows: List[Dict[str, str]]) -> Dict[str, int]:
    total = len(tile_rows)
    noted = sum(bool((r.get("notes") or "").strip()) for r in tile_rows)
    return {"total": total, "noted": noted, "pct": round(100 * noted / total) if total else 0}


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
def create_app(
    data_root: str,
    slide_root: Optional[str] = None,
    slide_map_csv: Optional[str] = None,
) -> Flask:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    wsi_root = Path(slide_root).expanduser().resolve() if slide_root else root
    if not wsi_root.is_dir():
        raise FileNotFoundError(f"Slide root does not exist: {wsi_root}")

    slides_store = SlideStore(
        wsi_root, _load_two_col(slide_map_csv, "slide_path") if slide_map_csv else {}
    )
    lock = threading.Lock()

    static_dir = REVIEW_APP_DIR / "static"
    app = Flask(__name__, static_folder=None)
    osd_candidates = [static_dir / "openseadragon.min.js", REVIEW_APP_DIR / "openseadragon.min.js"]
    osd_file = next((p for p in osd_candidates if p.is_file()), None)
    osd_js = (
        "/osd/openseadragon.min.js"
        if osd_file
        else "https://cdn.jsdelivr.net/npm/openseadragon@5.0.1/build/openseadragon/openseadragon.min.js"
    )

    # ---------------- data helpers ----------------
    def wa_path(plate: str) -> Optional[Path]:
        files = [f for f in (root / plate).glob(WA_GLOB) if not f.name.startswith(".")]
        if len(files) > 1:  # prefer <...>_well_assignment_<plate>.csv
            files = [f for f in files if f.stem.endswith(f"_{plate}")] or files
        return sorted(files)[0] if files else None

    def list_plates() -> List[str]:
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and wa_path(p.name) is not None
        )

    def load_wa(plate: str) -> WellAssignment:
        if plate not in list_plates():  # only names discovered on disk -> no path traversal
            abort(404)
        return WellAssignment(wa_path(plate))

    def table_ids(wa: WellAssignment) -> List[str]:
        """Slides on the plate, in well order."""
        out: List[str] = []
        for r in wa.tile_rows():
            if r["table_id"] not in out:
                out.append(r["table_id"])
        return out

    def ann_slides(plate: str) -> List[str]:
        d = root / plate / "annotations"
        if not d.is_dir():
            return []
        return sorted(f.name[: -len(ANN_SUFFIX)] for f in d.glob("*" + ANN_SUFFIX) if not f.name.startswith("."))

    def slide_name(plate: str, table_id: str, rows: List[Dict[str, str]]) -> Tuple[Optional[str], bool]:
        """(slide name, has annotations). Same rule as ROISelector: table_id appears in the slide name."""
        pat = re.compile(r"(?<![0-9A-Za-z])" + re.escape(table_id) + r"(?![0-9])")
        hits = [s for s in ann_slides(plate) if pat.search(s)]
        if hits:
            return hits[0], True
        for r in rows:
            sp = (r.get("slide_path") or "").strip()
            if sp:
                return PureWindowsPath(sp).stem, False
        return None, False

    def load_ann(plate: str, slide: Optional[str]) -> Dict:
        if not slide:
            return {}
        f = root / plate / "annotations" / f"{slide}{ANN_SUFFIX}"
        try:
            return json.loads(f.read_text())
        except (OSError, ValueError):
            return {}

    def wsi_path(plate: str, slide: Optional[str], rows: List[Dict[str, str]]) -> Optional[Path]:
        if not slide:
            return None
        p = slides_store.find(slide, root / plate)
        if p is None:
            for r in rows:  # path recorded by write_lmd_and_metadata, if it exists here
                sp = (r.get("slide_path") or "").strip()
                if sp and Path(sp).is_file():
                    return Path(sp)
        return p

    def slide_rows(wa: WellAssignment, table_id: str) -> List[Dict[str, str]]:
        return [r for r in wa.tile_rows() if r["table_id"] == table_id]

    def tile_label(r: Dict[str, str]) -> str:
        try:
            k = int(float(r["lmd_shape_index"]))
        except (TypeError, ValueError):
            k = None
        return f"{(r.get('arm') or '?')[:1]}{'' if k is None else k}"

    def build_tiles(rows: List[Dict[str, str]], ann: Dict) -> List[Dict]:
        tiles = []
        for arm, key in (("disease", "disease_polygons"), ("healthy", "healthy_polygons")):
            polys = ann.get(key) or []
            arm_rows = [r for r in rows if r.get("arm") == arm]
            arm_rows.sort(key=lambda r: float(r["lmd_shape_index"]) if r.get("lmd_shape_index") else 1e9)
            for r in arm_rows:
                poly = bbox = None
                try:
                    pts = [[float(p[0]), float(p[1])] for p in polys[int(float(r["lmd_shape_index"]))] if len(p) >= 2]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    poly, bbox = pts, [min(xs), min(ys), max(xs), max(ys)]
                except (IndexError, KeyError, TypeError, ValueError):
                    pass
                tiles.append({
                    "run_id": r["run_id"], "id": tile_label(r), "group": arm[0],
                    "poly": poly, "bbox": bbox,
                    "lmd_well": r.get("LMD_well", ""), "plate_well": r.get("plate_well", ""),
                    "note": r.get("notes", ""),
                })
        return tiles

    def cal_points(ann: Dict) -> List[Dict]:
        out = []
        for j, p in enumerate(ann.get("calibration_points") or [], start=1):
            try:
                out.append({"id": f"C{j}", "x": float(p["x"]), "y": float(p["y"])})
            except (KeyError, TypeError, ValueError):
                pass
        return out

    def next_todo(wa: WellAssignment) -> Optional[str]:
        for t in table_ids(wa):
            if any(not (r.get("notes") or "").strip() for r in slide_rows(wa, t)):
                return t
        return None

    def page(title: str, tpl: str, **ctx) -> str:
        return render_template_string(LAYOUT, title=title, body=render_template_string(tpl, **ctx))

    # ---------------- headers ----------------
    @app.after_request
    def headers(resp):
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/dzi/") and resp.status_code == 200:
            resp.headers["Cache-Control"] = "private, max-age=3600"
        elif not request.path.startswith("/osd/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(PlateCSVError)
    def csv_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": str(e)}), 409
        return Response(f"<p style='font:16px system-ui;padding:20px'>{e}</p>", status=500)

    # ---------------- pages ----------------
    @app.route("/osd/openseadragon.min.js")
    def osd():
        if not osd_file:
            abort(404)
        return Response(osd_file.read_bytes(), mimetype="application/javascript")

    @app.route("/")
    def index():
        plates = []
        for p in list_plates():
            try:
                wa = WellAssignment(wa_path(p))
                plates.append({"name": p, "n_slides": len(table_ids(wa)), **progress(wa.tile_rows()), "error": ""})
            except PlateCSVError as e:
                plates.append({"name": p, "n_slides": 0, "total": 0, "noted": 0, "pct": 0, "error": str(e)})
        return page("Plates", PLATES, plates=plates)

    @app.route("/plate/<plate>")
    def plate(plate):
        wa = load_wa(plate)
        slides = []
        for t in table_ids(wa):
            rows = slide_rows(wa, t)
            name, has_ann = slide_name(plate, t, rows)
            kinds = [note_kind(r.get("notes", "")) for r in rows]
            slides.append({
                "table_id": t, "slide": name, "has_ann": has_ann,
                "has_wsi": wsi_path(plate, name, rows) is not None,
                "n": len(rows), "noted": sum(bool(k) for k in kinds),
                "difficult": kinds.count("difficult"), "other": kinds.count("other"),
            })
        return page(plate, PLATE, plate=plate, slides=slides, prog=progress(wa.tile_rows()),
                    csv_name=wa.path.name, next_todo=next_todo(wa))

    @app.route("/slide/<plate>/<table_id>")
    def slide(plate, table_id):
        wa = load_wa(plate)
        order = table_ids(wa)
        if table_id not in order:
            abort(404)
        rows = slide_rows(wa, table_id)
        name, _ = slide_name(plate, table_id, rows)
        ann = load_ann(plate, name)
        path = wsi_path(plate, name, rows)
        mpp = None
        if path is not None:
            try:
                _, mpp = slides_store.deepzoom(path)
            except openslide.OpenSlideError:
                path = None
        i = order.index(table_id)
        cal = cal_points(ann)
        cfg = {
            "plate": plate,
            "tiles": build_tiles(rows, ann),
            "cal": cal,
            "has_wsi": path is not None,
            "mpp": mpp,
            "dzi_url": url_for("dzi", plate=plate, table_id=table_id),
            "save_url": url_for("save_notes", plate=plate, table_id=table_id),
            "progress": progress(wa.tile_rows()),
            "note_cut": NOTE_CUT,
            "note_difficult": NOTE_DIFFICULT,
        }
        link = lambda t: url_for("slide", plate=plate, table_id=t)  # noqa: E731
        return page(table_id, SLIDE, plate=plate, table_id=table_id, slide=name, cfg=cfg,
                    osd_js=osd_js, has_wsi=path is not None, cal_n=len(cal),
                    pos=i + 1, n_slides=len(order),
                    prev_url=link(order[i - 1]) if i > 0 else None,
                    next_url=link(order[i + 1]) if i + 1 < len(order) else None)

    # ---------------- deep zoom ----------------
    def get_dz(plate: str, table_id: str) -> DeepZoomGenerator:
        wa = load_wa(plate)
        if table_id not in table_ids(wa):
            abort(404)
        rows = slide_rows(wa, table_id)
        path = wsi_path(plate, slide_name(plate, table_id, rows)[0], rows)
        if path is None:
            abort(404)
        return slides_store.deepzoom(path)[0]

    @app.route("/dzi/<plate>/<table_id>.dzi")
    def dzi(plate, table_id):
        return Response(get_dz(plate, table_id).get_dzi("jpeg"), mimetype="application/xml")

    @app.route("/dzi/<plate>/<table_id>_files/<int:level>/<int:col>_<int:row>.jpeg")
    def dzi_tile(plate, table_id, level, col, row):
        dz = get_dz(plate, table_id)
        try:
            img = dz.get_tile(level, (col, row))
        except ValueError:
            abort(404)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        return Response(buf.getvalue(), mimetype="image/jpeg")

    # ---------------- saving ----------------
    @app.route("/api/notes/<plate>/<table_id>", methods=["POST"])
    def save_notes(plate, table_id):
        data = request.get_json(silent=True) or {}
        notes = data.get("notes")
        if not isinstance(notes, dict) or not notes:
            abort(400)
        with lock:
            wa = load_wa(plate)  # read fresh: the notebook may have written in between
            by_run = {r["run_id"]: r for r in slide_rows(wa, table_id)}
            if not set(notes) <= set(by_run):
                return jsonify({"error": "tile not found on this slide – reload the page"}), 409
            for run_id, note in notes.items():
                by_run[run_id]["notes"] = " ".join(str(note or "").split())[:500]
            wa.write()
        return jsonify({"progress": progress(wa.tile_rows())})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LMD cutting app: whole-slide viewer that writes cut notes into the plate CSV",
        epilog="Example: python lmd_app.py E:\\Christine\\100k_lymphoma",
    )
    ap.add_argument("project_dir",
                    help="Project folder: contains the plate folders (with *_well_assignment_*.csv) and slides/")
    ap.add_argument("--slide-root", default=None,
                    help="Override: folder searched for slide files (default: <project_dir>/slides, "
                         "or <project_dir> if there is no slides folder)")
    ap.add_argument("--slide-map", default=None, help="Optional CSV: slide_id,slide_path (overrides the search)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this machine; 0.0.0.0 = reachable from the network/VPN")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--no-browser", action="store_true", help="Don't open the page in a browser on start")
    args = ap.parse_args()

    project = Path(args.project_dir).expanduser().resolve()
    if not project.is_dir():
        sys.exit(f"Project folder not found: {project}")

    slides_dir = project / "slides"
    slide_root = Path(args.slide_root) if args.slide_root else (slides_dir if slides_dir.is_dir() else project)

    app = create_app(data_root=str(project), slide_root=str(slide_root), slide_map_csv=args.slide_map)

    print(f"Project: {project}")
    print(f"Slides:  {slide_root}")
    url_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{url_host}:{args.port}/"
    print(f"Open {url} in a browser  (Ctrl+C to stop)")
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        print("waitress not installed – using Flask's development server (pip install waitress).")
        app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
