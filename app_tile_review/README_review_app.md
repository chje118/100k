# Pathologist review app (whole-slide viewer)

A small password-protected website where the pathologist:

1. picks a **plate**,
2. picks a **slide**,
3. reviews the whole slide with full zoom and pan, the same viewer as your calibration
   wizard. The disease tiles (red, `d1, d2, …`) and healthy tiles (blue, `h1, h2, …`) are noutlined on top. Next to the slide she writes **disease tile notes**, **healthy tile notes** and a **diagnosis**, then clicks *Submit*.

Submitted slides are marked ✓ Done in the slide list, so she can see what's left.

## What it reads

```
<data-root>/<plate>/annotations/<slide>_annotations.json   ← written by interactive_viewer.py
<slide-root>/…/<slide>.mrxs | …      ← the slide files themselves
```

- Plates are folders under `--data-root` that have an `annotations/` folder.
- The disease/healthy outlines are the `disease_polygons` / `healthy_polygons` in the annotation JSON, numbered in the same order as the tile-inspection images (`d1.png`, `h1.png`, …), so the labels match.
- The slide file is found by **file name** (`<slide>.mrxs`) anywhere under `--slide-root`.
  If a slide sits inside the plate's own folder, that copy is preferred. If the names don't match, give a `--slide-map` CSV with columns `slide_id,slide_path`.
- Slides whose file can't be found show *"slide file not found"* in the list. She can still write notes for them.

## Setup

```bash
pip install flask pillow waitress openslide-python openslide-bin
```

Keep the `static/` folder with `openseadragon.min.js` next to `review_app.py`; it is the same file
your calibration viewer uses. Without it the viewer is loaded from a CDN.

## Run

```bash
# once: make a password hash (min. 10 characters)
python review_app.py --hash-password
export REVIEW_PASSWORD_HASH='scrypt:...'
export REVIEW_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"

python review_app.py \
    --data-root  /path/to/plates \
    --slide-root /path/to/wsi_files \
    --csv        pathologist_reviews.csv
```

Optional: `--diagnoses suggested.csv` (columns `slide_id,diagnosis`). This shows your guessed
diagnosis above her diagnosis field.

If `REVIEW_PASSWORD_HASH` isn't set, the app asks for a password at startup.
`REVIEW_SECRET_KEY` keeps her logged in across restarts.

## In the viewer

| | |
|---|---|
| scroll / drag | zoom / pan; double-click zooms in |
| tile chips (`d1 d2 … h1 …`) | jump to that tile, with one tile of context on each side |
| **N** / **P** | next / previous tile |
| **O** | outlines on/off, to see the tissue underneath |
| checkboxes | show/hide disease outlines, healthy outlines, labels |
| *All tiles* / *Whole slide* | zoom out to all marked tiles / the full slide |

The top-left shows the approximate magnification (from the slide's µm/px, 0.25 µm/px ≈ 40×).
There is a scale bar bottom-left and an overview map bottom-right.

## Output: `pathologist_reviews.csv`

One row per slide (UTF-8, opens in Excel with æøå):

`plate, slide_id, status, suggested_diagnosis, disease_tile_notes, healthy_tile_notes, diagnosis, updated_at, completed_at`

- `status` is `draft` while she is writing (notes autosave as she types) and `done` after *Submit*.
- *Submit* requires all three fields (only those that apply: no healthy field if a slide has no
  healthy tiles). It then jumps to the next unreviewed slide.
- She can reopen a done slide to edit it; it stays done.

## Making it reachable for her

The app listens only on your own machine by default, and it streams the actual slide, so it has
to run where the slide files are. To give her a link you need HTTPS in front of it:

1. **Recommended for patient data: via KU.** Run it on a KU server or VM next to the slides,
   reachable over the KU VPN or behind KU IT's HTTPS proxy. Start it with `--https --behind-proxy`.
2. **Tunnel from your own machine,** e.g. `cloudflared tunnel --url http://localhost:8050`, with
   the app started with `--https --behind-proxy`. You send her the printed `https://…` link,
   and the password separately (SMS). Your machine must stay on, and the image data then passes
   through a third-party service.

**Check with your group / data protection contact which route is allowed** for these slides
(GDPR / your data agreements). Also check that plate and slide names contain no CPR numbers,
because they are shown in the app and written to the CSV.

Viewing whole slides remotely needs a reasonable connection on both ends. Tiles are generated on
the fly, as in your calibration viewer, and cached in her browser.

Built-in protection:

- password login (hashed; lockout after 8 wrong attempts per 15 min)
- sessions expire after 12 h
- every page and every slide tile sits behind the login
- only plates and slides that exist on disk can be opened
- pages are marked no-cache / no-index
