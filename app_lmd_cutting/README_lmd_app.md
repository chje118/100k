# LMD cutting app

Opens next to the LMD while you cut. Pick a **plate**, then a **slide**. The slide opens in the same whole-slide viewer as the tile review app, with:

- disease tiles (red) and healthy tiles (blue), named like the shapes in the LMD XML: `d0 … d4` / `h0 … h4` (= `lmd_shape_index`, the cut order);
- the three **calibration points** (yellow crosshairs, `C1 C2 C3`). Click a chip or press **1**/**2**/**3** to zoom to one (≈ 600 µm field). *All calibration points* fits all three in view.

For every tile you click **Cut**, **Difficult to cut**, or type your own note in *Other note…*. The note goes straight into the `notes` column of the plate's well assignment CSV, which is the same CSV `ROISelector.write_lmd_and_metadata()` fills. Each tile also shows its `LMD_well` and `plate_well`.

The top bar always shows the slide's **table_id** and **how much of the plate is cut**: the percentage of tiles on the plate that have a note.

## What it reads / writes

```
<project>/<plate>/*_well_assignment_<plate>.csv            read + notes column written
<project>/<plate>/annotations/<slide>_annotations.json     polygons + calibration points (interactive_viewer.py)
<project>/slides/…/<slide>.mrxs                            found by file name, else the CSV's slide_path
```

- A plate is a folder that contains a `*_well_assignment_*.csv`.
- Slides are the `table_id`s that have tiles recorded (`tile_id` filled), in well order. Wells marked "no tile" are skipped.
- A slide is matched to its annotation file the same way `ROISelector.patient_id_from_plate` does it: the `table_id` appears in the slide name.
- Only the `notes` cells change. The rest of the CSV is left exactly as pandas wrote it. Every click re-reads the CSV first, so it is safe to have the notebook write other slides in between.
- If the CSV is open in Excel (Windows locks it), saving fails with a message. Close Excel and click again.
- `write_lmd_and_metadata(..., overwrite=True)` clears the notes for that patient, as before.

## Run

Uses the same environment as the review app (`flask pillow waitress openslide-python openslide-bin`).
It imports the slide loader from `../app_tile_review/review_app.py`, so keep both folders side by side.

```bash
python lmd_app.py E:\Christine\100k_lymphoma          # http://localhost:8060
```

On Windows: double-click `start_lmd.bat`.

## Keys

| | |
|---|---|
| scroll / drag | zoom / pan; double-click zooms in |
| **N** / **P** | next / previous tile |
| **C** / **D** | mark the selected tile cut / difficult to cut, then jump to the next |
| **1** **2** **3** | zoom to calibration point |
| **O** | outlines on/off |

Click an active *Cut* / *Difficult to cut* button again to clear it. *Mark all remaining tiles as cut* fills every empty tile on the slide.
*Next slide →* always goes to the next slide on the plate, done or not. *Continue cutting →* on the plate page opens the first slide that still has tiles without a note.
