"""
Evosep / Xcalibur import list from a plate's well assignment CSV.

Layout (matches evosep_import_100k_P1.csv):
- Each 384-well quadrant is one 96-well Evotip box in its own slot. Default
      Q1 -> EvoSlot 3, Q2 -> EvoSlot 4, Q3 -> EvoSlot 5, Q4 -> EvoSlot 6
  override with slots=, e.g. slots={"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.
  Each quadrant is run as one block, Q1 first, Q4 last.
- Only wells with a selected tile (tile_id filled) are written, so no empty
  tips are run. Pass drop_empty=False to run every assigned well.
- 384 well -> 96 vial (row-major, 1-96):
      vial = (row_idx // 2) * 12 + (col - 1) // 2 + 1      (row_idx: A=0)
- Each quadrant block starts with one HeLa then one blank, both in that
  quadrant's outer-ring wells (excluded from samples):
      Q1/Q2: HeLa = vial 1 (A1), blank = vial 12 (A12)    -> 384 rows A
      Q3/Q4: HeLa = vial 85 (H1), blank = vial 96 (H12)   -> 384 rows P
  numbered Hela01..04 / blank01..04 across the plate.
- Sample order inside each quadrant is randomised (seeded, reproducible).
- Sample Name = Xcalibur Filename =
      <date>_<prefix>_P<plate>_pos<well lowercase>_q<n>_<vial>
      <date>_<prefix>_P<plate>_QC_Hela01 / _QC_blank01

Usage:
    wa = load_well_assignment(planner.well_assignment_path(1))
    evo, wa_ms = make_evosep_import(wa, date="20261023", seed=1,
                                    slots={"Q1": 3, "Q2": 4, "Q3": 5, "Q4": 6})
    write_evosep_csv(evo, "evosep_import_100k_P1.csv")
    # wa_ms = well assignment + run_order / ms_sample_name / evo_slot / evo_vial
"""
from __future__ import annotations

import datetime as _dt
import re

import numpy as np
import pandas as pd

PLATE_ROWS = "ABCDEFGHIJKLMNOP"
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_SLOT = {"Q1": 3, "Q2": 4, "Q3": 5, "Q4": 6}   # default EvoSlot per quadrant
# (HeLa vial, blank vial) per quadrant: outer-ring wells of that quadrant
QC_VIALS = {"Q1": (1, 12), "Q2": (1, 12), "Q3": (85, 96), "Q4": (85, 96)}

DEFAULT_PREFIX = "100K_AST4_ENO0_120SPD"
DEFAULT_METHOD = r"C:\Xcalibur\methods\OVB\Whisper_120SPDZoom_10-5min_240KMS1_3Th5ms_DIA.meth"
DEFAULT_OUTPUT_DIR = r"D:\100K_PROJECT"

EVOSEP_COLUMNS = ["Column1", "Analysis Method", "Source Tray", "Source Vial", "Sample Name",
                  "Xcalibur Method", "Xcalibur Filename", "Xcalibur Processing",
                  "Xcalibur Output Dir", "Comment"]


def _parse_well(well: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-P])(\d{1,2})", str(well).strip().upper())
    if m is None:
        raise ValueError(f"Unrecognised 384 well: {well!r}")
    return PLATE_ROWS.index(m.group(1)), int(m.group(2))


def well_quadrant(well: str) -> str:
    """Same interleaved definition as plate_randomization.well_quadrant."""
    row_idx, col = _parse_well(well)
    if row_idx % 2 == 0:
        return "Q1" if col % 2 else "Q2"
    return "Q3" if col % 2 else "Q4"


def _slot_map(slots: dict | None) -> dict[str, str]:
    """{"Q1": 3, ...} or {"Q1": "EvoSlot 3", ...} -> {"Q1": "EvoSlot 3", ...}."""
    slots = {**QUADRANT_SLOT, **(slots or {})}
    unknown = set(slots) - set(QUADRANTS)
    if unknown:
        raise ValueError(f"Unknown quadrant(s) in slots: {sorted(unknown)}; use {QUADRANTS}.")
    out = {}
    for q, s in slots.items():
        m = re.fullmatch(r"(?:EvoSlot\s*)?(\d+)", str(s).strip(), flags=re.IGNORECASE)
        if m is None:
            raise ValueError(f"{q}: slot {s!r} is not a number or 'EvoSlot <n>'.")
        out[q] = f"EvoSlot {int(m.group(1))}"
    if len(set(out.values())) != len(out):
        raise ValueError(f"Two quadrants share a slot: {out}")
    return out


def well_to_vial(well: str) -> int:
    """384 well -> position (1-96, row-major) in its quadrant's 96 Evotip box."""
    row_idx, col = _parse_well(well)
    return (row_idx // 2) * 12 + (col - 1) // 2 + 1


def make_evosep_import(well_df: pd.DataFrame, date: str | None = None, seed: int = 0,
                       prefix: str = DEFAULT_PREFIX, method: str = DEFAULT_METHOD,
                       output_dir: str = DEFAULT_OUTPUT_DIR, well_col: str = "plate_well",
                       plate: int | None = None, drop_empty: bool = True,
                       slots: dict | None = None):
    """
    Build the Evosep import table for ONE plate.

    well_df:    the plate's well assignment (one row per well).
    date:       YYYYMMDD prefix for file names (default: today).
    seed:       seed for the within-quadrant run order.
    drop_empty: drop wells with no selected tile (tile_id empty, e.g. "no tile"
                notes) so no empty tips are run. Default True; False = run
                every assigned well, like the template.
    slots:      EvoSlot per quadrant, e.g. {"Q1": 3, "Q2": 4} (ints or
                "EvoSlot 3"). Quadrants left out keep the default 3/4/5/6.

    Returns (evosep_df, well_df_with_ms_columns).
    """
    df = well_df.copy()
    if plate is None:
        plates = df["plate"].dropna().unique()
        if len(plates) != 1:
            raise ValueError(f"Expected one plate in well_df, found {list(plates)}; pass plate=.")
        plate = int(plates[0])
    slot_of = _slot_map(slots)
    if drop_empty:
        if "tile_id" not in df.columns:
            raise ValueError("drop_empty=True needs a 'tile_id' column; pass drop_empty=False to run all wells.")
        n_before = len(df)
        df = df[df["tile_id"].notna() & (df["tile_id"].astype(str).str.strip() != "")]
        print(f"Plate {plate}: {len(df)}/{n_before} wells have a selected tile; {n_before - len(df)} empty wells skipped.")
        if df.empty:
            raise ValueError(f"Plate {plate}: no wells with a selected tile (tile_id is empty everywhere).")

    date = date or _dt.date.today().strftime("%Y%m%d")
    base = f"{date}_{prefix}_P{plate}"

    df["_q"] = df[well_col].map(well_quadrant)
    if "quadrant" in df.columns and (df["quadrant"] != df["_q"]).any():
        raise ValueError("'quadrant' column disagrees with the well positions.")
    df["evo_vial"] = df[well_col].map(well_to_vial)
    df["evo_slot"] = df["_q"].map(slot_of)
    df["ms_sample_name"] = [
        f"{base}_pos{str(w).strip().lower()}_{q.lower()}_{v}"
        for w, q, v in zip(df[well_col], df["_q"], df["evo_vial"])
    ]
    # Sanity: a sample must never sit on a QC vial
    for q, vials in QC_VIALS.items():
        clash = df[(df["_q"] == q) & df["evo_vial"].isin(vials)]
        if not clash.empty:
            raise ValueError(f"{q}: sample wells {clash[well_col].tolist()} collide with HeLa/blank vials.")

    rng = np.random.default_rng(seed)
    runs = []          # (slot, vial, sample name, well index or None)
    qc_n = 0
    for q in QUADRANTS:
        block = df[df["_q"] == q]
        if block.empty:
            continue
        qc_n += 1
        hela_v, blank_v = QC_VIALS[q]
        runs.append((slot_of[q], hela_v, f"{base}_QC_Hela{qc_n:02d}", None))
        runs.append((slot_of[q], blank_v, f"{base}_QC_blank{qc_n:02d}", None))
        for idx in rng.permutation(block.index.to_numpy()):
            r = block.loc[idx]
            runs.append((r["evo_slot"], int(r["evo_vial"]), r["ms_sample_name"], idx))

    evo = pd.DataFrame({
        "Column1": np.arange(1, len(runs) + 1),
        "Analysis Method": np.nan,
        "Source Tray": [r[0] for r in runs],
        "Source Vial": [r[1] for r in runs],
        "Sample Name": [r[2] for r in runs],
        "Xcalibur Method": method,
        "Xcalibur Filename": [r[2] for r in runs],
        "Xcalibur Processing": np.nan,
        "Xcalibur Output Dir": output_dir,
        "Comment": np.nan,
    })[EVOSEP_COLUMNS]

    # Link back: run order per well so MS raw files map to tiles/patients
    order = {r[3]: i + 1 for i, r in enumerate(runs) if r[3] is not None}
    df["ms_run_order"] = df.index.map(order).astype("Int64")
    df = df.drop(columns="_q")
    return evo, df


def write_evosep_csv(evo: pd.DataFrame, path: str) -> None:
    """UTF-8 with BOM and CRLF, like the instrument-side template."""
    evo.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
