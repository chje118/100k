"""
Slide randomization, plate layout and LMD/MS metadata tracking for DVP.

What it does:
- Patients are assigned to plates via STRATIFIED RANDOMIZATION by disease
  subtype / control group, so every plate carries a representative mix
  (controls technical/batch variation across plates).
- A patient's FULL tile set (disease + healthy) always stays on one plate,
  so plate-level batch effects cancel in the paired within-patient
  comparison (the primary analysis).
- Per plate: a box-by-box slide list to pull, then a metadata CSV with one
  row per well (= one LMD cut = one MS sample). ROISelector fills in the
  ROI columns via record_rois() when it writes each slide's LMD XML.

Usage:
    cfg = PlateConfig(disease_tiles=5, healthy_tiles=5)
    planner = PlatePlanner(patients_df, project_id="DVP1", project_dir="meta",
                           config=cfg, mapping_excel="mapping.xlsx")
    planner.summarize()
    planner.get_rekvnr_on_plate(plate_id=1)          1) pull + scan the slides
    planner.prepare_well_assignment(plate_id=1)      2) well assignment, ROI/MS columns empty
    
    3) per slide: 
    ROISelector(...).write_lmd_and_metadata(planner.well_assignment_path(plate_id=1), patient_id)
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import re

@dataclass
class PlateConfig:
    wells_per_plate: int = 300   # usable wells per 384-well plate (outer wells excluded)
    disease_tiles: int = 5       # disease tiles per patient
    healthy_tiles: int = 5       # healthy tiles per patient
    seed: int = 0

    @property
    def tiles_per_patient(self) -> int:
        return self.disease_tiles + self.healthy_tiles

    @property
    def patients_per_plate(self) -> int:
        # Integer division: 300 usable wells // 10 tiles/patient = 30 patients/plate
        # Any remainder wells are unused
        return self.wells_per_plate // self.tiles_per_patient


# --------------------------------------------------------------------------
# Constants and small helpers
# --------------------------------------------------------------------------

SLIDES_PER_BOX = 25
PLATE_ROWS = "ABCDEFGHIJKLMNOP"   # 16 rows
PLATE_COLS = 24

ARMS = ("healthy", "disease")
SCORE_COLUMNS = ["attention", "contribution_score", "contrast_score"]
# lmd_shape_index = position of the tile in its arm's LMD XML (0-based, = cut order)
ROI_COLUMNS = ["slide_path", "lmd_xml", "lmd_shape_index", "tile_id", "notes"] + SCORE_COLUMNS   
_TEXT_COLUMNS = ["patient_id", "table_id", "slide_path", "tile_id", "lmd_xml", "notes"]

def _format_microscopy_nr(patient_id: object) -> str:
    digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Expected an 8-digit patient ID, got {patient_id!r}")
    return f"{digits[:2]}-{digits[2:]}"

def _table_id_number(table_id: object) -> int | None:
    """'PR86_0027' -> 27. Returns None if there is no trailing number."""
    table_id_string = str(table_id).strip()
    if not table_id_string or table_id_string.lower() == "nan":
        return None
    tail = table_id_string.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None

def _box_of(table_id: object, slides_per_box: int = SLIDES_PER_BOX) -> int | None:
    """Given a table_id like 'PR86_0027', return the box number (1-indexed)."""
    n = _table_id_number(table_id)
    return None if n is None else (n - 1) // slides_per_box + 1

def inner_wells_384() -> list[str]:
    """Wells B2..O23 in row-major order (308 wells; rows A/P and columns 1/24 excluded)."""
    return [f"{r}{c}" for r in PLATE_ROWS[1:-1] for c in range(2, PLATE_COLS)]

def well_quadrant(well: str) -> str:
    """
    384 -> 4x96 quadrant (interleaved pattern):
      Q1: rows A,C,E,... x odd columns    Q2: rows A,C,E,... x even columns
      Q3: rows B,D,F,... x odd columns    Q4: rows B,D,F,... x even columns
    """
    row_idx = PLATE_ROWS.index(well[0])     # A=0
    odd_col = int(well[1:]) % 2 == 1        # odd column number
    if row_idx % 2 == 0:                    # A, C, E, ...
        return "Q1" if odd_col else "Q2"
    return "Q3" if odd_col else "Q4"

def load_microscopy_table_mapping(excel_path: str, microscopy_col: str = "Microscopy number",
                                  table_id_col: str = "Table Id") -> pd.DataFrame:
    """Load microscopy number -> table ID mapping from Excel."""
    mapping = pd.read_excel(excel_path, usecols=[microscopy_col, table_id_col], dtype=str)
    mapping = mapping.rename(columns={microscopy_col: "microscopy_nr", table_id_col: "table_id"})
    mapping["microscopy_nr"] = mapping["microscopy_nr"].astype(str).str.strip()
    mapping["table_id"] = mapping["table_id"].astype(str).str.strip()
    return mapping.drop_duplicates(subset=["microscopy_nr"]).reset_index(drop=True)

def load_well_assignment(path: str) -> pd.DataFrame:
    # Text columns as object so IDs keep leading zeros and empty columns accept strings;
    # Shape index as nullable int so it stays an integer while some wells are empty.
    dtypes = {c: "object" for c in _TEXT_COLUMNS}
    dtypes["lmd_shape_index"] = "Int64"
    return pd.read_csv(path, dtype=dtypes)

def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written plate CSV."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def plate_to_lmd(well):
    m = re.fullmatch(r"([A-P])(\d{1,2})", str(well).strip().upper())
    if m is None:
        raise ValueError(f"Unrecognised well: {well}")
    row, col = m.group(1), int(m.group(2))
    if row == "A" or col == 1:
        raise ValueError(f"{well} has no LMD equivalent (row A / column 1)")
    return f"{chr(ord(row) - 1)}{col - 1}"

# df = df.rename(columns={"well": "plate_well"})
# df["LMD_well"] = df["plate_well"].map(plate_to_lmd)

# --------------------------------------------------------------------------
# Filling ROI columns (called by ROISelector after writing the LMD XML)
# --------------------------------------------------------------------------

def record_rois(well_assignment_path: str, patient_id: object, arm: str, tiles: pd.DataFrame,
                slide_path: str, lmd_xml: str | None, overwrite: bool = False) -> pd.DataFrame:
    """
    Write one arm's selected tiles into that patient's wells, in XML order.

    `tiles` must be the exact (Geo)DataFrame passed to the LMD XML, in the same
    row order: row k -> lmd_shape_index k -> the patient's k-th well for this arm.

    - Raises if the patient/arm has no wells on this plate, if there are more
      tiles than wells, or if the wells are already filled (unless overwrite=True).
    - Fewer tiles than wells: the extra wells stay empty and get a note.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    # Read the plate CSV fresh from disk every call (build on the latest state)
    df = load_well_assignment(well_assignment_path)

    # Get rows for this patient/arm
    pid = str(patient_id)
    rows = df.index[(df["patient_id"] == pid) & (df["arm"] == arm)]
    n_wells, n_tiles = len(rows), len(tiles)

    # No wells means this patient isn't on this plate: most likely the wrong plate CSV passed
    if n_wells == 0:
        raise KeyError(f"Patient {pid!r} has no {arm} wells in {well_assignment_path} - wrong plate?")

    # More tiles than wells (usually caused by ROISelector disease_k/healthy_k != PlateConfig)
    if n_tiles > n_wells:
        raise ValueError(
            f"Patient {pid!r} {arm}: {n_tiles} tiles selected but only {n_wells} wells. "
            "Set ROISelector disease_k/healthy_k to PlateConfig disease_tiles/healthy_tiles."
        )

    # Wells already have a tile_id: this slide was recorded and possibly cut before
    if not overwrite and df.loc[rows, "tile_id"].notna().any():
        raise ValueError(
            f"Patient {pid!r} {arm} wells are already filled in {well_assignment_path}. "
            "Pass overwrite=True to replace (make sure the old XML is replaced too)."
        )

    # Clear previous ROI values (matters on overwrite))
    df.loc[rows, ROI_COLUMNS] = np.nan  # Blank all ROI columns for these wells

    # Fill wells, one tile per well, in XML order
    used = rows[:n_tiles]
    tile_ids = tiles.index
    df.loc[used, "slide_path"] = slide_path            # which slide the cut came from
    df.loc[used, "lmd_xml"] = lmd_xml                  # which XML file holds the shape
    df.loc[used, "lmd_shape_index"] = np.arange(n_tiles)  # position of the shape in that XML (0-based)
    df.loc[used, "tile_id"] = [str(t) for t in tile_ids]  # as text, e.g. so IDs keep leading zeros

    # Copy ABMIL scores (attention, contribution_score, contrast_score)
    for col in SCORE_COLUMNS: 
        df.loc[used, col] = tiles[col].to_numpy() if col in tiles.columns else np.nan

    # Fewer tiles than wells (small pool, spacing constraint, ...)
    if n_tiles < n_wells:
        df.loc[rows[n_tiles:], "notes"] = f"no tile: only {n_tiles}/{n_wells} {arm} tiles selected"

    _write_csv_atomic(df, well_assignment_path)
    print(f"Recorded {n_tiles}/{n_wells} {arm} tiles for patient {pid} -> {well_assignment_path}")

    return df.loc[rows]

# --------------------------------------------------------------------------
# PlatePlanner
# --------------------------------------------------------------------------

class PlatePlanner:
    """
    Holds the cohort, the plate assignment and where the per-plate metadata
    CSVs live. Randomization runs once in __init__ (seeded via config.seed),
    so re-creating the planner with the same inputs gives the same plates.
    """
    def __init__(self, patients_df: pd.DataFrame, project_id: str, project_dir: str, 
                 id_col: str = "rekvnr", group_col: str = "diagnose", filename_col: str = "filename",
                 config: PlateConfig | None = None, mapping_excel: str | None = None, 
                 microscopy_mapping: pd.DataFrame | None = None):
         
        self.project_id = project_id
        self.project_dir = project_dir
        self.id_col = id_col
        self.group_col = group_col
        self.filename_col = filename_col
        self.config = config or PlateConfig()

        # Load microscopy number -> table ID mapping from Excel if not provided
        if microscopy_mapping is None and mapping_excel is not None:
            microscopy_mapping = load_microscopy_table_mapping(mapping_excel)
        self.mapping = microscopy_mapping
        self._lookup = {} if microscopy_mapping is None else dict(
            zip(microscopy_mapping["microscopy_nr"], microscopy_mapping["table_id"]))

        self.assigned_df = self._stratified_plate_assignment(patients_df)

    # --------------------------------------------------------------------------
    # Stratified randomization of patients to plates
    # --------------------------------------------------------------------------

    def _stratified_plate_assignment(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each patient to a plate so that every plate carries a
        representative mix of `group_col`, in roughly cohort proportions.

        Method: shuffle each group independently, then round-robin its members
        across the plates (each group's per-plate count within +/-1 of its
        proportional share). A rebalancing pass fixes any plate over capacity.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        df = patients_df.copy().reset_index(drop=True)
        df[self.id_col] = df[self.id_col].astype(str)

        n_patients = len(df)
        if n_patients == 0:
            raise ValueError("No patients to assign.")
        n_plates = int(np.ceil(n_patients / cfg.patients_per_plate))
        print(f"Number of patients: {n_patients}, tiles per patient: {cfg.tiles_per_patient}, plates needed: {n_plates}")

        # Placeholder column; -1 means "not yet assigned" (overwritten below)
        df["plate"] = -1

        for _, group_df in df.groupby(self.group_col): # Loop over each subtype/control group
            idx = group_df.index.to_numpy().copy()  # Row positions (in `df`) belonging to this one group.
            rng.shuffle(idx) # Shuffle just this group's patients into a random order

            # Round-robin the shuffled patients across the plates: patient 0 of the shuffled group
            # -> plate 1, patient 1 -> plate 2, ..., wrapping back to plate 1 after n_plates.
            df.loc[idx, "plate"] = (np.arange(len(idx)) % n_plates) + 1   # 1-indexed plates

        # Rebalance any plates that are over capacity
        self._rebalance_overflow(df, rng)

        return df.sort_values(["plate"]).reset_index(drop=True)

    def _rebalance_overflow(self, df: pd.DataFrame, rng: np.random.Generator) -> None:
        """Move patients off over-capacity plates onto under-capacity ones."""
        cap = self.config.patients_per_plate
        for _ in range(len(df)):
            counts = df["plate"].value_counts()
            over = counts[counts > cap] # plates over capacity
            if over.empty:
                return
            under = counts[counts < cap] # plates with spare room

            from_plate = over.index[0] # first over-capacity plate (arbitrary order, but deterministic)
            to_plate = under.index[0] if not under.empty else counts.idxmin() # assign to the first under-capacity plate, or if none exist, to the plate with the fewest patients (arbitrary tie-breaker)
            move_idx = rng.choice(df.index[df["plate"] == from_plate]) # pick one patient at random from the over-capacity plate
            df.loc[move_idx, "plate"] = to_plate # move patient 

    def summarize(self) -> pd.DataFrame:
        """Sanity-check table: patient count per group, per plate."""
        return self.assigned_df.groupby(["plate", self.group_col]).size().unstack(fill_value=0)

    @property
    def n_plates(self) -> int:
        return int(self.assigned_df["plate"].max())

    # --------------------------------------------------------------------------
    # Step 1: slide list per plate
    # --------------------------------------------------------------------------

    def plate_patients(self, plate_number: int) -> pd.DataFrame:
        """One row per patient on the plate, with microscopy_nr / table_id / box, sorted by table_id."""
        plate_df = self.assigned_df[self.assigned_df["plate"] == plate_number].drop_duplicates(subset=[self.id_col]).copy()
        if plate_df.empty:
            raise KeyError(f"Plate {plate_number} has no patients (plates 1-{self.n_plates}).")
        plate_df["microscopy_nr"] = plate_df[self.id_col].apply(_format_microscopy_nr)
        plate_df["table_id"] = plate_df["microscopy_nr"].map(self._lookup).fillna("")
        plate_df["box"] = plate_df["table_id"].apply(_box_of).astype("Int64")
        plate_df["_table_num"] = plate_df["table_id"].apply(_table_id_number)
        # Patients without a table_id go last
        plate_df = plate_df.sort_values(["_table_num", self.id_col], na_position="last").drop(columns="_table_num")
        return plate_df.reset_index(drop=True)

    def get_rekvnr_on_plate(self, plate_number: int) -> pd.DataFrame:
        """Print the slides to pull for one plate, grouped by storage box."""
        plate_df = self.plate_patients(plate_number)
        print(f"Plate: {plate_number}, Patients: {len(plate_df)}\n")
        for box, box_df in plate_df.groupby("box", dropna=False, sort=False):
            header = f"Box {box}" if pd.notna(box) else "No table_id (box unknown)"
            print(f"{header}  ({len(box_df)} slides)")
            for _, row in box_df.iterrows():
                t_id = f"  -  {row['table_id']}" if row["table_id"] else ""
                print(f"  {row[self.id_col]}  -  {row['microscopy_nr']}{t_id}")
            print("-" * 40)
        return plate_df

    # --------------------------------------------------------------------------
    # Step 2: prepare per-plate metadata CSV
    # --------------------------------------------------------------------------

    def well_assignment_path(self, plate_number: int) -> str:
        plate_dir = os.path.join(self.project_dir, f"P{plate_number}")
        return os.path.join(plate_dir, f"{self.project_id}_well_assignment_P{plate_number}.csv")

    def prepare_well_assignment(self, plate_number: int) -> pd.DataFrame:
        """
        Build the well assignment CSV for one plate.

        Patients in the same order as get_rekvnr_on_plate(); per patient
        healthy_tiles healthy rows then disease_tiles disease rows, in
        consecutive inner wells from B2. 
        
        ROI and MS columns are left empty (ROI columns are filled by record_rois).
        """
        cfg = self.config
        path = self.well_assignment_path(plate_number)
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists - delete file if you want to rebuild.")

        patients = self.plate_patients(plate_number)
        n_rows = len(patients) * cfg.tiles_per_patient
        if n_rows > cfg.wells_per_plate:
            raise ValueError(f"Plate {plate_number} needs {n_rows} wells, only {cfg.wells_per_plate} usable.")

        # Expand each patient into its healthy + disease rows
        cols_to_keep = [self.id_col, "microscopy_nr", "table_id", self.group_col]
        df = patients.loc[patients.index.repeat(cfg.tiles_per_patient), cols_to_keep].reset_index(drop=True)
        df = df.rename(columns={self.id_col: "patient_id"})
        df["arm"] = (["healthy"] * cfg.healthy_tiles + ["disease"] * cfg.disease_tiles) * len(patients)

        # Add columns at specific positions
        wells = inner_wells_384()[:n_rows]
        df.insert(0, "plate", plate_number)
        df.insert(1, "plate_well", wells)
        df.insert(2, "LMD_well", [plate_to_lmd(w) for w in wells])
        df.insert(3, "quadrant", [well_quadrant(w) for w in wells])
        df.insert(0, "run_id", [f"{self.project_id}_P{plate_number}_{well_quadrant(w)}_{w}" for w in wells])

        for col in ROI_COLUMNS:
            df[col] = np.nan

        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_csv_atomic(df, path)
        print(f"Wrote {len(df)} rows ({len(patients)} patients) for plate {plate_number} -> {path}")
        return df