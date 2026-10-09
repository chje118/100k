"""
Slide randomization and cutting plan for DVP LMD / plate prep.

Scope: slide-to-plate randomization and LMD cutting order.

What it does:
- Patients are assigned to plates via STRATIFIED RANDOMIZATION by disease
  subtype / control group, so every plate carries a representative mix
  (controls technical/batch variation across plates). This is the
  "slide randomization" - which patients land on which plate is randomized,
  but the resulting mix on every plate is still representative.
- A patient's FULL tile set (disease + healthy) always stays on one plate,
  so plate-level batch effects cancel in the paired within-patient
  comparison (the primary analysis).

Usage as a library:
    from plate_randomization import PlateConfig, stratified_plate_assignment, summarize_plate_composition
        assigned_df = stratified_plate_assignment(patients_df, id_col="patient_id", group_col="subtype", config=PlateConfig())
        print(summarize_plate_composition(assigned_df, group_col="subtype"))
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


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
# Patient -> plate assignment ("slide randomization")
# --------------------------------------------------------------------------

def stratified_plate_assignment(patients_df: pd.DataFrame, id_col: str, group_col: str,
                                 config: PlateConfig) -> pd.DataFrame:
    """
    Assign each patient to a plate so that every plate carries a
    representative mix of `group_col` (disease subtype / control status),
    in roughly the same proportions as the full cohort.

    Method: shuffle each group independently, then round-robin its members
    across the plates. This keeps every group's per-plate count within +/-1
    of its proportional share. A small rebalancing pass fixes any plate
    that still ends up over capacity when group sizes don't divide evenly.

    Returns `patients_df` with an added 1-indexed 'plate' column.
    """
    rng = np.random.default_rng(config.seed)
    df = patients_df.copy().reset_index(drop=True)

    n_patients = len(df)

    # Plates needed, rounded up to whole plates
    n_plates = int(np.ceil(n_patients / config.patients_per_plate))
    if n_plates < 1:
        raise ValueError("No patients to assign.")
    print(f"Number of patients: {n_patients}, tiles per patient: {config.tiles_per_patient}, plates needed: {n_plates} ")

    # Placeholder column; -1 means "not yet assigned" (overwritten below)
    df["plate"] = -1

    # Core Stratification Step
    for _, group_df in df.groupby(group_col): # Loop over each subtype/control group
        idx = group_df.index.to_numpy().copy() # Row positions (in `df`) belonging to this one group.
        rng.shuffle(idx) # Shuffle just THIS group's patients into a random order

        # Round-robin the shuffled patients across the plates: 
        # patient 0 of the shuffled group -> plate 1, patient 1 -> plate 2, 
        # ..., wrapping back to plate 1 after n_plates. 
        plate_for_each = (np.arange(len(idx)) % n_plates) + 1  # +1: plates are 1-indexed, not 0-indexed
        df.loc[idx, "plate"] = plate_for_each

    # Rebalance any plates that are over capacity
    _rebalance_overflow(df, config, rng)

    return df.sort_values(["plate"]).reset_index(drop=True)


def _rebalance_overflow(df: pd.DataFrame, config: PlateConfig, rng: np.random.Generator) -> None:
    """Move patients off over-capacity plates onto under-capacity ones."""
    cap = config.patients_per_plate

    for _ in range(len(df)):
        counts = df["plate"].value_counts()
        over = counts[counts > cap]     # plates over capacity
        if over.empty:
            return
        under = counts[counts < cap]    # plates with spare room

        from_plate = over.index[0]      # pick the first over-capacity plate (arbitrary order, but deterministic)
        to_plate = under.index[0] if not under.empty else counts.idxmin() # assign to the first under-capacity plate, or if none exist, to the plate with the fewest patients (arbitrary tie-breaker)
        movable = df.index[df["plate"] == from_plate]   # candidate rows to move
        move_idx = rng.choice(movable)                  # pick one patient at random from the over-capacity plate
        df.loc[move_idx, "plate"] = to_plate            # move patient


def summarize_plate_composition(plate_assignment: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Sanity-check table: patient count per group, per plate."""
    return plate_assignment.groupby(["plate", group_col]).size().unstack(fill_value=0)


def _format_microscopy_nr(patient_id: object) -> str:
    digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Expected an 8-digit patient ID, got {patient_id!r}")
    return f"{digits[:2]}-{digits[2:]}"


def load_microscopy_table_mapping(excel_path: str, microscopy_col: str = "Microscopy number", table_id_col: str = "Table Id") -> pd.DataFrame:
    """Load microscopy number -> table ID mapping from Excel."""
    mapping = pd.read_excel(excel_path, usecols=[microscopy_col, table_id_col], dtype=str)
    mapping = mapping.rename(columns={microscopy_col: "microscopy_nr", table_id_col: "table_id"})
    mapping["microscopy_nr"] = mapping["microscopy_nr"].astype(str).str.strip()
    mapping["table_id"] = mapping["table_id"].astype(str).str.strip()
    return mapping.drop_duplicates(subset=["microscopy_nr"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Box helpers (slides are stored 25 per box: PR86_0001-0025 = box 1, ...)
# --------------------------------------------------------------------------

SLIDES_PER_BOX = 25

def _table_id_number(table_id: object) -> int | None:
    """'PR86_0027' -> 27. Returns None if there is no trailing number."""
    table_id_string = str(table_id).strip()
    if not table_id_string or table_id_string.lower() == "nan":
        return None
    tail = table_id_string.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None

def _box_of(table_id: object, slides_per_box: int = SLIDES_PER_BOX) -> int | None:
    """Given a table_id like 'PR86_0027', return the box number (1-indexed)."""
    table_id = _table_id_number(table_id)
    return None if table_id is None else (table_id - 1) // slides_per_box + 1

def _plate_patients(plate_assignment: pd.DataFrame, id_col: str, plate_number: int,
                    microscopy_mapping: pd.DataFrame | None) -> pd.DataFrame:
    """One row per patient on `plate_number`, with microscopy_nr / table_id / box, sorted by table_id."""
    lookup = {}
    if microscopy_mapping is not None:
        lookup = dict(zip(microscopy_mapping["microscopy_nr"], microscopy_mapping["table_id"]))

    df = plate_assignment[plate_assignment["plate"] == plate_number].drop_duplicates(subset=[id_col]).copy()
    df["microscopy_nr"] = df[id_col].apply(_format_microscopy_nr)
    df["table_id"] = df["microscopy_nr"].map(lookup).fillna("")
    df["box"] = df["table_id"].apply(_box_of).astype("Int64")
    df["_table_num"] = df["table_id"].apply(_table_id_number)
    # Patients without a table_id go last
    df = df.sort_values(["_table_num", id_col], na_position="last").drop(columns="_table_num")
    return df.reset_index(drop=True)

def get_rekvnr_on_plate(plate_assignment: pd.DataFrame, id_col: str, plate_number: int,
                        microscopy_mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Print the slides to pull for one plate, grouped by storage box
    (25 slides per box, from the table_id number), so you can pull one box at a time.
    Returns the per-patient table as well.
    """
    plate_df = _plate_patients(plate_assignment, id_col, plate_number, microscopy_mapping)
    n_tiles = int((plate_assignment["plate"] == plate_number).sum())
    print(f"Plate: {plate_number}, Patients: {len(plate_df)}, Rows in assignment: {n_tiles}\n")

    for box, box_df in plate_df.groupby("box", dropna=False, sort=False):
        header = f"Box {box}" if pd.notna(box) else "No table_id (box unknown)"
        print(f"{header}  ({len(box_df)} slides)")
        for _, row in box_df.iterrows():
            tid = f"  -  {row['table_id']}" if row["table_id"] else ""
            print(f"  {row[id_col]}  -  {row['microscopy_nr']}{tid}")
        print("-" * 40)

    return plate_df



def prepare_blinded_plate_assignment(patients_df: pd.DataFrame, id_col: str, group_col: str,
                                 config: PlateConfig) -> pd.DataFrame:
    # Assign patients to plates
    assigned_df = stratified_plate_assignment(patients_df, id_col=id_col, group_col=group_col, config=config)

    # Keep only id and plate columns (blinded to group)
    cols_to_keep = [id_col, "plate"]
    blinded_df = assigned_df[cols_to_keep]

    # Prepare table with one row per tile (disease + healthy) for each patient
    blinded_df = pd.DataFrame(np.repeat(blinded_df.values, config.tiles_per_patient, axis=0), columns=blinded_df.columns)
    blinded_df = blinded_df.reset_index(drop=True)
    blinded_df["tile_group"] = np.tile(["healthy"] * config.healthy_tiles + ["disease"] * config.disease_tiles, len(blinded_df) // config.tiles_per_patient)

    # Add empty col for well assignment
    blinded_df["well"] = ""

    return blinded_df