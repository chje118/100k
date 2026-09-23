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
    from plate_randomization import PlateConfig, build_acquisition_plan
    plan = build_acquisition_plan(patients_df, id_col="patient_id",
                                   group_col="subtype", config=PlateConfig())
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

    return df.sort_values(["plate", group_col, id_col]).reset_index(drop=True)


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


