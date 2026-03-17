import os
from wsidata import open_wsi
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np


class SpatialAgreement:
    def __init__(self, filenames, zarr_dir, models, tile_key = 'tiles_224'):
        self.filenames = filenames
        self.zarr_dir = zarr_dir
        self.models = models
        self.tile_key = tile_key
        self.dom_cols = [f"domain_{m}" for m in self.models]

    def load_slide(self, path):
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(path, zarr_path)
        return wsi

    def get_shapes(self, wsi):
        gdf = wsi.shapes[self.tile_key]
        return gdf

    def contingency_tabels(self, gdf):
        """
        Build contingency tables using the first model as reference.

        Expects `gdf` to contain one column per model named `domain_{model}`.
        Returns a dict mapping each non-reference model -> crosstab DataFrame.
        """
        dom_cols = [f"domain_{m}" for m in self.models]
        missing = [c for c in dom_cols if c not in gdf.columns]
        if missing:
            raise KeyError(f"Missing expected domain columns in gdf: {missing}")

        ref_model = self.models[0]
        ref_col = f"domain_{ref_model}"

        cont_tables = {}
        for model in self.models[1:]:
            col = f"domain_{model}"
            tab = pd.crosstab(gdf[ref_col], gdf[col], dropna=False)
            cont_tables[model] = tab

        return cont_tables

    def domain_alignment(self, gdf, cont_tables):
        """
        Align domain labels across models by remapping each non-reference model's
        domain IDs onto the reference model's domain IDs.

        - Reference model is `self.models[0]`.
        - Uses a max-overlap one-to-one assignment (Hungarian algorithm) on the
          contingency table.
        - Any remaining (unassigned) target labels are mapped to the reference
          label with the largest overlap (many-to-one fallback).

        Returns a `gdf_aligned` where each `domain_{model}` column has been
        remapped to the reference domain ID space.
        """
        ref_model = self.models[0]
        ref_col = f"domain_{ref_model}"

        gdf_aligned = gdf.copy()
        for model in self.models[1:]:
            tab = cont_tables.get(model)
            if tab is None:
                raise KeyError(f"Missing contingency table for model '{model}'.")

            # tab rows are reference labels, columns are target labels
            cost = -tab.to_numpy(dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping = {tab.columns[c]: tab.index[r] for r, c in zip(row_ind, col_ind)}

            # fallback for any target labels not covered by the 1:1 assignment
            for target_label in tab.columns:
                if target_label in mapping:
                    continue
                col_counts = tab[target_label]
                if col_counts.sum() == 0: # no overlap -> NaN
                    mapping[target_label] = np.nan
                else: # assign to best match
                    mapping[target_label] = col_counts.idxmax()

            target_col = f"domain_{model}"
            gdf_aligned[target_col] = gdf_aligned[target_col].map(mapping)

        gdf_aligned[ref_col] = gdf_aligned[ref_col]
        return gdf_aligned

    def agreement_level(self, gdf_aligned):
        """
        Compute per-tile agreement across aligned domain columns.

        Expects `gdf_aligned` to contain `domain_{model}` for each model in
        `self.models`, where all non-reference columns have already been remapped
        into the reference model's label space.

        Produces `agreement_level` = number of agreeing pairs among models.
        """
        agree = np.zeros(len(gdf_aligned), dtype=int)
        for i in range(len(self.dom_cols)):
            a = gdf_aligned[self.dom_cols[i]]
            for j in range(i + 1, len(self.dom_cols)):
                b = gdf_aligned[self.dom_cols[j]]
                agree += (a == b).astype(int).to_numpy()

        gdf = gdf_aligned.copy()
        gdf["agreement_level"] = agree
        return gdf

    def agreement_map(self, slide_idx):
        raise NotImplementedError("agreement_map is not implemented yet.")

    def agreement_dict(self):
        agreement_dict = {}

        for i, path in enumerate(self.filenames):
            wsi = self.load_slide(path)
            gdf = self.get_shapes(wsi)
            cont_tables = self.contingency_tabels(gdf)
            gdf_aligned = self.domain_alignment(gdf, cont_tables)
            gdf_aligned = self.agreement_level(gdf_aligned)
            agreement_dict[i] = gdf_aligned

        return agreement_dict
