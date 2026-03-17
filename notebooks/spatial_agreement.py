import os
from wsidata import open_wsi
import pandas as pd


class SpatialAgreement:
    def __init__(self, filenames, zarr_dir, models, tile_key = 'tiles_224'):
        self.filenames = filenames
        self.zarr_dir = zarr_dir
        self.models = models
        self.tile_key = tile_key

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
    
    def run(self):
        for path in self.filenames:
            wsi = self.load_slide(path)
            gdf = self.get_shapes(wsi)
