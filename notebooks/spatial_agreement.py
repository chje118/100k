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

    def align_domains(self, gdf):
        # automatically generate domain names based on models
        cont_table = pd.crosstab(gdf['domain_conch'], gdf['domain_uni'])
        cont_table
    
    def run(self):
        for path in self.filenames:
            wsi = self.load_slide(path)
            gdf = self.get_shapes(wsi)
