import os
from wsidata import open_wsi


class SpatialAgreement:
    def __init__(self, filenames, zarr_dir, models):
        self.filenames = filenames
        self.zarr_dir = zarr_dir
        self.models = models

    def load_slide(self, path):
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(path, zarr_path)
        return wsi
    
    def run(self):
        for path in self.filenames:
            wsi = self.load_slide(path)

