import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
from wsidata import open_wsi
import geopandas as gpd
import napari

class ROISelector:
    """ Handle ROI selection from cached ABMIL inference results. """
    def __init__(self, cache_path: str, slide_path: str, top_k: int = 100):
        self.cache_path = cache_path
        self.slide_path = slide_path
        self.top_k = top_k
        self.slide_cache = self.load_cache(cache_path)
        self.slide_data = self.get_slide_data()

    @staticmethod
    def load_cache(input_path: str):
        with open(input_path, "rb") as f:
            cached = pickle.load(f)
        return cached.get("slide_cache", cached) if isinstance(cached, dict) else {}

    def get_slide_data(self):
        slide_data = self.slide_cache.get(self.slide_path)
        if slide_data is None:
            raise KeyError(f"No cached data found for slide: {self.slide_path}")
        return slide_data

    def top_tiles_from_slide_data(self) -> gpd.GeoDataFrame:
        """ Return the top-k tiles from cached slide data. """
        tile_table = self.slide_data.get("tile_table")
        if tile_table is None:
            raise KeyError("slide_data must contain 'tile_table'")
        top_tiles = tile_table.sort_values("attention", ascending=False).head(self.top_k).copy()
        return gpd.GeoDataFrame(top_tiles, geometry="geometry")

    def zoomed_view(self, margin: int = 0, max_tiles: int = 4):
        """ Plot zoomed tiles (grid) for review from cached slide data. """
        top_tiles_gdf = self.top_tiles_from_slide_data()
        if max_tiles is not None:
            top_tiles_gdf = top_tiles_gdf.head(max_tiles)

        slide_path = self.slide_data["slide_path"]
        zarr_path = self.slide_data["zarr_path"]
        tile_key = self.slide_data["tile_key"]
        wsi = open_wsi(slide_path, zarr_path)

        cols = 2
        rows = int(np.ceil(len(top_tiles_gdf) / cols)) if len(top_tiles_gdf) > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.asarray(axes).flatten()

        for i, (_, tile_row) in enumerate(top_tiles_gdf.iterrows()):
            geometry = tile_row["geometry"]
            if not hasattr(geometry, "bounds"):
                axes[i].axis("off")
                continue

            minx, miny, maxx, maxy = geometry.bounds
            xmin = minx - margin
            ymin = miny - margin
            xmax = maxx + margin
            ymax = maxy + margin

            zs.pl.tiles(wsi, tile_key=tile_key, zoom=(xmin, xmax, ymin, ymax), ax=axes[i])
            axes[i].set_title(f"tile {tile_row.get('tile_id', i)}")

        for j in range(len(top_tiles_gdf), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def get_wsi(self):
        slide_path = self.slide_data["slide_path"]
        zarr_path = self.slide_data["zarr_path"]
        self.wsi = open_wsi(slide_path, zarr_path)
        return self.wsi
    
    def get_sdata(self):
        self.sdata = self.wsi.to_spatialdata()
        return self.sdata

    def napari_polygons(self):
        """ Return list of polygon coordinate arrays suitable for Napari `add_shapes`."""
        top_tiles_gdf = self.top_tiles_from_slide_data()
        polygons = [
            np.array(g.exterior.coords)
            for g in top_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        return polygons