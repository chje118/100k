import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
from wsidata import open_wsi
import geopandas as gpd
from spatialdata.models import ShapesModel

class ROISelector:
    """ Handle ROI selection from cached ABMIL inference results. """
    def __init__(self, cache_path: str, slide_path: str, top_k: int = 20, bottom_k: int = 10, top_pct: float = 0.10, bottom_pct: float = 0.10):
        self.cache_path = cache_path
        self.slide_path = slide_path
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.slide_cache = self.load_cache(cache_path)
        self.slide_data = self.get_slide_data()

    @staticmethod
    def load_cache(input_path: str):
        with open(input_path, "rb") as f:
            cached = pickle.load(f)
        return cached.get("slide_cache", cached) if isinstance(cached, dict) else {}

    @staticmethod
    def _get_pool_size(n_tiles: int, pct: float) -> int:
        return max(1, int(np.ceil(n_tiles * pct)))

    def get_slide_data(self):
        slide_data = self.slide_cache.get(self.slide_path)
        if slide_data is None:
            raise KeyError(f"No cached data found for slide: {self.slide_path}")
        return slide_data

    def get_tiles_gdf(self):
        """Return top-k and bottom-k tiles from the top/bottom x% of attention scores."""
        tile_table = self.slide_data.get("tile_table")
        if tile_table is None:
            raise KeyError("slide_data must contain 'tile_table'")

        tile_table = tile_table.dropna(subset=["attention", "geometry"]).copy()
        n_tiles = len(tile_table)

        if n_tiles == 0:
            empty = gpd.GeoDataFrame(tile_table, geometry="geometry")
            return empty, empty

        top_pool_n = self._get_pool_size(n_tiles, self.top_pct)
        bottom_pool_n = self._get_pool_size(n_tiles, self.bottom_pct)
    
        ranked_desc = tile_table.sort_values("attention", ascending=False)
        ranked_asc = tile_table.sort_values("attention", ascending=True)

        top_pool = ranked_desc.head(top_pool_n)
        bottom_pool = ranked_asc.head(bottom_pool_n)

        top_tiles = top_pool.head(self.top_k).copy()
        bottom_tiles = bottom_pool.head(self.bottom_k).copy()

        return gpd.GeoDataFrame(top_tiles, geometry="geometry"), gpd.GeoDataFrame(bottom_tiles, geometry="geometry")

    def zoomed_view(self, margin: int = 0, max_tiles: int = 4, top: bool = True):
        """ Plot zoomed tiles (grid) for review from cached slide data. """
        top_tiles_gdf, bottom_tiles_gdf = self.get_tiles_gdf()
        tiles_gdf = top_tiles_gdf if top else bottom_tiles_gdf

        if max_tiles is not None:
            tiles_gdf = tiles_gdf.head(max_tiles)

        slide_path = self.slide_data["slide_path"]
        zarr_path = self.slide_data["zarr_path"]
        tile_key = self.slide_data["tile_key"]
        wsi = open_wsi(slide_path, zarr_path)

        cols = 2
        rows = int(np.ceil(len(tiles_gdf) / cols)) if len(tiles_gdf) > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.asarray(axes).flatten()

        for i, (_, tile_row) in enumerate(tiles_gdf.iterrows()):
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

        for j in range(len(tiles_gdf), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def tiles_to_cut(self):
        """ Plot the full slide and highlight the top-k and bottom-k tiles to cut with red and blue. """
        top_tiles_gdf, bottom_tiles_gdf = self.get_tiles_gdf()
        wsi = self.get_wsi()

        fig, ax = plt.subplots(figsize=(12, 12))
        zs.pl.tissue(
            wsi,
            ax=ax,
            show_contours=True,
        )
        top_tiles_gdf.plot(
            ax=ax,
            facecolor="#ff2d2d",
            edgecolor="#b30000",
            linewidth=1.5,
            alpha=0.35,
        )
        bottom_tiles_gdf.plot(
            ax=ax,
            facecolor="#2d2dff",
            edgecolor="#0000b3",
            linewidth=1.5,
            alpha=0.35,
        )
        ax.set_title(f"Top {len(top_tiles_gdf)} (red) and bottom {len(bottom_tiles_gdf)} (blue) tiles selected for cutting")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def get_wsi(self):
        slide_path = self.slide_data["slide_path"]
        zarr_path = self.slide_data["zarr_path"]
        self.wsi = open_wsi(slide_path, zarr_path)
        return self.wsi
    
    def get_sdata_lmd(self):
        self.sdata = self.wsi.to_spatialdata()
        top_gdf, bottom_gdf = self.get_tiles_gdf()
        self.sdata.shapes["top_tiles"] = ShapesModel.parse(top_gdf)
        self.sdata.shapes["bottom_tiles"] = ShapesModel.parse(bottom_gdf)
        export_layers = ["wsi_thumbnail", "top_tiles", "bottom_tiles"]
        sdata_lmd = self.sdata.subset(element_names=export_layers)
        return sdata_lmd

    def napari_polygons(self):
        """ Return list of polygon coordinate arrays suitable for Napari `add_shapes`."""
        top_tiles_gdf, bottom_tiles_gdf = self.get_tiles_gdf()
        polygons_top = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in top_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        polygons_bottom = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in bottom_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        return polygons_top, polygons_bottom


    def viewer_polygons(self):
        """ Return top/bottom tile outlines as closed polygons in standard image coordinate order: (x, y).

        Each tile is returned as a 5-point closed ring:
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)].
        """
        top_tiles_gdf, bottom_tiles_gdf = self.get_tiles_gdf()

        def geom_to_polygon_xy(geom):
            if geom is None or geom.is_empty:
                return None

            minx, miny, maxx, maxy = geom.bounds
            return np.array([
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ], dtype=float)

        polygons_top = []
        for geom in top_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_top.append(poly)

        polygons_bottom = []
        for geom in bottom_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_bottom.append(poly)

        return polygons_top, polygons_bottom