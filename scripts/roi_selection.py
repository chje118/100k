import json
import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
from wsidata import open_wsi
import geopandas as gpd
from spatialdata.models import PointsModel, ShapesModel
from dvpio.write import write_lmd
import os

class ROISelector:
    """ Handle ROI selection from cached ABMIL inference results. """
    def __init__(self, cache_path: str, slide_path: str, top_k: int = 20, bottom_k: int = 10, top_pct: float = 0.10, bottom_pct: float = 0.10, random_state: int | None = 42):
        self.cache_path = cache_path
        self.slide_path = slide_path
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.random_state = random_state
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
        """Return random top-k and bottom-k tiles sampled from the top/bottom x% of attention scores.
        Sorted by shortest path (greedy nearest-neighbor from top-left corner)."""
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

        top_pool = ranked_desc.head(top_pool_n).copy()
        bottom_pool = ranked_asc.head(bottom_pool_n).copy()

        top_sample_n = min(self.top_k, len(top_pool))
        bottom_sample_n = min(self.bottom_k, len(bottom_pool))

        top_tiles = top_pool.sample(
            n=top_sample_n,
            random_state=self.random_state,
            replace=False,
        ).copy()
        
        bottom_tiles = bottom_pool.sample(
            n=bottom_sample_n,
            random_state=self.random_state,
            replace=False,
        ).copy()

        # Sort each by shortest path (greedy nearest-neighbor)
        top_tiles = self._sort_tiles_tsp(top_tiles)
        bottom_tiles = self._sort_tiles_tsp(bottom_tiles)

        return gpd.GeoDataFrame(top_tiles, geometry="geometry"), gpd.GeoDataFrame(bottom_tiles, geometry="geometry")

    def _sort_tiles_tsp(self, tiles_gdf):
        """Sort tiles by greedy nearest-neighbor from top-left corner."""
        if len(tiles_gdf) <= 1:
            return tiles_gdf
    
        tiles_gdf = tiles_gdf.copy()
        
        if not isinstance(tiles_gdf, gpd.GeoDataFrame):
            tiles_gdf = gpd.GeoDataFrame(tiles_gdf, geometry="geometry")

        # Compute centroids
        centroids = tiles_gdf.geometry.centroid
        tiles_gdf["centroid_x"] = centroids.x
        tiles_gdf["centroid_y"] = centroids.y
    
        # Start from top-left tile (min y, then min x)
        tiles_gdf["sort_key"] = tiles_gdf["centroid_y"] * 1e6 + tiles_gdf["centroid_x"]
        current_idx = tiles_gdf["sort_key"].idxmin()
        tiles_gdf = tiles_gdf.drop(columns=["sort_key"])
    
        # Greedy nearest-neighbor traversal
        remaining = set(tiles_gdf.index)
        order = []
    
        while remaining:
            order.append(current_idx)
            remaining.remove(current_idx)
        
            if not remaining:
                break
        
            # Find nearest unvisited tile
            current_x = tiles_gdf.loc[current_idx, "centroid_x"]
            current_y = tiles_gdf.loc[current_idx, "centroid_y"]
        
            remaining_gdf = tiles_gdf.loc[list(remaining)]
            distances = np.sqrt(
                (remaining_gdf["centroid_x"] - current_x)**2 + 
                (remaining_gdf["centroid_y"] - current_y)**2
            )
            current_idx = distances.idxmin()
    
        # Reorder by traversal path
        tiles_sorted = tiles_gdf.loc[order].drop(columns=["centroid_x", "centroid_y"])
    
        return tiles_sorted

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

    # VIEWER SELECT CALBIRATION POINTS
    # SAVED TO JSON FILE

    def set_paths(self, annotations_path: str, lmd_dir: str):
        self.annotations_path = annotations_path
        self.lmd_dir = lmd_dir
        os.makedirs(self.lmd_dir, exist_ok=True)
        print(f"Set annotations path: {self.annotations_path}")
        print(f"Set LMD directory: {self.lmd_dir}")

    def _open_annotations(self):
        if not hasattr(self, "annotations_path"):
            raise AttributeError("Annotations path not set. Call set_paths() first.")
        
        with open(self.annotations_path, "r") as f:
            self.ann = json.load(f)

    def add_calibration_points(self):
        self._open_annotations()

        # Extract calibration points as an (N, 2) array in image coordinates
        cal_points = self.ann.get("calibration_points", [])
        coords = np.array([[p["x"], p["y"]] for p in cal_points], dtype=float)

        self.sdata_lmd = self.get_sdata_lmd()

        # Add to spatialdata object
        self.sdata_lmd.points["calibration_points"] = PointsModel.parse(coords)
        return self.sdata_lmd

    def write_to_lmd(self):
        slide_name = os.path.splitext(os.path.basename(self.slide_path))[0]
        try:
            for tiles in ["top", "bottom"]:
                path_lmd = os.path.join(self.lmd_dir, slide_name, f'{slide_name}_{tiles}.xml')
                
                # Transform coordinates to LMD coordinate system
                H = self.sdata_lmd.images["wsi_thumbnail"].data.shape[1]
                
                affine_transformation = np.array([
                    [1,  0, 0],
                    [0, -1, H],
                    [0,  0, 1]
                ])

                # Write LMD file with tiles and calibration points
                write_lmd(
                    path = path_lmd,
                    annotation = self.sdata_lmd.shapes[f"{tiles}_tiles"],
                    calibration_points=self.sdata_lmd.points["calibration_points"],
                    affine_transformation=affine_transformation
                )
                print(f"Wrote LMD file for {tiles} tiles: {path_lmd}")

        except Exception as e:
            print(f"Error writing LMD files: {e}")