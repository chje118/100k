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
    """ Handle ROI selection from cached ABMIL inference results.

    Selection is driven by `contrast_score` (the classifier's own read of
    each tile's raw features: + = disease-like, - = healthy-like). 
    
    Attention tells how much a tile influenced the slide-level prediction.

    `min_attention_pct` is used as a relevance floor. Tiles
    in the bottom `min_attention_pct` of attention for this slide are
    dropped from BOTH the disease and healthy candidate pools before
    ranking, since those are the tiles the model essentially ignored and
    are the most likely to be background, blur or folds - not a
    class judgement.
    """
    def __init__(self, cache_path: str, slide_path: str, healthy_k: int = 20, non_healthy_k: int = 10, healthy_pct: float = 0.10, non_healthy_pct: float = 0.10, random_state: int | None = 42, score_col: str = "contrast_score", min_attention_pct: float = 5.0):
        self.cache_path = cache_path
        self.slide_path = slide_path
        self.healthy_k = healthy_k
        self.non_healthy_k = non_healthy_k
        self.healthy_pct = healthy_pct
        self.non_healthy_pct = non_healthy_pct
        self.random_state = random_state
        self.score_col = score_col
        self.min_attention_pct = min_attention_pct
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
        """ Return random healthy-k and non-healthy-k tiles sampled from the 
        top/bottom x% of `self.score_col` (contrast_score by default: + = disease-like, 
        - = healthy-like), restricted to tiles that cleared the attention
        relevance floor. Sorted by shortest path (greedy nearest-neighbor
        from top-left corner). """
        tile_table = self.slide_data.get("tile_table")
        if tile_table is None:
            raise KeyError("slide_data must contain 'tile_table'")

        required_cols = ["attention", self.score_col, "geometry"]
        missing = [c for c in required_cols if c not in tile_table.columns]
        if missing:
            raise KeyError(
                f"slide_data['tile_table'] is missing column(s) {missing}. "
                "Run ABMIL inference first. "
            )

        tile_table = tile_table.dropna(subset=required_cols).copy()
        n_tiles = len(tile_table)

        if n_tiles == 0:
            empty = gpd.GeoDataFrame(tile_table, geometry="geometry")
            return empty, empty

        # Relevance floor: drop the least-attended tiles from BOTH pools so
        # ignored/background tiles can't masquerade as "healthy" just for
        # having a low or negative contrast_score by chance.
        if self.min_attention_pct and self.min_attention_pct > 0:
            attn_floor = np.percentile(tile_table["attention"], self.min_attention_pct)
            candidate_table = tile_table[tile_table["attention"] >= attn_floor].copy()
            if candidate_table.empty:  # guard against a degenerate/uniform attention distribution
                candidate_table = tile_table
        else:
            candidate_table = tile_table

        n_candidates = len(candidate_table)
        healthy_pool_n = self._get_pool_size(n_candidates, self.top_pct)
        non_healthy_pool_n = self._get_pool_size(n_candidates, self.bottom_pct)

        ranked_desc = candidate_table.sort_values(self.score_col, ascending=False)  # most disease-like first
        ranked_asc = candidate_table.sort_values(self.score_col, ascending=True)    # most healthy-like first

        healthy_pool = ranked_desc.head(healthy_pool_n).copy()
        non_healthy_pool = ranked_asc.head(non_healthy_pool_n).copy()

        healthy_sample_n = min(self.healthy_k, len(healthy_pool))
        non_healthy_sample_n = min(self.non_healthy_k, len(non_healthy_pool))

        healthy_tiles = healthy_pool.sample(
            n=healthy_sample_n,
            random_state=self.random_state,
            replace=False,
        ).copy()

        non_healthy_tiles = non_healthy_pool.sample(
            n=non_healthy_sample_n,
            random_state=self.random_state,
            replace=False,
        ).copy()
    
        # Sort each by shortest path (greedy nearest-neighbor)
        healthy_tiles = self._sort_tiles_tsp(healthy_tiles)
        non_healthy_tiles = self._sort_tiles_tsp(non_healthy_tiles)

        return gpd.GeoDataFrame(healthy_tiles, geometry="geometry"), gpd.GeoDataFrame(non_healthy_tiles, geometry="geometry")

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

    def zoomed_view(self, margin: int = 0, max_tiles: int = 4, healthy: bool = True):
        """ Plot zoomed tiles (grid) for review from cached slide data. """
        healthy_tiles_gdf, non_healthy_tiles_gdf = self.get_tiles_gdf()
        tiles_gdf = healthy_tiles_gdf if healthy else non_healthy_tiles_gdf

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

    def tiles_to_cut(self, save = False, output_dir = None):
        """ Plot the full slide and highlight the top-k and bottom-k tiles to cut with red and blue. """
        healthy_tiles_gdf, non_healthy_tiles_gdf = self.get_tiles_gdf()
        wsi = self.get_wsi()

        fig, ax = plt.subplots(figsize=(12, 12))
        zs.pl.tissue(
            wsi,
            ax=ax,
            show_contours=True,
        )
        healthy_tiles_gdf.plot(
            ax=ax,
            facecolor="#ff2d2d",
            edgecolor="#b30000",
            linewidth=1.5,
            alpha=0.35,
        )
        non_healthy_tiles_gdf.plot(
            ax=ax,
            facecolor="#2d2dff",
            edgecolor="#0000b3",
            linewidth=1.5,
            alpha=0.35,
        )
        ax.set_title(f"Top {len(healthy_tiles_gdf)} (red) and bottom {len(non_healthy_tiles_gdf)} (blue) tiles selected for cutting")
        ax.axis("off")
        plt.tight_layout()

        if save:
            if output_dir is None:
                output_dir = "tiles_to_cut"
                
            os.makedirs(output_dir, exist_ok=True)
            slide_name = os.path.splitext(os.path.basename(self.slide_path))[0]
            output_path = os.path.join(output_dir, f"{slide_name}_tiles_to_cut.jpg")

            fig.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path
        
        plt.show()
        return fig
    
    def get_wsi(self):
        slide_path = self.slide_data["slide_path"]
        zarr_path = self.slide_data["zarr_path"]
        self.wsi = open_wsi(slide_path, zarr_path)
        return self.wsi
    
    def get_sdata_lmd(self):
        self.sdata = self.wsi.to_spatialdata()
        healthy_gdf, non_healthy_gdf = self.get_tiles_gdf()
        self.sdata.shapes["healthy_tiles"] = ShapesModel.parse(healthy_gdf)
        self.sdata.shapes["non_healthy_tiles"] = ShapesModel.parse(non_healthy_gdf)
        export_layers = ["wsi_thumbnail", "healthy_tiles", "non_healthy_tiles"]
        sdata_lmd = self.sdata.subset(element_names=export_layers)
        return sdata_lmd

    def napari_polygons(self):
        """ Return list of polygon coordinate arrays suitable for Napari `add_shapes`."""
        healthy_tiles_gdf, non_healthy_tiles_gdf = self.get_tiles_gdf()
        polygons_healthy = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in healthy_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        polygons_non_healthy = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in non_healthy_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        return polygons_healthy, polygons_non_healthy

    def viewer_polygons(self):
        """ Return top/bottom tile outlines as closed polygons in standard image coordinate order: (x, y).

        Each tile is returned as a 5-point closed ring:
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)].
        """
        healthy_tiles_gdf, non_healthy_tiles_gdf = self.get_tiles_gdf()

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

        polygons_healthy = []
        for geom in healthy_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_healthy.append(poly)

        polygons_non_healthy = []
        for geom in non_healthy_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_non_healthy.append(poly)

        return polygons_healthy, polygons_non_healthy

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
            for tiles in ["healthy", "non_healthy"]:
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