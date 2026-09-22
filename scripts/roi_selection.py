import json
import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import warnings
from wsidata import open_wsi
import geopandas as gpd
from spatialdata.models import PointsModel, ShapesModel
from dvpio.write import write_lmd
import os

class ROISelector:
    """ Handle ROI selection from cached ABMIL inference results.

    - Disease arm: ranked by `disease_score_col` ("contribution_score"
      = attention * contrast_score). This is the EXACT per-tile
      contribution to the model's class logit.
      It answers "what tissue actually drove the non-healthy call" - the
      tiles you want as diagnostic evidence for DVP.

    - Healthy arm: ranked by `healthy_score_col` ("contrast_score",
      NOT attention-weighted). This answers "does this tile's own tissue
      read as healthy", independent of whether the model happened to need
      it for this particular slide's verdict. 
      
      Note this independence is only within the attention-floored pool:
      `min_attention_pct` drops the bottom fraction of tiles by attention
      from BOTH arms first (to exclude background/blur/ink).

    - 'min_effect_size' is a threshold on the contribution_score or 
      contrast_score to exclude tiles that are too close to neutral (0.0).
    """
    def __init__(self, cache_path: str, slide_path: str, disease_k: int = 20, healthy_k: int = 10, sample_pct: float = 0.20, random_state: int | None = 42, disease_score_col: str = "contribution_score", healthy_score_col: str = "contrast_score", attention_floor: float = 0.05, min_effect_size: float = 0.0, strict_topk: bool = False, min_spatial_distance: float | None = None):
        self.cache_path = cache_path
        self.slide_path = slide_path
        self.disease_k = disease_k
        self.healthy_k = healthy_k
        self.sample_pct = sample_pct
        if random_state is None:
            warnings.warn(
                f"ROISelector for {slide_path!r} created with random_state=None: "
                "tile selection within each candidate pool will NOT be reproducible "
                "across runs. Pass a fixed int unless non-determinism is intentional.",
                stacklevel=2,
            )
        self.random_state = random_state
        self.healthy_score_col = healthy_score_col
        self.disease_score_col = disease_score_col
        self.attention_floor = attention_floor
        self.min_effect_size = min_effect_size
        self.strict_topk = strict_topk
        self.min_spatial_distance = min_spatial_distance
        self.slide_cache = self.load_cache(cache_path)
        self.slide_data = self._slide_data()

    @staticmethod
    def load_cache(input_path: str):
        with open(input_path, "rb") as f:
            cached = pickle.load(f)
        return cached.get("slide_cache", cached) if isinstance(cached, dict) else {}

    @staticmethod
    def _pool_size(n_tiles: int, pct: float) -> int:
        return max(1, int(np.ceil(n_tiles * pct)))

    def _slide_data(self):
        slide_data = self.slide_cache.get(self.slide_path)
        if slide_data is None:
            raise KeyError(f"No cached data found for slide: {self.slide_path}")
        return slide_data

    @staticmethod
    def _default_min_spatial_distance(pool):
        """
        Default minimum center-to-center spacing between randomly sampled
        tiles, if min_spatial_distance isn't set explicitly:
        1.5x one tile's own width, so two selected tiles can't default to
        being immediately adjacent. Falls back to 0.0 (no constraint) if
        geometry is missing or degenerate.
        """
        if len(pool) == 0:
            return 0.0
        minx, miny, maxx, maxy = pool["geometry"].iloc[0].bounds
        tile_width = maxx - minx
        if not np.isfinite(tile_width) or tile_width <= 0:
            return 0.0
        return 1.5 * tile_width

    def select_tiles(self):
        """
        Return  healthy_k and disease_k tiles sampled at random from top 
        sample_pct% by `self.disease_score_col` (disease arm) and bottom
        sample_pct% by `self.healthy_score_col` (healthy arm).

        Restricted to tiles that cleared the attention relevance floor
        and the effect-size floor.

        Sorted by shortest path for LMD (greedy nearest-neighbor from 
        top-left corner).
        """
        tile_table = self.slide_data.get("tile_table")
        if tile_table is None:
            raise KeyError("slide_data must contain 'tile_table'")

        required_cols = ["attention", self.disease_score_col, self.healthy_score_col, "geometry"]
        missing = [c for c in dict.fromkeys(required_cols) if c not in tile_table.columns]
        if missing:
            raise KeyError(
                f"slide_data['tile_table'] is missing column(s) {missing}. Run ABMILInference."
            )

        tile_table = tile_table.dropna(subset=list(dict.fromkeys(required_cols))).copy()
        n_tiles = len(tile_table)

        if n_tiles == 0:
            empty = gpd.GeoDataFrame(tile_table, geometry="geometry")
            return empty, empty

        # Filter out tiles below attention floor
        attention_threshold = tile_table["attention"].quantile(self.attention_floor)
        tile_table = tile_table[tile_table["attention"] >= attention_threshold].copy()
        n_tiles = len(tile_table)

        if n_tiles == 0:
            empty = gpd.GeoDataFrame(tile_table, geometry="geometry")
            return empty, empty

        # Determine pool size based on sample_pct
        pool_n = self._pool_size(n_tiles, self.sample_pct)

        # Disease arm = most positive contribution_score, above the effect-size floor
        ranked_desc = tile_table.sort_values(self.disease_score_col, ascending=False)
        ranked_desc = ranked_desc[ranked_desc[self.disease_score_col] > self.min_effect_size].copy()

        # Control arm = most negative contrast_score, below -effect-size floor
        ranked_asc = tile_table.sort_values(self.healthy_score_col, ascending=True)
        ranked_asc = ranked_asc[ranked_asc[self.healthy_score_col] < -self.min_effect_size].copy()

        # Select pool_n tiles from each arm
        disease_pool = ranked_desc.head(pool_n).copy()
        healthy_pool = ranked_asc.head(pool_n).copy()

        # Select k tiles from each pool
        disease_tiles = self._select_from_pool(disease_pool, self.disease_k, arm="disease")
        healthy_tiles = self._select_from_pool(healthy_pool, self.healthy_k, arm="healthy")

        # Sort each by shortest path (greedy nearest-neighbor)
        disease_tiles = self._sort_tiles_tsp(disease_tiles)
        healthy_tiles = self._sort_tiles_tsp(healthy_tiles)

        return gpd.GeoDataFrame(disease_tiles, geometry="geometry"), gpd.GeoDataFrame(healthy_tiles, geometry="geometry")

    def _select_from_pool(self, pool, k, arm):
        """
        Draw up to k tiles from a candidate pool.
        Warning on any shortfall.
        """
        if len(pool) == 0:
            warnings.warn(
                f"[{self.slide_path}] {arm} arm: candidate pool is EMPTY after score/effect-size/"
                f"attention-floor filtering. 0/{k} tiles selected for this arm on this slide.",
                stacklevel=3,
            )
            return pool.copy()

        if len(pool) < k:
            warnings.warn(
                f"[{self.slide_path}] {arm} arm: candidate pool has only {len(pool)} tile(s), "
                f"fewer than the requested k={k}. Selecting all {len(pool)} available.",
                stacklevel=3,
            )

        sample_n = min(k, len(pool))

        if self.strict_topk:
            print(f"[{self.slide_path}] {arm} arm: strict_topk=True, selecting top-{sample_n} tiles without spatial diversity constraint.")
            return pool.head(sample_n).copy()

        min_distance = self.min_spatial_distance
        if min_distance is None:
            min_distance = self._default_min_spatial_distance(pool)
        print(f"[{self.slide_path}] {arm} arm: selecting up to {sample_n} tiles with min_spatial_distance={min_distance:.2f}.")

        return self._sample_with_min_distance(pool, sample_n, min_distance)

    def _sample_with_min_distance(self, pool, k, min_distance):
        """
        Randomly draw up to k tiles from `pool` such that every pair of drawn
        tiles has centroid distance >= min_distance, so a "diverse" random draw
        can't still land entirely within one contiguous patch of tissue.
        Greedy: shuffle the pool, then accept a tile only if it clears
        min_distance from every tile already accepted. If fewer than k tiles
        can be accepted under the constraint, returns as many as fit.
        """
        pool = pool.copy()
        if not isinstance(pool, gpd.GeoDataFrame):
            pool = gpd.GeoDataFrame(pool, geometry="geometry")

        centroids = pool.geometry.centroid
        coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])

        rng = np.random.RandomState(self.random_state)
        shuffled_order = rng.permutation(len(pool))

        accepted_idx = []
        accepted_coords = []
        for i in shuffled_order:
            if len(accepted_idx) >= k:
                break
            xy = coords[i]
            if accepted_coords:
                dists = np.linalg.norm(np.array(accepted_coords) - xy, axis=1)
                if dists.min() < min_distance:
                    continue
            accepted_idx.append(i)
            accepted_coords.append(xy)

        if len(accepted_idx) < k:
            warnings.warn(
                f"[{self.slide_path}] Spatial-diversity sampling with min_spatial_distance="
                f"{min_distance} could only place {len(accepted_idx)}/{k} tiles without "
                "violating the minimum-distance constraint from this pool. Consider a "
                "smaller min_spatial_distance or a larger sample_pct.",
                stacklevel=3,
            )

        return pool.iloc[accepted_idx].copy()
    
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

    def zoomed_view(self, margin: int = 0, max_tiles: int = 4, disease: bool = True):
        """ Plot zoomed tiles (grid) for review from cached slide data. """
        disease_tiles_gdf, healthy_tiles_gdf = self.select_tiles()
        tiles_gdf = disease_tiles_gdf if disease else healthy_tiles_gdf

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
        """Plot the full slide and highlight the disease-k (red) and healthy-k (blue) tiles to cut."""
        disease_tiles_gdf, healthy_tiles_gdf = self.select_tiles()
        wsi = self.get_wsi()

        fig, ax = plt.subplots(figsize=(12, 12))
        zs.pl.tissue(
            wsi,
            ax=ax,
            show_contours=True,
        )
        disease_tiles_gdf.plot(
            ax=ax,
            facecolor="#ff2d2d",
            edgecolor="#b30000",
            linewidth=1.5,
            alpha=0.35,
        )
        healthy_tiles_gdf.plot(
            ax=ax,
            facecolor="#2d2dff",
            edgecolor="#0000b3",
            linewidth=1.5,
            alpha=0.35,
        )
        ax.set_title(f"Top {len(disease_tiles_gdf)} (red) and bottom {len(healthy_tiles_gdf)} (blue) tiles selected for cutting")
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
        disease_gdf, healthy_gdf = self.select_tiles()
        self.sdata.shapes["disease_tiles"] = ShapesModel.parse(disease_gdf)
        self.sdata.shapes["healthy_tiles"] = ShapesModel.parse(healthy_gdf)
        export_layers = ["wsi_thumbnail", "disease_tiles", "healthy_tiles"]
        sdata_lmd = self.sdata.subset(element_names=export_layers)
        return sdata_lmd

    def napari_polygons(self):
        """Return list of polygon coordinate arrays suitable for Napari `add_shapes`."""
        disease_tiles_gdf, healthy_tiles_gdf = self.select_tiles()
        polygons_disease = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in disease_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        polygons_healthy = [
            np.array(g.exterior.coords)[:, [1, 0]]
            for g in healthy_tiles_gdf.geometry
            if g.geom_type == "Polygon"
        ]
        return polygons_disease, polygons_healthy

    def viewer_polygons(self):
        """
        Return disease and healthy tile outlines as closed polygons in standard 
        image coordinate order: (x, y).
        Each tile is returned as a 5-point closed ring:
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)].
        """
        disease_tiles_gdf, healthy_tiles_gdf = self.select_tiles()

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

        polygons_disease = []
        for geom in disease_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_disease.append(poly)
    
        polygons_healthy = []
        for geom in healthy_tiles_gdf.geometry:
            poly = geom_to_polygon_xy(geom)
            if poly is not None:
                polygons_healthy.append(poly)

        return polygons_disease, polygons_healthy

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
            for tiles in ["disease", "healthy"]:
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