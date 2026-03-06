from typing import List, Literal
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wsidata import open_wsi
import lazyslide as zs

class TileSelector:
    def __init__(
        self,
        wsi,
        feature_key: str,
        domain_keys: List[str]|str, 
        tile_key: str = "tiles_224",
        n_per_domain: int = 10,
        min_distance_px: float = 500.0,
        score_mode: Literal["maxmin", "sum"] = "maxmin",
        on_fail: Literal["stop", "relax"] = "relax",
        agreement_mode: Literal["all_same", "all_different", "at_least", "exactly", "at_most"] = "all_same",
        min_agreement: int | None = None,
        ):
        self.wsi = wsi
        self.feature_key = feature_key
        self.domain_keys = [domain_keys] if isinstance(domain_keys, str) else domain_keys
        self.tile_key = tile_key
        self.n_per_domain = n_per_domain
        self.min_distance_px = float(min_distance_px)
        self.score_mode = score_mode
        self.on_fail = on_fail
        self.agreement_mode = agreement_mode
        self.min_agreement = min_agreement
        self.load_tile_table()

    def get_features_and_tiles(self):
        """ Fetch tile geometries and features. """
        tiles_gdf = self.wsi.shapes[self.tile_key]
        adata = self.wsi.tables[self.feature_key]

        X = adata.X
        if hasattr(X, "toarray"):
            features = X.toarray()
        else:
            features = np.asarray(X)

        if len(tiles_gdf) != features.shape[0]:
            raise ValueError(
                f"Number of tiles ({len(tiles_gdf)}) and feature rows ({features.shape[0]}) do not match."
            )

        return tiles_gdf, features

    @staticmethod
    def compute_domain_consensus(domain_values, mode, min_agreement) -> object:
        """ Compute consensus domain label from multiple annotations.
        
        Parameters
        - domain_values (list): Values from all domain columns for a single tile.
        - mode (str): Agreement mode:
            - 'all_same': all values must be identical
            - 'all_different': all values must be unique
            - 'at_least': at least min_agreement values must match
            - 'exactly': exactly min_agreement values must match
            - 'at_most': at most min_agreement values must match
        - min_agreement (int or None): Threshold for 'at_least', 'exactly', 'at_most' modes.
        
        Returns:
        - Consensus label or None if agreement condition not met.
        """

        counter = Counter(domain_values)
        most_common, count = counter.most_common(1)[0]
        n_values = len(domain_values)
        
        if mode == "all_same":
            return most_common if count == n_values else None
        elif mode == "all_different":
            return "all_different" if count == 1 else None
        elif mode == "at_least":
            return most_common if count >= min_agreement else None
        elif mode == "exactly":
            return most_common if count == min_agreement else None
        elif mode == "at_most":
            return most_common if count <= min_agreement else None
        return None

    def extract_domain_columns(self, tiles_gdf: pd.DataFrame) -> List[np.ndarray]:
        """Extract domain column arrays from GeoDataFrame."""
        domains_list = []
        for dom_key in self.domain_keys:
            if dom_key not in tiles_gdf.columns:
                raise KeyError(f"Domain key '{dom_key}' not found.")
            domains_list.append(tiles_gdf[dom_key].to_numpy())
        return domains_list

    def collapse_domains(self, domains_list: List[np.ndarray]) -> np.ndarray:
        """ Collapse multiple domain annotations into consensus labels."""
        if len(domains_list) == 1:
            return domains_list[0]

        consensus = []
        for i in range(len(domains_list[0])):
            domain_values = [d[i] for d in domains_list]
            label = self.compute_domain_consensus(
                domain_values, 
                mode=self.agreement_mode,
                min_agreement=self.min_agreement
            )
            consensus.append(label)
        return np.array(consensus, dtype=object)

    @staticmethod
    def extract_centroids(tiles_gdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """ Extract (x, y) coordinates from geometry column. """
        centroids = tiles_gdf["geometry"].centroid
        return centroids.x.to_numpy(), centroids.y.to_numpy()

    @staticmethod
    def extract_tile_ids(tiles_gdf: pd.DataFrame) -> np.ndarray | None:
        """Extract tile_id column if present."""
        return tiles_gdf["tile_id"].to_numpy() if "tile_id" in tiles_gdf.columns else None

    def build_meta_df(self, tiles_gdf, domains, cx, cy, tile_id=None) -> pd.DataFrame:
        """Build metadata table with domains, centroids, and tile indices."""
        tile_idx = np.arange(len(tiles_gdf), dtype=int)
        return pd.DataFrame(
            {
                **({"tile_id": tile_id} if tile_id is not None else {}),
                "tile_idx": tile_idx,
                "domain": domains,
                "cx": cx,
                "cy": cy,
            }
        )

    def load_tile_table(self):
        """Build a per-tile table with domain labels, centroids, and features."""
        tiles_gdf, features = self.get_features_and_tiles()

        domains_list = self.extract_domain_columns(tiles_gdf)
        domains = self.collapse_domains(domains_list)
        cx, cy = self.extract_centroids(tiles_gdf)
        tile_id = self.extract_tile_ids(tiles_gdf)
        meta_df = self.build_meta_df(tiles_gdf, domains, cx, cy, tile_id)

        # Store results in instance variables
        self.meta_df = meta_df
        self.features = features
        self.centroids = np.stack([cx, cy], axis=1)
        # If multiple domain keys, use "domain" as the column name for consensus; otherwise use the single domain key
        self.domain_col = "domain" if len(self.domain_keys) > 1 else self.domain_keys[0]

        return meta_df, features

    @staticmethod
    def standardize_features(features: np.ndarray) -> np.ndarray:
        """
        Zero-mean, unit-variance scaling per feature, with numerical safety.
        """
        if features.size == 0:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std_safe = np.where(std == 0, 1.0, std)
        return (features - mean) / std_safe

    @staticmethod
    def squared_euclidean_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """ Compute pairwise squared Euclidean distances. 

        Parameters
        - coords1: (n1, d) array of coordinates
        - coords2: (n2, d) array of coordinates

        Returns
        - (n1, n2) array of squared distances

        d: dimensionality of coordinates (e.g., 2 for (x, y))
        """
        diff = coords1[:, None, :] - coords2[None, :, :]
        return np.sum(diff**2, axis=2)

    @staticmethod
    def satisfies_distance_constraint(candidate_coords, selected_coords, min_distance):
        """ Check if a candidate location satisfies minimum distance constraint.
        
        Parameters
        - candidate_coords (2,): Coordinates of the candidate tile.
        - selected_coords (n, 2): Coordinates of already-selected tiles.
        - min_distance (float): Minimum required distance.
        
        Returns
        - bool: True if the candidate is at least `min_distance` away from all selected tiles, False otherwise.
        """
        if min_distance <= 0 or selected_coords.shape[0] == 0:
            return True
        distances_sq = TileSelector.squared_euclidean_distances(
            candidate_coords.reshape(1, -1),
            selected_coords,
        )
        return np.min(distances_sq) >= (min_distance**2)

    def greedy_diverse_subset(self, features: np.ndarray, centroid: np.ndarray, n: int) -> List[int]:
        """ Greedy diversity sampling with spatial distance constraint.
        
        For each selection step, candidates are ranked by diversity score and selected in order
        of decreasing diversity, with the first candidate that meets the spatial distance
        constraint being chosen.

        Parameters
        - features (m, d): Feature matrix.
        - centroid (m, 2): Centroid coordinates.
        - n (int): Number of tiles to select.

        Returns
        - List[int]: Indices of selected tiles.
        """
        # Handle edge cases: no tiles, n <= 0, or only one tile
        m = features.shape[0]
        if m == 0 or n <= 0:
            return []
        if m == 1:
            return [0]

        # Ensure n does not exceed available tiles
        n = min(n, m)
        features_std = self.standardize_features(features)

        # Start with the most "central" tile
        mean_vec = features_std.mean(axis=0, keepdims=True)
        d2_center = np.sum((features_std - mean_vec) ** 2, axis=1)
        first_idx = int(np.argmax(d2_center))

        # Initialize selection and availability tracking
        selected = [first_idx]
        available = np.ones(m, dtype=bool)
        available[first_idx] = False

        # Initialize current minimum distance for selection, which can be relaxed if needed
        current_min_distance = self.min_distance_px
        base_min_distance = self.min_distance_px

        while len(selected) < n:
            available_idx = np.nonzero(available)[0]
            if available_idx.size == 0:
                break

            # Compute diversity scores using feature distances
            features_sel = features_std[selected]
            features_cand = features_std[available_idx]
            d2_features = self.squared_euclidean_distances(features_cand, features_sel)

            # sum: sum of squared distances to selected set
            if self.score_mode == "sum":
                diversity_scores = d2_features.sum(axis=1)
            # maxmin: minimum squared distance to selected set
            else:
                diversity_scores = d2_features.min(axis=1)

            # Try candidates in order of decreasing diversity
            sorted_order = np.argsort(-diversity_scores)
            sorted_candidates = available_idx[sorted_order]

            sel_coords = centroid[selected]
            selected_idx = None

            # Check candidates against distance constraint in order of diversity
            for candidate_idx in sorted_candidates:
                cand_coord = centroid[candidate_idx]
                
                if self.satisfies_distance_constraint(
                    cand_coord, sel_coords, current_min_distance
                ):
                    selected_idx = candidate_idx
                    break

            if selected_idx is None:
                if (self.on_fail == "relax" and current_min_distance > base_min_distance * 0.3):
                    current_min_distance *= 0.9
                    continue
                else: # on_fail == "stop" or minimum distance already relaxed significantly
                    break

            selected.append(selected_idx)
            available[selected_idx] = False

        return selected

    def select_tiles_for_domain(self, df_dom):
        """ Select tiles within a single domain."""
        if df_dom.empty:
            return None

        features_dom = self.features[df_dom["tile_idx"].to_numpy()]
        centroid_dom = df_dom[["cx", "cy"]].to_numpy()

        if len(df_dom) <= 1:
            chosen = list(range(len(df_dom)))
        else:
            chosen = self.greedy_diverse_subset(features_dom, centroid_dom, n=self.n_per_domain)

        if not chosen:
            return None

        df_sel = df_dom.iloc[chosen].copy()
        return df_sel

    def iterate_domains(self, tile_table):
        """ Get sorted unique domain values (excluding None). """
        return sorted(tile_table[self.domain_col].dropna().unique())

    def select_tiles_per_domain(self):
        """ Run selection independently within each domain. """
        tile_table = self.meta_df
        selected_rows = []

        for domain_value in self.iterate_domains(tile_table):
            df_dom = (
                tile_table[tile_table[self.domain_col] == domain_value]
                .reset_index(drop=True)
                .copy()
            )
            result = self.select_tiles_for_domain(df_dom)
            if result is not None:
                selected_rows.append(result)

        if not selected_rows:
            return tile_table.iloc[0:0].copy()

        return pd.concat(selected_rows, axis=0, ignore_index=True)

    def check_min_distance(self, df, min_distance_px):
        """ Verify all selected tiles within each domain meet the distance constraint. """
        if df.empty or min_distance_px <= 0:
            return True

        min_d2 = float(min_distance_px**2)
        domain_col = self.domain_col

        for _, group in df.groupby(domain_col):
            coords = group[["cx", "cy"]].to_numpy()
            if len(coords) <= 1:
                continue

            d2 = self.squared_euclidean_distances(coords, coords)
            mask = np.triu(np.ones_like(d2, dtype=bool), k=1)
            if mask.any() and np.any(d2[mask] < min_d2):
                return False

        return True

    def plot_domains(self, selected: pd.DataFrame | None = None):
        """ Visualize domains and overlay selected tiles. """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        df = self.meta_df
        if df.empty:
            return ax

        # Assign a color to each domain value
        # convert None/NaN to a consistent string for coloring
        domains = df[self.domain_col].where(pd.notna(df[self.domain_col]), other="nan")
        domains = domains.astype(str)
        unique = domains.unique()
        cmap = plt.get_cmap("tab10")
        color_map = {val: cmap(i % cmap.N) for i, val in enumerate(unique) if val != "nan"}
        # fallback for NaN/None (light, almost white)
        color_map["nan"] = (0.9, 0.9, 0.9, 0.3)

        colors = domains.map(lambda v: color_map.get(v, color_map["nan"]))

        # smaller markers to avoid pixel-scale overcrowding
        ax.scatter(df["cx"], df["cy"], c=list(colors), s=2, alpha=0.4, label="all tiles")
        handles = []
        for val, col in color_map.items():
            handles.append(plt.Line2D([], [], marker="o", color=col, linestyle="", label=val))

        if selected is not None and not selected.empty:
            # black crosses for selected points
            ax.scatter(selected["cx"], selected["cy"],
                       c="black", s=15, marker="x", label="selected")
        
        ax.legend(handles=handles + ([plt.Line2D([], [], marker="x", color="black", linestyle="", label="selected")] if selected is not None and not selected.empty else []))
        ax.set_xlabel("cx")
        ax.set_ylabel("cy")
        ax.set_title("Tile domains and selections")
        return ax
    
    def zoom_to_selected(self, selected: pd.DataFrame):
        raise NotImplementedError("Zooming to selected tiles is not implemented yet.")
        """ Visualize selected tiles with zoomed-in view. """
        # TODO: domain_consensus column added to the tile table 
        # TODO: zoom to selected tiles 
        # viewer = zs.pl.WSIViewer(wsi)
        # viewer.show()
 

if __name__ == "__main__":
    import open_wsi

    slide_path = "path/to/slide.mrxs"
    zarr_path = "path/to/slide.zarr"
    wsi = open_wsi(slide_path, zarr_path)
    
    # Single Domain
    selector = TileSelector(
        wsi=wsi,
        feature_key="features",
        domain_keys="domain_conch",
        n_per_domain=5,
        min_distance_px=500.0
    )
    selected_tiles = selector.select_tiles_per_domain()
    print(f"Selected {len(selected_tiles)} tiles")
    print(selected_tiles.head())
    selector.plot_domains(selected_tiles)
    
    # Multiple Domains
    selector_multi = TileSelector(
        wsi=wsi,
        feature_key="features", 
        domain_keys=["domain_conch", "domain_uni", "domain_hopt"],
        n_per_domain=3,
        agreement_mode="all_same",
        min_distance_px=300.0
    )
    selected_tiles_multi = selector_multi.select_tiles_per_domain()
    print(f"Selected {len(selected_tiles_multi)} tiles with consensus")
    print(selected_tiles_multi.head())
    selector_multi.plot_domains(selected_tiles_multi)
    