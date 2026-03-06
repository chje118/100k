from typing import List, Literal
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wsidata import open_wsi
import lazyslide as zs

class ConsensusTileSelector:
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


class ABMILTileSelector:
    """ Select tiles from a slide based on ABMIL attention scores.
    
    Given a trained ABMIL model and attention scores for tiles on a new slide,
    this class selects the most important/informative tiles for further analysis
    or validation (e.g., laser microdissection).
    
    The selection can be:
    - Pure attention-based: top-N tiles with highest attention scores
    - Spatially-aware: top tiles ensuring minimum spatial distance between selections
    - Stratified: balanced selection across slide regions while respecting high attention scores
    """
    
    def __init__(
        self,
        wsi,
        attention_scores: np.ndarray | pd.Series,
        tile_key: str = "tiles_224",
        n_tiles: int = 10,
        min_distance_px: float = 500.0,
        selection_mode: Literal["top_k", "top_k_spatial", "stratified"] = "top_k",
        verbose: bool = True,
    ):
        """Initialize ABMIL tile selector.
        
        Parameters
        ----------
        wsi : WSI object
            Whole slide image object with tile geometries.
        attention_scores : np.ndarray or pd.Series
            Attention scores from ABMIL model, shape (n_tiles,).
            Should be normalized to [0, 1] for interpretability.
        tile_key : str, optional
            Key in wsi.shapes for tile geometries (default: "tiles_224").
        n_tiles : int, optional
            Number of tiles to select (default: 10).
        min_distance_px : float, optional
            Minimum spatial distance (pixels) between selected tiles (default: 500).
            Only used in "top_k_spatial" and "stratified" modes.
        selection_mode : str, optional
            Selection strategy:
            - "top_k": Select top-N tiles by attention score (fastest, may cluster).
            - "top_k_spatial": Top tiles with spatial distance constraint.
            - "stratified": Balance attention with spatial coverage (recommended for LMD).
        verbose : bool, optional
            Print selection summary (default: True).
        """
        self.wsi = wsi
        self.tile_key = tile_key
        self.n_tiles = n_tiles
        self.min_distance_px = float(min_distance_px)
        self.selection_mode = selection_mode
        self.verbose = verbose
        
        # Load tiles and validate attention scores
        self.tiles_gdf = wsi.shapes[tile_key]
        self.attention_scores = self._validate_and_prepare_scores(attention_scores)
        
        # Extract centroids for spatial operations
        cx, cy = self._extract_centroids()
        self.centroids = np.stack([cx, cy], axis=1)
        
        # Build metadata table
        self.meta_df = self._build_meta_df()
    
    def _validate_and_prepare_scores(self, scores) -> np.ndarray:
        """Convert and validate attention scores to numpy array."""
        if isinstance(scores, pd.Series):
            scores = scores.values
        elif not isinstance(scores, np.ndarray):
            scores = np.asarray(scores)
        
        if scores.shape[0] != len(self.tiles_gdf):
            raise ValueError(
                f"Number of attention scores ({scores.shape[0]}) does not match "
                f"number of tiles ({len(self.tiles_gdf)})."
            )
        
        # Warn if scores are outside expected range
        if scores.min() < 0 or scores.max() > 1:
            if self.verbose:
                print(f"WARNING: Attention scores outside [0,1]: min={scores.min():.4f}, max={scores.max():.4f}")
                print("Consider normalizing scores (e.g., via softmax or sigmoid) for better interpretability.")
        
        return scores
    
    def _extract_centroids(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract tile centroid coordinates."""
        centroids = self.tiles_gdf["geometry"].centroid
        return centroids.x.to_numpy(), centroids.y.to_numpy()
    
    def _build_meta_df(self) -> pd.DataFrame:
        """Build metadata table with tile indices, centroids, and attention scores."""
        tile_idx = np.arange(len(self.tiles_gdf), dtype=int)
        cx, cy = self.centroids[:, 0], self.centroids[:, 1]
        
        meta_df = pd.DataFrame({
            "tile_idx": tile_idx,
            "cx": cx,
            "cy": cy,
            "attention_score": self.attention_scores,
        })
        
        # Add tile_id if present
        if "tile_id" in self.tiles_gdf.columns:
            meta_df["tile_id"] = self.tiles_gdf["tile_id"].values
        
        return meta_df
    
    def select_top_k(self) -> pd.DataFrame:
        """Select top-N tiles by attention score (no spatial constraint).
        
        Returns
        -------
        pd.DataFrame
            Selected tiles sorted by attention score (highest first).
        """
        top_indices = np.argsort(-self.attention_scores)[:self.n_tiles]
        result = self.meta_df.iloc[top_indices].copy()
        result = result.sort_values("attention_score", ascending=False).reset_index(drop=True)
        return result
    
    def select_top_k_spatial(self) -> pd.DataFrame:
        """Select top-N tiles by attention score with spatial distance constraint.
        
        Greedy algorithm: iteratively pick highest-attention tile that is
        at least min_distance_px away from all previously selected tiles.
        
        Returns
        -------
        pd.DataFrame
            Selected tiles sorted by attention score (highest first).
        """
        selected = []
        available = np.ones(len(self.meta_df), dtype=bool)
        
        while len(selected) < self.n_tiles:
            # Find highest-attention available tile
            available_idx = np.nonzero(available)[0]
            if available_idx.size == 0:
                break
            
            scores_available = self.attention_scores[available_idx]
            best_local_idx = np.argmax(scores_available)
            best_global_idx = available_idx[best_local_idx]
            
            # Check spatial constraint
            if len(selected) > 0:
                sel_coords = self.centroids[selected]
                cand_coord = self.centroids[best_global_idx]
                
                # Compute squared distances to all selected tiles
                diff = cand_coord - sel_coords
                distances_sq = np.sum(diff**2, axis=1)
                
                if np.min(distances_sq) < (self.min_distance_px ** 2):
                    # Too close; mark as unavailable and try next
                    available[best_global_idx] = False
                    continue
            
            selected.append(best_global_idx)
            available[best_global_idx] = False
        
        result = self.meta_df.iloc[selected].copy()
        result = result.sort_values("attention_score", ascending=False).reset_index(drop=True)
        return result
    
    def select_stratified(self, n_bins: int = 5) -> pd.DataFrame:
        """Stratified selection: balance attention scores with spatial coverage.
        
        Divides the slide into spatial bins and selects high-attention tiles
        from each bin, ensuring coverage across the slide while respecting
        attention-based importance.
        
        Parameters
        ----------
        n_bins : int, optional
            Number of spatial bins per dimension (default: 5, creating 5x5=25 bins).
        
        Returns
        -------
        pd.DataFrame
            Selected tiles stratified across slide with high attention scores.
        
        TODO: Consider hexagonal or adaptive binning for more uniform coverage.
        """
        # Create spatial bins
        cx_min, cx_max = self.centroids[:, 0].min(), self.centroids[:, 0].max()
        cy_min, cy_max = self.centroids[:, 1].min(), self.centroids[:, 1].max()
        
        cx_bins = np.linspace(cx_min, cx_max, n_bins + 1)
        cy_bins = np.linspace(cy_min, cy_max, n_bins + 1)
        
        # Assign each tile to a bin
        cx_bin_idx = np.digitize(self.centroids[:, 0], cx_bins) - 1
        cy_bin_idx = np.digitize(self.centroids[:, 1], cy_bins) - 1
        bin_ids = cx_bin_idx * n_bins + cy_bin_idx
        
        # Select top tile from each bin (or per-bin quota)
        selected = []
        tiles_per_bin = max(1, self.n_tiles // (n_bins ** 2))
        
        for bin_id in np.unique(bin_ids):
            bin_mask = bin_ids == bin_id
            bin_indices = np.nonzero(bin_mask)[0]
            
            if bin_indices.size == 0:
                continue
            
            # Get top tiles in this bin by attention score
            bin_scores = self.attention_scores[bin_indices]
            top_local_indices = np.argsort(-bin_scores)[:tiles_per_bin]
            top_global_indices = bin_indices[top_local_indices]
            
            selected.extend(top_global_indices)
        
        # Trim to n_tiles if necessary
        selected = selected[:self.n_tiles]
        result = self.meta_df.iloc[selected].copy()
        result = result.sort_values("attention_score", ascending=False).reset_index(drop=True)
        return result
    
    def select_tiles(self) -> pd.DataFrame:
        """Execute tile selection based on configured selection_mode.
        
        Returns
        -------
        pd.DataFrame
            Selected tiles with columns: tile_idx, cx, cy, attention_score, [tile_id].
        """
        if self.selection_mode == "top_k":
            result = self.select_top_k()
        elif self.selection_mode == "top_k_spatial":
            result = self.select_top_k_spatial()
        elif self.selection_mode == "stratified":
            result = self.select_stratified()
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")
        
        if self.verbose:
            print(f"Selected {len(result)} tiles using '{self.selection_mode}' mode")
            print(f"Attention score range: [{result['attention_score'].min():.4f}, {result['attention_score'].max():.4f}]")
        
        return result
    
    def plot_attention_heatmap(self, selected: pd.DataFrame | None = None, figsize=(8, 8)):
        """Visualize attention scores as a heatmap overlay, with selected tiles highlighted.
        
        Parameters
        ----------
        selected : pd.DataFrame, optional
            DataFrame of selected tiles (output from select_tiles()).
        figsize : tuple, optional
            Figure size (default: (8, 8)).
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes object with plot.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color tiles by attention score
        scatter = ax.scatter(
            self.meta_df["cx"], self.meta_df["cy"],
            c=self.meta_df["attention_score"],
            s=10, cmap="YlOrRd", alpha=0.6, label="all tiles"
        )
        plt.colorbar(scatter, ax=ax, label="Attention Score")
        
        # Overlay selected tiles with markers
        if selected is not None and not selected.empty:
            ax.scatter(
                selected["cx"], selected["cy"],
                c="blue", s=50, marker="*", edgecolors="black", linewidths=1.5,
                label="selected", zorder=5
            )
        
        ax.set_xlabel("X coordinate (px)")
        ax.set_ylabel("Y coordinate (px)")
        ax.set_title(f"ABMIL Attention Heatmap (mode='{self.selection_mode}')")
        ax.legend()
        ax.set_aspect("equal")
        
        return ax



def zoom_to_selected(wsi, selected_tiles):
    """ Visualize selected tiles with zoomed-in view. """
    raise NotImplementedError("Zooming to selected tiles is not implemented yet.")
    # TODO: domain_consensus column added to the tile table 
    # TODO: zoom to selected tiles 
    # viewer = zs.pl.WSIViewer(wsi)
    # viewer.show()


if __name__ == "__main__":
    import open_wsi

    slide_path = "path/to/slide.mrxs"
    zarr_path = "path/to/slide.zarr"
    
    try:
        wsi = open_wsi(slide_path, zarr_path)
    except FileNotFoundError:
        print(f"Test slide not found at {slide_path}. Skipping examples.")
        exit(0)
    
    # =====================================================================
    # Example 1: Domain-based selection (existing functionality)
    # =====================================================================
    print("\n=== Domain-based Tile Selection ===\n")
    try:
        selector = TileSelector(
            wsi=wsi,
            feature_key="features",
            domain_keys="domain_conch",
            n_per_domain=5,
            min_distance_px=500.0
        )
        selected_tiles = selector.select_tiles_per_domain()
        print(f"Selected {len(selected_tiles)} tiles by domain")
        print(selected_tiles.head())
        selector.plot_domains(selected_tiles)
    except (KeyError, ValueError) as e:
        print(f"Domain-based selection error: {e}")
    
    # =====================================================================
    # Example 2: ABMIL-based selection (new functionality)
    # =====================================================================
    print("\n=== ABMIL Attention-based Tile Selection ===\n")
    
    # TODO: In practice, attention_scores would come from running ABMIL inference
    # For now, simulate with random scores (sorted so we can see top-k selection)
    n_tiles = len(wsi.shapes["tiles_224"])
    simulated_attention = np.random.rand(n_tiles)
    
    print(f"Simulated ABMIL attention scores for {n_tiles} tiles\n")
    
    # Example 2a: Pure top-K selection (fastest, may cluster)
    print("--- Mode 1: top_k (pure attention, no spatial constraint) ---")
    selector_topk = ABMILTileSelector(
        wsi=wsi,
        attention_scores=simulated_attention,
        tile_key="tiles_224",
        n_tiles=10,
        selection_mode="top_k",
        verbose=True
    )
    selected_topk = selector_topk.select_tiles()
    print(f"\nSelected tiles:\n{selected_topk}\n")
    selector_topk.plot_attention_heatmap(selected_topk)
    
    # Example 2b: Top-K with spatial constraint (avoids clustering)
    print("\n--- Mode 2: top_k_spatial (attention + min distance) ---")
    selector_spatial = ABMILTileSelector(
        wsi=wsi,
        attention_scores=simulated_attention,
        tile_key="tiles_224",
        n_tiles=10,
        min_distance_px=500.0,
        selection_mode="top_k_spatial",
        verbose=True
    )
    selected_spatial = selector_spatial.select_tiles()
    print(f"\nSelected tiles:\n{selected_spatial}\n")
    selector_spatial.plot_attention_heatmap(selected_spatial)
    
    # Example 2c: Stratified selection (balanced spatial + attention)
    print("\n--- Mode 3: stratified (attention + spatial coverage) ---")
    selector_strat = ABMILTileSelector(
        wsi=wsi,
        attention_scores=simulated_attention,
        tile_key="tiles_224",
        n_tiles=10,
        min_distance_px=500.0,
        selection_mode="stratified",
        verbose=True
    )
    selected_strat = selector_strat.select_tiles()
    print(f"\nSelected tiles:\n{selected_strat}\n")
    selector_strat.plot_attention_heatmap(selected_strat)
    
    # =====================================================================
    # Example 3: Workflow for LMD (Laser Microdissection)
    # =====================================================================
    print("\n=== Practical LMD Workflow ===\n")
    print("TODO: After ABMIL inference on slide, get attention_scores from model")
    print("TODO: Use ABMILTileSelector with selection_mode='top_k_spatial' or 'stratified'")
    print("TODO: Export selected tiles (coordinates) to LMD platform for cutting")
    print("TODO: Consider combining with domain-based selection for validation")

    