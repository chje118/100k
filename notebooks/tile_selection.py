from typing import List, Literal
import numpy as np
import pandas as pd
from collections import Counter

class TileSelector:
    def __init__(
        self,
        wsi,
        feature_key: str,
        domain_keys,  # str or list of str
        tile_key: str = "tiles_224",
        n_per_domain: int = 10,
        min_distance_px: float = 500.0,
        score_mode: Literal["maxmin", "sum"] = "maxmin",
        on_fail: Literal["stop", "relax"] = "relax",
        agreement_mode: Literal["at_least", "exactly", "all_different"] = "at_least",
        min_agreement: int = None,
    ) -> None:
        self.wsi = wsi
        self.feature_key = feature_key
        self.domain_keys = [domain_keys] if isinstance(domain_keys, str) else domain_keys
        self.tile_key = tile_key
        self.n_per_domain = n_per_domain
        self.min_distance_px = float(min_distance_px)
        self.score_mode = score_mode
        self.on_fail = on_fail
        self.agreement_mode = agreement_mode
        self.min_agreement = min_agreement if min_agreement is not None else len(self.domain_keys)
        self.domain_col = self.domain_keys[0] if len(self.domain_keys) == 1 else "domain"
        self.load_tile_table()

    def get_features_and_tiles(self):
        """
        Fetch tile geometries and features.
        """
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

    def load_tile_table(self):
        """
        Build a per-tile table with domain labels, centroids and features.
        """
        tiles_gdf, features = self.get_features_and_tiles()

        domains_list = []
        for dk in self.domain_keys:
            if dk in tiles_gdf.columns:
                domains_list.append(tiles_gdf[dk].to_numpy())
            else:
                raise KeyError(
                    f"Domain key '{dk}' not found in tile GeoDataFrame."
                )

        if len(self.domain_keys) == 1:
            domains = domains_list[0]
        else:
            domains = []
            for i in range(len(tiles_gdf)):
                domain_values = [d[i] for d in domains_list]
                counter = Counter(domain_values)
                max_count = counter.most_common(1)[0][1]
                if self.agreement_mode == "at_least":
                    if max_count >= self.min_agreement:
                        consensus = counter.most_common(1)[0][0]
                    else:
                        consensus = None
                elif self.agreement_mode == "exactly":
                    if max_count == self.min_agreement:
                        consensus = counter.most_common(1)[0][0]
                    else:
                        consensus = None
                elif self.agreement_mode == "all_different":
                    if max_count == 1:
                        consensus = "all_different"
                    else:
                        consensus = None
                domains.append(consensus)
            domains = np.array(domains, dtype=object)

        centroids = tiles_gdf["geometry"].centroid
        cx = centroids.x.to_numpy()
        cy = centroids.y.to_numpy()

        tile_idx = np.arange(len(tiles_gdf), dtype=int)

        tile_id = None
        if "tile_id" in tiles_gdf.columns:
            tile_id = tiles_gdf["tile_id"].to_numpy()

        meta_df = pd.DataFrame(
            {
                **({"tile_id": tile_id} if tile_id is not None else {}),
                "tile_idx": tile_idx,
                "domain": domains,
                "cx": cx,
                "cy": cy,
            }
        )

        self.meta_df = meta_df
        self.features = features
        self.centroids = np.stack([cx, cy], axis=1)

        return meta_df, features

    @staticmethod
    def standardize_features(features: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance scaling per feature, with numerical safety."""
        if features.size == 0:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std_safe = np.where(std == 0, 1.0, std)
        return (features - mean) / std_safe

    def greedy_diverse_subset(self, features: np.ndarray, centroid: np.ndarray, n: int) -> List[int]:
        """
        Greedy diversity sampling with minimum spatial distance constraint enforced during selection.
        
        For each selection step, candidates are ranked by diversity score and selected in order
        of decreasing diversity, with the first candidate that meets the spatial distance
        constraint being chosen.

        Parameters:
        - features: Feature matrix (m x d)
        - centroids: Centroid coordinates (m x 2)
        - n: Number of tiles to select

        Returns:
        - List of indices of selected tiles
        """
        m = features.shape[0]
        if m == 0 or n <= 0:
            return []

        if m == 1:
            return [0]

        n = min(n, m)
        features_std = self.standardize_features(features)

        mean_vec = features_std.mean(axis=0, keepdims=True)
        d2_center = np.sum((features_std - mean_vec) ** 2, axis=1)
        first_idx = int(np.argmax(d2_center))

        selected = [first_idx]
        available = np.ones(m, dtype=bool)
        available[first_idx] = False

        current_min_distance = self.min_distance_px
        base_min_distance = self.min_distance_px

        while len(selected) < n:
            available_idx = np.nonzero(available)[0]
            if available_idx.size == 0:
                break

            # Compute diversity scores for all available candidates
            features_sel = features_std[selected]
            features_cand = features_std[available_idx]
            diff_f = features_cand[:, None, :] - features_sel[None, :, :]
            d2_features = np.sum(diff_f**2, axis=2)

            if self.score_mode == "sum":
                diversity_scores = d2_features.sum(axis=1)
            else:
                diversity_scores = d2_features.min(axis=1)

            # Sort candidates by diversity score (highest first)
            sorted_order = np.argsort(-diversity_scores)
            sorted_candidates = available_idx[sorted_order]

            # Try candidates in order of decreasing diversity
            selected_idx = None
            sel_coords = centroid[selected]

            for candidate_idx in sorted_candidates:
                cand_coord = centroid[candidate_idx:candidate_idx+1]
                
                # Check spatial distance constraint
                if current_min_distance > 0:
                    diff = cand_coord - sel_coords
                    d2 = np.sum(diff**2, axis=1)
                    min_d2 = d2.min()
                    
                    if min_d2 >= (current_min_distance**2):
                        selected_idx = candidate_idx
                        break
                else:
                    selected_idx = candidate_idx
                    break

            if selected_idx is None:
                # No candidate met the distance constraint
                if (
                    self.on_fail == "relax"
                    and current_min_distance > base_min_distance * 0.3
                ):
                    current_min_distance *= 0.9
                    continue
                else:
                    break

            selected.append(selected_idx)
            available[selected_idx] = False

        return selected

    def select_tiles_per_domain(self) -> pd.DataFrame:
        """
        Run greedy diversity sampling independently within each domain.
        
        Returns:
        - DataFrame with selected tiles per domain, ranked by domain_tile_rank
        """
        tile_table = self.meta_df
        features = self.features
        n = self.n_per_domain

        if self.domain_col not in tile_table.columns:
            raise KeyError(f"Expected column '{self.domain_col}' in tile_table.")

        if features.shape[0] != len(tile_table):
            raise ValueError(
                "features and tile_table must have the same number of rows."
            )

        selected_rows = []

        for dom in sorted(tile_table[self.domain_col].dropna().unique()):
            df_dom = tile_table[tile_table[self.domain_col] == dom].reset_index(drop=True).copy()
            if df_dom.empty:
                continue

            features_dom = features[df_dom.index.to_numpy()]
            centroid_dom = df_dom[["cx", "cy"]].to_numpy()

            if len(df_dom) <= 1:
                chosen = list(range(len(df_dom)))
            else:
                chosen = self.greedy_diverse_subset(
                    features_dom,
                    centroid_dom,
                    n=n,
                )

            if not chosen:
                continue

            df_sel = df_dom.iloc[chosen].copy()
            df_sel["domain_tile_rank"] = np.arange(1, len(df_sel) + 1, dtype=int)
            selected_rows.append(df_sel)

        if not selected_rows:
            return tile_table.iloc[0:0].copy()

        selected_df = pd.concat(selected_rows, axis=0, ignore_index=True)
        return selected_df


    def check_min_distance(
        self,
        df: pd.DataFrame,
        min_distance_px: float,
    ) -> bool:
        """
        Verify that all selected tiles within each domain are at least `min_distance_px` apart.
        """
        if df.empty or min_distance_px <= 0:
            return True

        min_d2 = float(min_distance_px**2)

        for _, group in df.groupby(self.domain_col):
            coords = group[["cx", "cy"]].to_numpy()
            if len(coords) <= 1:
                continue

            diff = coords[:, None, :] - coords[None, :, :]
            d2 = np.sum(diff**2, axis=2)
            mask = np.triu(np.ones_like(d2, dtype=bool), k=1)
            if mask.any() and np.any(d2[mask] < min_d2):
                return False

        return True
    