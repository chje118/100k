import os
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from wsidata import open_wsi


class TileSelector:
    """
    Class-based tile selection:
    - builds per-tile metadata/feature tables
    - performs greedy diversity sampling per domain
    - enforces a minimum spatial distance constraint
    """

    def __init__(
        self,
        feature_key: str = "features_conch",
        tile_key: str = "tiles_224",
        domain_key: str = "domain",
        n_per_domain: int = 10,
        min_distance_px: float = 500.0,
        score_mode: Literal["maxmin", "sum"] = "maxmin",
        on_fail: Literal["stop", "relax"] = "stop",
    ) -> None:
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.domain_key = domain_key
        self.n_per_domain = n_per_domain
        self.min_distance_px = float(min_distance_px)
        self.score_mode = score_mode
        self.on_fail = on_fail

    # ------- Data access -------

    def get_features_and_tiles(self, wsi) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fetch tile geometries and features and ensure they are aligned.
        """
        tiles_gdf = wsi.shapes[self.tile_key]
        adata = wsi.tables[self.feature_key]

        X = adata.X
        if hasattr(X, "toarray"):
            features = X.toarray()
        else:
            features = np.asarray(X)

        if len(tiles_gdf) != features.shape[0]:
            raise ValueError(
                f"Number of tiles ({len(tiles_gdf)}) and feature rows "
                f"({features.shape[0]}) do not match."
            )

        return tiles_gdf, features

    def load_tile_table(self, wsi) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build a per-tile table with domain labels, centroids and features.
        """
        tiles_gdf, features = self.get_features_and_tiles(wsi)

        if self.domain_key in tiles_gdf.columns:
            domains = tiles_gdf[self.domain_key].to_numpy()
        else:
            raise KeyError(
                f"Domain key '{self.domain_key}' not found in tile GeoDataFrame."
            )

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

        return meta_df, features

    # ------- Feature utilities -------

    @staticmethod
    def standardize_features(features: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance scaling per feature, with numerical safety."""
        if features.size == 0:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std_safe = np.where(std == 0, 1.0, std)
        return (features - mean) / std_safe

    # ------- Core greedy sampler -------

    def greedy_diverse_subset(
        self,
        features: np.ndarray,
        centroids: np.ndarray,
        n: int,
    ) -> List[int]:
        """
        Greedy diversity sampling with a minimum spatial distance constraint.
        """
        m = features.shape[0]
        if m == 0 or n <= 0:
            return []

        if m == 1:
            return [0]

        F_std = self.standardize_features(features)

        mean_vec = F_std.mean(axis=0, keepdims=True)
        d2_center = np.sum((F_std - mean_vec) ** 2, axis=1)
        first_idx = int(np.argmax(d2_center))

        selected: List[int] = [first_idx]
        available = np.ones(m, dtype=bool)
        available[first_idx] = False

        current_min_distance = self.min_distance_px
        base_min_distance = self.min_distance_px

        while len(selected) < min(n, m):
            # Enforce spatial constraint.
            if current_min_distance > 0 and selected:
                idx_candidates = np.nonzero(available)[0]
                if idx_candidates.size == 0:
                    break

                sel_coords = centroids[selected]
                cand_coords = centroids[idx_candidates]
                diff = cand_coords[:, None, :] - sel_coords[None, :, :]
                d2 = np.sum(diff**2, axis=2)
                min_d2 = d2.min(axis=1)

                keep = min_d2 >= (current_min_distance**2)
                mask = np.zeros_like(available)
                mask[idx_candidates[keep]] = True
            else:
                mask = available.copy()

            candidate_idx = np.nonzero(mask & available)[0]

            if candidate_idx.size == 0:
                if (
                    self.on_fail == "relax"
                    and current_min_distance > base_min_distance * 0.3
                ):
                    current_min_distance *= 0.9
                    continue
                break

            F_sel = F_std[selected]
            F_cand = F_std[candidate_idx]
            diff_f = F_cand[:, None, :] - F_sel[None, :, :]
            d2 = np.sum(diff_f**2, axis=2)

            if self.score_mode == "sum":
                scores = d2.sum(axis=1)
            else:
                scores = d2.min(axis=1)

            best_local = int(np.argmax(scores))
            best_idx = int(candidate_idx[best_local])

            selected.append(best_idx)
            available[best_idx] = False

        return selected


    # ------- Domain-level selection -------

    def select_tiles_per_domain(
        self,
        tile_table: pd.DataFrame,
        features: np.ndarray,
        n: int,
    ) -> pd.DataFrame:
        """
        Run greedy diversity sampling independently within each domain.
        """
        if "domain" not in tile_table.columns:
            raise KeyError("Expected column 'domain' in tile_table.")

        if features.shape[0] != len(tile_table):
            raise ValueError(
                "features and tile_table must have the same number of rows."
            )

        selected_rows: List[pd.DataFrame] = []

        for dom in sorted(tile_table["domain"].dropna().unique()):
            df_dom = tile_table[tile_table["domain"] == dom].copy()
            if df_dom.empty:
                continue

            idx_dom = df_dom.index.to_numpy()
            F_dom = features[idx_dom]
            C_dom = df_dom[["cx", "cy"]].to_numpy()

            if len(df_dom) <= 1:
                chosen = list(range(len(df_dom)))
            else:
                chosen = self.greedy_diverse_subset(
                    F_dom,
                    C_dom,
                    n=n,
                )

            if not chosen:
                continue

            df_sel = df_dom.iloc[chosen].copy()
            df_sel["domain_tile_rank"] = np.arange(1, len(df_sel) + 1, dtype=int)
            selected_rows.append(df_sel)

        if not selected_rows:
            return tile_table.iloc[0:0].copy()

        selected_df = pd.concat(selected_rows, axis=0)
        return selected_df


    # ------- Sanity check -------

    @staticmethod
    def check_min_distance(
        df: pd.DataFrame,
        min_distance_px: float,
    ) -> bool:
        """
        Verify that all selected tiles within each domain are at least
        `min_distance_px` apart.
        """
        if df.empty or min_distance_px <= 0:
            return True

        min_d2 = float(min_distance_px**2)

        for _, group in df.groupby("domain"):
            coords = group[["cx", "cy"]].to_numpy()
            if len(coords) <= 1:
                continue

            diff = coords[:, None, :] - coords[None, :, :]
            d2 = np.sum(diff**2, axis=2)
            mask = np.triu(np.ones_like(d2, dtype=bool), k=1)
            if mask.any() and np.any(d2[mask] < min_d2):
                return False

        return True


    # ------- High-level WSI API -------

    def select_tiles_for_wsi(
        self,
        wsi_path: str,
        zarr_dir: str,
        run_sanity_check: bool = True,
    ) -> pd.DataFrame:
        """
        Open a single WSI, select tiles per spatial domain and return a tidy DataFrame.
        """
        slide_id = os.path.basename(wsi_path).replace(".mrxs", "")
        zarr_path = os.path.join(zarr_dir, slide_id + ".zarr")

        if os.path.exists(zarr_path):
            wsi = open_wsi(wsi_path, zarr_path)
        else:
            wsi = open_wsi(wsi_path)

        tile_table, features = self.load_tile_table(wsi)

        selected_df = self.select_tiles_per_domain(
            tile_table,
            features=features,
            n=self.n_per_domain,
        ).copy()

        selected_df.insert(0, "slide_id", slide_id)

        if run_sanity_check:
            ok = self.check_min_distance(
                selected_df,
                min_distance_px=self.min_distance_px,
            )
            if not ok:
                raise RuntimeError(
                    "Minimum distance constraint violated in the selected tiles."
                )

        return selected_df


# ------- Thin function wrappers (backwards compatible) -------

def load_tile_table(wsi, feature_key: str, domain_key: str, tile_key: str = "tiles_224"):
    selector = TileSelector(
        feature_key=feature_key,
        tile_key=tile_key,
        domain_key=domain_key,
    )
    return selector.load_tile_table(wsi)


def select_tiles_per_domain(
    tile_table: pd.DataFrame,
    features: np.ndarray,
    n: int,
    min_distance_px: float,
    score_mode: Literal["maxmin", "sum"] = "maxmin",
    on_fail: Literal["stop", "relax"] = "stop",
) -> pd.DataFrame:
    selector = TileSelector(
        min_distance_px=min_distance_px,
        score_mode=score_mode,
        on_fail=on_fail,
    )
    return selector.select_tiles_per_domain(tile_table, features, n)


def select_tiles_for_wsi(
    wsi_path: str,
    zarr_dir: str,
    tile_key: str = "tiles_224",
    feature_key: str = "features_conch",
    domain_key: str = "domain",
    n_per_domain: int = 10,
    min_distance_px: float = 500.0,
    score_mode: Literal["maxmin", "sum"] = "maxmin",
    on_fail: Literal["stop", "relax"] = "stop",
    run_sanity_check: bool = True,
) -> pd.DataFrame:
    selector = TileSelector(
        feature_key=feature_key,
        tile_key=tile_key,
        domain_key=domain_key,
        n_per_domain=n_per_domain,
        min_distance_px=min_distance_px,
        score_mode=score_mode,
        on_fail=on_fail,
    )
    return selector.select_tiles_for_wsi(
        wsi_path=wsi_path,
        zarr_dir=zarr_dir,
        run_sanity_check=run_sanity_check,
    )
