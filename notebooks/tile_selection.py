import os
from pyexpat import features
from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
from wsidata import open_wsi


class TileSelector:
    def __init__(
        self,
        wsi,
        feature_key: str,
        domain_key: str,
        tile_key: str = "tiles_224",
        n_per_domain: int = 10,
        min_distance_px: float = 500.0,
        score_mode: Literal["maxmin", "sum"] = "maxmin",
        on_fail: Literal["stop", "relax"] = "relax",
    ) -> None:
        self.wsi = wsi
        self.feature_key = feature_key
        self.domain_key = domain_key
        self.tile_key = tile_key
        self.n_per_domain = n_per_domain
        self.min_distance_px = float(min_distance_px)
        self.score_mode = score_mode
        self.on_fail = on_fail
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

    def greedy_diverse_subset(self) -> List[int]:
        """
        Greedy diversity sampling with a minimum spatial distance constraint.
        """
        m = self.features.shape[0]
        if m == 0 or self.n_per_domain <= 0:
            return []

        if m == 1:
            return [0]

        features_std = self.standardize_features(self.features)

        mean_vec = features_std.mean(axis=0, keepdims=True)
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

    def select_tiles_per_domain(self) -> pd.DataFrame:
        """
        Run greedy diversity sampling independently within each domain.
        """
        if tile_table is None or features is None:
            if self.meta_df is None or self.features is None:
                raise ValueError(
                    "No cached tile_table/features. Call `load_tile_table()` first "
                    "or pass `tile_table` and `features` explicitly."
                )
            tile_table = self.meta_df
            features = self.features

        if n is None:
            n = self.n_per_domain

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
