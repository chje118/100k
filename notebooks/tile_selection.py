import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from wsidata import open_wsi


def _get_features_and_tiles(
    wsi,
    tile_key: str,
    feature_key: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Internal helper to fetch tile geometries and features and align them.

    Returns
    -------
    tiles_gdf : GeoDataFrame-like (returned as-is, but typed as DataFrame here)
    features  : np.ndarray of shape (n_tiles, d)
    """
    tiles_gdf = wsi.shapes[tile_key]
    adata = wsi[feature_key]

    # AnnData.X may be sparse; convert to dense for distance computations.
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


def load_tile_table(
    wsi,
    tile_key: str = "tiles_224",
    feature_key: str = "features_conch",
    domain_key: str = "domain",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build a per-tile table with domain labels, centroids and features.

    This function is robust to `domain` being stored either in the
    feature AnnData's `.obs` or directly on the tile GeoDataFrame.

    Parameters
    ----------
    wsi
        Lazyslide / wsidata WSI object.
    tile_key
        Key in `wsi.shapes` holding tile polygons.
    feature_key
        Key in `wsi` holding tile-level features (AnnData).
    domain_key
        Column / obs key where spatial domains are stored.

    Returns
    -------
    meta_df : DataFrame
        Columns: ['tile_idx', 'domain', 'cx', 'cy'] (and any extra
        columns copied from the domain source).
    features : np.ndarray
        Feature matrix of shape (n_tiles, n_features).
    """
    tiles_gdf, features = _get_features_and_tiles(wsi, tile_key, feature_key)
    adata = wsi[feature_key]

    # Determine domain labels.
    if domain_key in adata.obs:
        domains = adata.obs[domain_key].to_numpy()
    elif domain_key in tiles_gdf.columns:
        domains = tiles_gdf[domain_key].to_numpy()
    else:
        raise KeyError(
            f"Domain key '{domain_key}' not found in either "
            "feature AnnData.obs or tile GeoDataFrame."
        )

    # Compute centroids in the tile coordinate system used by lazyslide.
    centroids = tiles_gdf.geometry.centroid
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()

    tile_idx = np.arange(len(tiles_gdf), dtype=int)

    meta_df = pd.DataFrame(
        {
            "tile_idx": tile_idx,
            "domain": domains,
            "cx": cx,
            "cy": cy,
        }
    )

    return meta_df, features


def _standardize_features(F: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance scaling per feature, with numerical safety."""
    if F.size == 0:
        return F
    mean = F.mean(axis=0, keepdims=True)
    std = F.std(axis=0, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)
    return (F - mean) / std_safe


def _greedy_diverse_subset(
    F: np.ndarray,
    C: np.ndarray,
    n: int,
    min_distance: float,
    score_mode: Literal["maxmin", "sum"] = "maxmin",
    on_fail: Literal["stop", "relax"] = "stop",
    relax_factor: float = 0.9,
    min_relax_ratio: float = 0.3,
) -> List[int]:
    """
    Greedy diversity sampling with a minimum spatial distance constraint.

    Parameters
    ----------
    F
        Feature matrix (m x d) for tiles in a single domain.
    C
        Centroid coordinates (m x 2) for the same tiles.
    n
        Desired number of tiles.
    min_distance
        Minimum allowed distance between any two selected centroids, in
        the same units as `C`.
    score_mode
        'maxmin' → maximize minimum feature distance to current set.
        'sum'    → maximize sum of feature distances to current set.
    on_fail
        What to do if the spatial constraint is too strict:
        - 'stop': return fewer than `n` tiles.
        - 'relax': iteratively relax `min_distance` by `relax_factor`
          until at least one candidate remains or the distance has been
          reduced to `min_relax_ratio * min_distance`.

    Returns
    -------
    selected_indices : list[int]
        Indices into F / C (0..m-1) of selected tiles.
    """
    m = F.shape[0]
    if m == 0 or n <= 0:
        return []

    if m == 1:
        return [0]

    # Standardize features for more isotropic distance measures.
    F_std = _standardize_features(F)

    # Seed: farthest from the domain mean in feature space.
    mean_vec = F_std.mean(axis=0, keepdims=True)
    d2_center = np.sum((F_std - mean_vec) ** 2, axis=1)
    first_idx = int(np.argmax(d2_center))

    selected: List[int] = [first_idx]
    available = np.ones(m, dtype=bool)
    available[first_idx] = False

    current_min_distance = float(min_distance)
    base_min_distance = float(min_distance)

    def spatial_mask(min_dist: float) -> np.ndarray:
        """Return boolean mask of candidates satisfying spatial constraint."""
        if min_dist <= 0 or len(selected) == 0:
            return available.copy()

        idx_candidates = np.nonzero(available)[0]
        if idx_candidates.size == 0:
            return available.copy()

        sel_coords = C[selected]  # (k, 2)
        cand_coords = C[idx_candidates]  # (c, 2)
        diff = cand_coords[:, None, :] - sel_coords[None, :, :]
        d2 = np.sum(diff**2, axis=2)  # (c, k)
        min_d2 = d2.min(axis=1)

        keep = min_d2 >= (min_dist**2)
        mask = np.zeros_like(available)
        mask[idx_candidates[keep]] = True
        return mask

    while len(selected) < min(n, m):
        mask = spatial_mask(current_min_distance)
        candidate_idx = np.nonzero(mask)[0]

        if candidate_idx.size == 0:
            if on_fail == "relax" and current_min_distance > base_min_distance * min_relax_ratio:
                current_min_distance *= relax_factor
                continue
            # Cannot add more tiles under spatial constraint.
            break

        # Compute distances in feature space to existing selection.
        F_sel = F_std[selected]  # (k, d)
        F_cand = F_std[candidate_idx]  # (c, d)

        # (c, k) matrix of squared distances.
        diff_f = F_cand[:, None, :] - F_sel[None, :, :]
        d2 = np.sum(diff_f**2, axis=2)

        if score_mode == "sum":
            scores = d2.sum(axis=1)
        else:  # "maxmin"
            scores = d2.min(axis=1)

        best_local = int(np.argmax(scores))
        best_idx = int(candidate_idx[best_local])

        selected.append(best_idx)
        available[best_idx] = False

    return selected


def select_tiles_per_domain(
    tile_table: pd.DataFrame,
    features: np.ndarray,
    n: int,
    min_distance_px: float,
    score_mode: Literal["maxmin", "sum"] = "maxmin",
    on_fail: Literal["stop", "relax"] = "stop",
) -> pd.DataFrame:
    """
    Run greedy diversity sampling independently within each domain.

    Parameters
    ----------
    tile_table
        DataFrame as returned from `load_tile_table`.
    features
        Feature matrix aligned with `tile_table` rows.
    n
        Number of tiles to select per domain (upper bound).
    min_distance_px
        Minimum allowed distance between selected tiles in pixels.
    score_mode
        See `_greedy_diverse_subset`.
    on_fail
        See `_greedy_diverse_subset`.

    Returns
    -------
    selected_df : DataFrame
        Subset of `tile_table` with one row per selected tile, plus
        a column `domain_tile_rank` giving the selection order within
        each domain.
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
            chosen = _greedy_diverse_subset(
                F_dom,
                C_dom,
                n=n,
                min_distance=min_distance_px,
                score_mode=score_mode,
                on_fail=on_fail,
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


def _check_min_distance(
    df: pd.DataFrame,
    min_distance_px: float,
) -> bool:
    """
    Verify that all selected tiles within each domain are at least
    `min_distance_px` apart. Returns True if the constraint holds.
    """
    if df.empty or min_distance_px <= 0:
        return True

    min_d2 = float(min_distance_px**2)

    for dom, group in df.groupby("domain"):
        coords = group[["cx", "cy"]].to_numpy()
        if len(coords) <= 1:
            continue

        # Pairwise distances (upper triangle only).
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff**2, axis=2)
        # Ignore diagonal zeros.
        mask = np.triu(np.ones_like(d2, dtype=bool), k=1)
        if mask.any() and np.any(d2[mask] < min_d2):
            return False

    return True


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
    """
    Open a single WSI, select tiles per spatial domain and return a tidy DataFrame.

    Parameters
    ----------
    wsi_path
        Path to the MRXS (or other supported) slide.
    zarr_dir
        Directory where lazyslide zarr stores are kept.
    tile_key, feature_key, domain_key
        Keys controlling which tiles/features/domains to use.
    n_per_domain
        Maximum number of tiles to select per domain.
    min_distance_px
        Minimum distance between selected tiles in pixels.
    score_mode, on_fail
        Passed through to `_greedy_diverse_subset`.
    run_sanity_check
        If True, verify the min-distance constraint after selection.

    Returns
    -------
    DataFrame with columns:
        ['slide_id', 'tile_idx', 'domain', 'cx', 'cy', 'domain_tile_rank']
    """
    slide_id = os.path.basename(wsi_path).replace(".mrxs", "")
    zarr_path = os.path.join(zarr_dir, slide_id + ".zarr")

    if os.path.exists(zarr_path):
        wsi = open_wsi(wsi_path, zarr_path)
    else:
        wsi = open_wsi(wsi_path)

    tile_table, features = load_tile_table(
        wsi,
        tile_key=tile_key,
        feature_key=feature_key,
        domain_key=domain_key,
    )

    selected_df = select_tiles_per_domain(
        tile_table,
        features=features,
        n=n_per_domain,
        min_distance_px=min_distance_px,
        score_mode=score_mode,
        on_fail=on_fail,
    ).copy()

    selected_df.insert(0, "slide_id", slide_id)

    if run_sanity_check:
        ok = _check_min_distance(selected_df, min_distance_px=min_distance_px)
        if not ok:
            raise RuntimeError(
                "Minimum distance constraint violated in the selected tiles."
            )

    return selected_df

