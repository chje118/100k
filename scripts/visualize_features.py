import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from wsidata import open_wsi

from helper_functions import subset_df_processed


class FeatureVisualizer:
    def __init__(self, df: pd.DataFrame, zarr_dir: str, cache_path: str, model: str, label_col: str, aggregated: bool = True, filename_col: str = "filename"):
        self.df = df.copy()
        self.zarr_dir = zarr_dir
        self.cache_path = cache_path
        self.model = str(model)
        self.feature_key = f"features_{self.model}"
        self.label_col = label_col
        self.aggregated = aggregated
        self.filename_col = filename_col

    def processed_entries(self):
        return subset_df_processed(
            self.df,
            self.cache_path,
            category="features",
            status="complete",
            model=self.model,
            filename_col=self.filename_col,
        ).reset_index(drop=True)

    def _load_adata(self, wsi_path: str):
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(wsi_path, zarr_path) if os.path.exists(zarr_path) else open_wsi(wsi_path)
        return wsi.tables[self.feature_key]

    def _get_slide_embedding(self, adata) -> np.ndarray:
        if "agg_slide" not in adata.varm:
            raise KeyError(f"'agg_slide' not found in {self.feature_key}.varm")
        emb = np.asarray(adata.varm["agg_slide"]).ravel().astype(np.float32)
        return emb

    def _get_tile_embeddings(self, adata, every_nth: int = 1) -> np.ndarray:
        X = np.asarray(adata.X, dtype=np.float32)
        if every_nth > 1:
            X = X[::every_nth]
        return X

    def build_embedding_table(self, mode: str = "aggregated", every_nth: int = 1):
        df_proc = self.processed_entries()

        embeddings = []
        labels = []
        slide_paths = []

        for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc="Loading embeddings..."):
            wsi_path = row[self.filename_col]
            label = row[self.label_col]

            try:
                adata = self._load_adata(wsi_path)

                if mode == "aggregated":
                    emb = self._get_slide_embedding(adata)
                    embeddings.append(emb)
                    labels.append(label)
                    slide_paths.append(wsi_path)

                elif mode == "tile":
                    X = self._get_tile_embeddings(adata, every_nth=every_nth)
                    if len(X) == 0:
                        del adata
                        gc.collect()
                        continue

                    embeddings.append(X)
                    labels.extend([label] * len(X))
                    slide_paths.extend([wsi_path] * len(X))

                else:
                    raise ValueError("mode must be 'aggregated' or 'tile'")

                del adata
                gc.collect()

            except Exception as e:
                print(f"Skipping {os.path.basename(wsi_path)}: {type(e).__name__}: {e}")
                gc.collect()

        if not embeddings:
            raise ValueError("No embeddings could be loaded.")

        if mode == "aggregated":
            emb = np.vstack(embeddings).astype(np.float32)
        else:
            emb = np.concatenate(embeddings, axis=0).astype(np.float32)

        out = pd.DataFrame({
            "slide_path": slide_paths,
            self.label_col: labels,
        })

        del embeddings
        gc.collect()

        return emb, out

    def _standardize_features(self, emb: np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(emb).astype(np.float32)

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        return LabelEncoder().fit_transform(labels)

    def _reduce_pca(self, emb: np.ndarray, n_components: int = 2):
        emb_scaled = self._standardize_features(emb)
        reduced = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=42
        ).fit_transform(emb_scaled)
        return reduced, emb_scaled

    def _reduce_tsne(self, emb: np.ndarray, n_components: int = 2, perplexity: int | None = None):
        emb_scaled = self._standardize_features(emb)

        if perplexity is None:
            perplexity = max(5, min(30, len(emb_scaled) // 3))

        reduced = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(emb_scaled)

        return reduced, emb_scaled

    def silhouette_score(self, emb: np.ndarray, labels: np.ndarray):
        emb_scaled = self._standardize_features(emb)
        labels_encoded = self._encode_labels(labels)

        if len(np.unique(labels_encoded)) > 1:
            sil = silhouette_score(emb_scaled, labels_encoded)
        else:
            sil = np.nan

        return sil

    def plot_embeddings(self, method: str = "pca", mode: str = "aggregated", every_nth: int = 1):
        emb, meta = self.build_embedding_table(mode=mode, every_nth=every_nth)
        labels = meta[self.label_col].to_numpy()

        sil = self.silhouette_score(emb, labels)

        if method == "pca":
            reduced, _ = self._reduce_pca(emb, n_components=2)
            xlab, ylab = "PC1", "PC2"
            title = f"PCA ({mode}, silhouette score: {sil:.3f})"

        elif method == "tsne":
            reduced, _ = self._reduce_tsne(emb, n_components=2)
            xlab, ylab = "Dim 1", "Dim 2"
            title = f"t-SNE ({mode}, silhouette score: {sil:.3f})"

        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        unique_labels = pd.unique(labels)
        cmap = plt.get_cmap("tab20")

        plt.figure(figsize=(8, 8))

        for i, lbl in enumerate(unique_labels):
            idx = labels == lbl
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                s=20,
                alpha=0.8,
                color=cmap(i % cmap.N),
                label=str(lbl),
                rasterized=(mode == "tile"),
            )

        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(title=self.label_col, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()