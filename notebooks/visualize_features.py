
import os
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tissue_artifact_segmentation import load_big_cache, save_big_cache
from wsidata import open_wsi
from tqdm import tqdm


def build_diagnose(topography, morphology):
    if isinstance(topography, list):
        topography = " / ".join(str(t) for t in topography)
    
    if isinstance(morphology, list):
        return [f"{topography} - {m}" for m in morphology]
    
    return f"{topography} - {morphology}"

def apply_diagnose(df, topo_col, morph_col, new_col="diagnose"):
    df[new_col] = df.apply(
        lambda row: build_diagnose(row[topo_col], row[morph_col]),
        axis=1
    )
    return df

class FeatureDataBuilder:
    def __init__(self, filenames, df_metadata: pd.DataFrame, zarr_dir: str, cache_file: str, models: list[str]):
        self.filenames = filenames
        self.df_metadata = df_metadata
        self.zarr_dir = zarr_dir
        self.cache_file = cache_file
        self.models = models
        self.cached_data = load_big_cache(cache_file)
        self.df_merged = ()
        self.run()
        
    def subset_metadata(self):
        return self.df_metadata[self.df_metadata["filename"].isin(self.filenames)].copy()    

    def extract_from_cache(self):
        rows = []
        for slide, categories in self.cached_data.items():
            row = {"filename": slide}
        
            # Tissue size
            tissue_default = categories.get("tissue", {}).get("default")
            if tissue_default and tissue_default.get("status") == "complete":
                row["tissue_default_size"] = tissue_default.get("size")
            else:
                row["tissue_default_size"] = None
        
            # Artifact percentage
            artifact_default = categories.get("artifact", {}).get("default")
            if artifact_default and artifact_default.get("status") == "complete":
                row["artifact_default_pct"] = artifact_default.get("pct")
            else:
                row["artifact_default_pct"] = None
        
        rows.append(row)
        return pd.DataFrame(rows)

    def extract_features(self):
        rows = []
        for slide_path in tqdm(self.filenames, desc="Extracting features"):
            basename = os.path.basename(slide_path)
            row = {"filename": slide_path}
            try:
                zarr_path = os.path.join(self.zarr_dir, basename.replace(".mrxs", ".zarr"))
                wsi = open_wsi(slide_path, zarr_path)
                for model in self.models:
                    feature_key = (f"features_{model}")
                    adata = wsi[feature_key]
                    row[f"features_{model}"] = adata
            except Exception as e:
                row["error"] = str(e)

            rows.append(row)
        return pd.DataFrame(rows)
    
    def run(self):
        df_sub = self.subset_metadata()
        df_cache = self.extract_from_cache()
        df_features = self.extract_features()
        self.df_merged = (df_sub
            .merge(df_cache, on="filename", how="left")
            .merge(df_features, on="filename", how="left")
        )
        return self.df_merged 


class FeatureVisualizer:
    def __init__(self, df: pd.DataFrame, label_col: str, features_col: str = "features", artifact_col: Optional[str] = None):
        self.df = df
        self.label_col = label_col
        self.features_col = features_col
        self.artifact_col = artifact_col
        self._cache = {}

        for col in [label_col, features_col]:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        if artifact_col and artifact_col not in df.columns:
            raise KeyError(f"Missing artifact column: {artifact_col}")
        
        self.df[label_col] = self.df[label_col].apply(self._normalize_label)

    def _normalize_label(self, lbl):
        if isinstance(lbl, list):
            return " / ".join(map(str, lbl))
        return lbl

    def _extract_embeddings(self):
        """ Return slide-level aggregated embedding matrix and labels. """
        if "agg" in self._cache:
            return self._cache["agg"]

        embeddings = []
        labels = []
        artifacts = []

        for _, row in self.df.iterrows():
            adata = row[self.features_col]
            emb = np.ravel(adata.varm["agg_slide"])

            embeddings.append(emb)
            labels.append(row[self.label_col])
            
            if self.artifact_col is None:
                artifact_pct = 0.0
            else:
                val = row[self.artifact_col]
                artifact_pct = 0.0 if pd.isna(val) else float(val)

            artifacts.append(artifact_pct)

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        artifacts = np.array(artifacts)

        self._cache["agg"] = (embeddings, labels, artifacts)
        return self._cache["agg"]

    def _standardize_features(self, emb):
        if "scaler" not in self._cache:
            self._cache["scaler"] = StandardScaler().fit(emb)
        return self._cache["scaler"].transform(emb)

    def _encode_labels(self, lbls):
        if "label_encoder" not in self._cache:
            self._cache["label_encoder"] = LabelEncoder().fit(lbls)
        return self._cache["label_encoder"].transform(lbls)

    def _auto_pca_components(self, features_scaled, var_threshold=0.95)-> int:
        pca_temp = PCA().fit(features_scaled)
        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n = int(np.searchsorted(cum_var, var_threshold) + 1)
        return max(2, n)
    
    def _auto_tsne_perplexity(self, n_samples: int) -> int:
        return int(max(5, min(50, n_samples // 3)))

    def pca_plot(self, figsize=(8, 10)):
        emb, labels, artifacts = self._extract_embeddings()
        emb_scaled = self._standardize_features(emb)
        #labels_encoded = self._encode_labels(labels)
    
        n_components = self._auto_pca_components(emb_scaled)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(emb_scaled)

        alpha_vals = 1 - artifacts

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        
        plt.figure(figsize=figsize)
        
        for i, lbl in enumerate(unique_labels):
            idx = labels == lbl
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                alpha=alpha_vals[idx],
                color=cmap(i % cmap.N),
                label=str(lbl),
                s=40
            )
        plt.title("PCA Visualization")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(
            title=self.label_col,
            bbox_to_anchor=(1.04, 1),
            loc="upper left"
        )
        plt.show()      

    def tsne_plot(self, figsize=(8, 10)):
        emb, labels, artifacts = self._extract_embeddings()
        emb_scaled = self._standardize_features(emb)
        
        perplexity = self._auto_tsne_perplexity(len(emb_scaled))
    
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca"
        )
        reduced = tsne.fit_transform(emb_scaled)

        alpha_vals = 1 - artifacts

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        
        plt.figure(figsize=figsize)
        
        for i, lbl in enumerate(unique_labels):
            idx = labels == lbl
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                alpha=alpha_vals[idx],
                color=cmap(i % cmap.N),
                label=str(lbl),
                s=40
            )
        plt.title("t-SNE Visualization")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(
            title=self.label_col,
            bbox_to_anchor=(1.04, 1),
            loc="upper left"
        )
        plt.show()

    def label_separation_score(self):
        emb, labels, artifacts = self._extract_embeddings()
        emb_scaled = self._standardize_features(emb)
        labels_encoded = self._encode_labels(labels)

        # 1) Silhouette Score (higher = better)
        if len(np.unique(labels_encoded)) > 1:
            sil = silhouette_score(emb_scaled, labels_encoded)
        else:
            sil = np.nan
        
        # 2) Davies–Bouldin Score (lower = better)
        if len(np.unique(labels_encoded)) > 1:
            db = davies_bouldin_score(emb_scaled, labels_encoded)
        else:
            db = np.nan

        # 3) Fisher ratio (higher = better)
        overall_mean = emb_scaled.mean(axis=0)
        between_var = 0
        within_var = 0

        for cls in np.unique(labels_encoded):
            cls_emb = emb_scaled[labels_encoded == cls]
            cls_mean = cls_emb.mean(axis=0)

            n = len(cls_emb)
        
            between_var += n * np.sum((cls_mean - overall_mean) ** 2)
            within_var += np.sum((cls_emb - cls_mean) ** 2)

        fisher_ratio = between_var / within_var if within_var > 0 else np.nan

        return sil, db, fisher_ratio

    def print_score(self):
        sil, db, fisher_ratio = self.label_separation_score()
        print(f"Silhouette: {sil:.3f} (−1 to 1, higher → better separation)")
        print(f"Davies–Bouldin: {db:.3f} (0 to ∞, lower → better clustering)")
        print(f"Fisher Ratio: {fisher_ratio:.3f} (higher → better separation)")


# Example usage:
if __name__ == "__main__":
    # Build diagnose column
    df = pd.read_csv("path/to/dataframe.csv")
    df = apply_diagnose(df, "T category", "M category")