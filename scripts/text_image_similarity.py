import os
import pandas as pd
import numpy as np
import lazyslide as zs
from wsidata import agg_wsi, open_wsi
from tqdm import tqdm
import matplotlib.pyplot as plt


class SlideClassification:
    def __init__(self, wsi_path: str, local_zarr_dir: str, classes: list[str], text_embeddings,
                 model="conch", recompute=False, tissue="tissue_default", tiles="tiles_224"):
        self.wsi_path = wsi_path
        self.slide_id = os.path.basename(wsi_path).replace(".mrxs", "")
        self.zarr_path = os.path.join(local_zarr_dir, self.slide_id + ".zarr")

        if os.path.exists(self.zarr_path):
            self.wsi = open_wsi(self.wsi_path, self.zarr_path)
        else:
            self.wsi = open_wsi(self.wsi_path)
            os.makedirs(local_zarr_dir, exist_ok=True)

        self.classes = classes
        self.text_embeddings = text_embeddings
        self.model = model
        self.recompute = recompute
        self.TISSUE_KEY = tissue
        self.TILE_KEY = tiles
        self.FEATURE_KEY = f"features_{model}"
        self.SIMILARITY_KEY = f"{model}_text_similarity"

        self.compute_similarity()

    def compute_similarity(self):
        """Compute similarity between slide features and text embeddings."""
        if self.SIMILARITY_KEY in self.wsi and not self.recompute:
            return

        zs.tl.text_image_similarity(
            self.wsi,
            self.text_embeddings,
            model=self.model,
            tile_key=self.TILE_KEY,
            feature_key=self.FEATURE_KEY,
            key_added=self.SIMILARITY_KEY
        )
        self.wsi.write(self.zarr_path)

    def get_similarity_adata(self):
        return self.wsi[self.SIMILARITY_KEY]

    def get_MIL_scores(self, k: int = 10, agg_method="mean"):
        sim_adata = self.get_similarity_adata()
        k_eff = min(k, sim_adata.n_obs)
        scores = zs.metrics.topk_score(sim_adata, k=k_eff, agg_method=agg_method)
        return scores
    
    def heatmap(self, clss):
        print(f"Heatmap of class: {clss}")
        zs.pl.tiles(self.wsi, feature_key=self.SIMILARITY_KEY, color=clss, tile_key=self.TILE_KEY, alpha=0.9, show_contours=False)
    
    def all_heatmaps(self, n_cols: int = 4):
        n_classes = len(self.classes)
        n_rows = int(np.ceil(n_classes / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
        )

        axes = np.atleast_1d(axes).flatten()

        for ax, clss in zip(axes, self.classes):
            zs.pl.tiles(
                self.wsi,
                tile_key=self.TILE_KEY,
                feature_key=self.SIMILARITY_KEY,
                color=clss,
                alpha=0.8,
                show_image=False,
                show_contours=False,
                ax=ax,
            )
            ax.set_title(clss)
            ax.axis("off")

        for ax in axes[len(self.classes):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


class ClassifyMany:
    def __init__(self, wsi_paths: list[str], local_zarr_dir: str, classes: list[str], dataset: pd.DataFrame,
                 model="conch", recompute=False):
        self.wsi_paths = wsi_paths
        self.local_zarr_dir = local_zarr_dir
        self.classes = classes
        self.dataset = dataset
        self.model = model
        self.recompute = recompute

        print("Embedding class prompts...")
        self.text_embeddings = zs.tl.text_embedding(self.classes, model=self.model)

    def run_MIL(self, k: int = 10):
        slide_scores = {}
        valid_slides = set()

        # ---- Compute MIL scores for each slide ----
        for wsi_path in tqdm(self.wsi_paths):
            slide_id = os.path.basename(wsi_path).replace(".mrxs", "")
            try:
                classifier = SlideClassification(
                    wsi_path,
                    self.local_zarr_dir,
                    self.classes,
                    self.text_embeddings,
                    model=self.model,
                    recompute=self.recompute,
                )
                scores = classifier.get_MIL_scores(k)
                slide_scores[slide_id] = dict(zip(self.classes, scores))
                valid_slides.add(slide_id)
            except Exception as e:
                print(f"Skipping {slide_id}: {e}")

        # ---- Convert MIL scores to DataFrame ----
        self.results_df = pd.DataFrame(slide_scores).T.astype(float)

        # ---- Prepare dataset for aggregation ----
        dataset = self.dataset.copy()
        dataset["Tissue Sample Id"] = dataset["filename"].apply(
            lambda x: os.path.basename(x).replace(".mrxs", "")
        )
        dataset = dataset[dataset["Tissue Sample Id"].isin(valid_slides)]
        dataset["store"] = dataset["Tissue Sample Id"].apply(
            lambda x: os.path.join(self.local_zarr_dir, x + ".zarr")
        )
        dataset = dataset[dataset["store"].apply(os.path.exists)]

        if dataset.empty:
            raise RuntimeError("No valid slides available for aggregation.")

        for col in ["filename", "Tissue Sample Id"]:
            dataset[col] = dataset[col].astype(str)

        # ---- Aggregate slides ----
        try:
            self.agg_data = agg_wsi(
                dataset,
                feature_key=f"features_{self.model}",
                tile_key="tiles_224",
                store_col="store",
                agg_key="agg_slide"
            )
        except Exception as e:
            raise RuntimeError(f"Aggregation failed: {e}")

        # ---- Join MIL scores ----
        self.agg_data.obs = self.agg_data.obs.join(
            self.results_df,
            on="Tissue Sample Id"
        )

        # ---- Make obs H5AD-safe ----
        for col in self.agg_data.obs.columns:
            if self.agg_data.obs[col].dtype == "object":
                try:
                    self.agg_data.obs[col] = self.agg_data.obs[col].astype(float)
                except Exception:
                    self.agg_data.obs[col] = self.agg_data.obs[col].astype(str)

        # ---- Save aggregated data ----
        out_file = f"agg_{self.model}_features.h5ad"
        self.agg_data.write_h5ad(out_file)
        print(f"Saved aggregated features to: {out_file}")

        return self.agg_data