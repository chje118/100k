import os

import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
import torch

from abmil import ZarrSlideDataset, load_checkpoint
from abmil_tile_selection import SlideAttention


class DiseaseClassification:
    def __init__(self, checkpoint_path: str, zarr_dir: str, slides: list[str]):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)

        self.model, self.config, self.label_mapping = load_checkpoint(self.checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.feature_key = self.config.get("feature_key")
        self.tile_key = self.config.get("tile_key")
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()} if self.label_mapping else {}

    def _single_slide_dataset(self, slide_path: str, label: int = 0):
        df = pd.DataFrame({"slide_path": [slide_path], "label": [label]})
        return ZarrSlideDataset(
            df=df,
            filename_col="slide_path",
            label_col="label",
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
            max_tiles=None,
            seed=None,
        )

    def _build_slide_attention(self, slide_path: str) -> SlideAttention:
        dataset = self._single_slide_dataset(slide_path, label=0)
        return SlideAttention(
            model=self.model,
            dataset=dataset,
            slide_idx=0,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            filename_col="slide_path",
            zarr_dir=self.zarr_dir,
        )

    def _encode_label(self, label):
        if label is None:
            return None
        if isinstance(label, (int, np.integer)):
            return int(label)
        if self.label_mapping and label in self.label_mapping:
            return int(self.label_mapping[label])
        return None

    def predict(self, slide_path: str, true_label=None):
        dataset = self._single_slide_dataset(slide_path, label=0)
        feats, _, _ = dataset[0]

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        true_idx = self._encode_label(true_label)

        row = {
            "slide_path": slide_path,
            "pred_idx": pred_idx,
            "pred_label": self.idx_to_label.get(pred_idx, pred_idx),
            "confidence": float(probs[pred_idx]),
            "true_label": true_label,
            "true_idx": true_idx,
            "is_correct": None if true_idx is None else int(pred_idx == true_idx),
        }
        row.update({f"prob_{self.idx_to_label.get(i, i)}": float(p) for i, p in enumerate(probs)})
        return row

    def attention_heatmap(self, slide_path: str):
        slide_attention = self._build_slide_attention(slide_path)

        fig, ax = plt.subplots(figsize=(10, 10))
        zs.pl.tiles(
            slide_attention.wsi,
            tile_key=self.tile_key,
            feature_key=self.feature_key,
            color="attention",
            cmap="hot",
            alpha=0.8,
            show_contours=False,
            ax=ax,
        )
        ax.set_title(f"Attention heatmap: {os.path.basename(slide_path)}")
        ax.axis("off")
        plt.show()
        plt.close(fig)
        return None

    def top_k_tiles(self, slide_path: str, k: int = 5):
        slide_attention = self._build_slide_attention(slide_path)
        top_df = slide_attention.select_top_tiles(n_tiles=k).copy()

        if "geometry" in top_df.columns:
            bounds = top_df["geometry"].apply(lambda g: g.bounds if g is not None else (np.nan, np.nan, np.nan, np.nan))
            bounds_df = pd.DataFrame(bounds.tolist(), columns=["minx", "miny", "maxx", "maxy"])
            top_df = pd.concat([top_df.reset_index(drop=True), bounds_df], axis=1)

        return top_df

    def pipeline(self, true_labels: list | None = None, top_k: int = 10, output_dir: str = "abmil_inference_outputs"):
        os.makedirs(output_dir, exist_ok=True)

        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        combined_rows = []
        for idx, slide_path in enumerate(self.slides):
            true_label = None if true_labels is None else true_labels[idx]
            pred_row = self.predict(slide_path=slide_path, true_label=true_label)
            rows.append(pred_row)

            top_tiles_df = self.top_k_tiles(slide_path=slide_path, k=top_k)
            tile_cols = [c for c in ["tile_id", "attention", "minx", "miny", "maxx", "maxy"] if c in top_tiles_df.columns]
            top_tiles_df = top_tiles_df[tile_cols].reset_index(drop=True)

            if top_tiles_df.empty:
                combined_rows.append({
                    **pred_row,
                    "tile_rank": np.nan,
                    "tile_id": np.nan,
                    "attention": np.nan,
                    "minx": np.nan,
                    "miny": np.nan,
                    "maxx": np.nan,
                    "maxy": np.nan,
                })
            else:
                for rank, tile_row in enumerate(top_tiles_df.to_dict(orient="records"), start=1):
                    combined_rows.append({
                        **pred_row,
                        "tile_rank": rank,
                        **tile_row,
                    })

        results_df = pd.DataFrame(rows)
        combined_df = pd.DataFrame(combined_rows)
        pred_csv = os.path.join(output_dir, "pipeline_results.csv")
        combined_df.to_csv(pred_csv, index=False)

        accuracy = None
        valid = results_df["is_correct"].notna() if "is_correct" in results_df.columns else None
        if valid is not None and valid.any():
            accuracy = float(results_df.loc[valid, "is_correct"].mean())

        return {
            "results_df": results_df,
            "combined_df": combined_df,
            "accuracy": accuracy,
            "pipeline_csv": pred_csv,
            "output_dir": output_dir,
        }