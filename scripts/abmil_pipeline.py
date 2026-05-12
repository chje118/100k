import os
import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
import torch
from wsidata import open_wsi
from abmil import ZarrSlideDataset, load_checkpoint

class DiseaseClassification:
    """Simple ABMIL workflow for one-slide inference, batch processing, and cache-based reporting."""

    def __init__(self, checkpoint_path: str, zarr_dir: str, slides: list[str], cache_path: str):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)
        self.cache_path = cache_path
        self._slide_cache = {}

        # Load cache
        if self.cache_path and os.path.exists(self.cache_path):
            loaded_cache = self.load_cache(self.cache_path)
            self._slide_cache = loaded_cache.get("slide_cache", {})

        # Load the trained model
        self.model, self.config, self.label_mapping = load_checkpoint(self.checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.feature_key = self.config.get("feature_key")
        self.tile_key = self.config.get("tile_key")
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()} if self.label_mapping else {}

    def _label_to_index(self, label):
        if isinstance(label, (int, np.integer)):
            return int(label)
        if label is not None and self.label_mapping and label in self.label_mapping:
            return int(self.label_mapping[label])
        return None

    def _single_slide_dataset(self, slide_path: str, label=None):
        """Build a one-row dataset for a single slide."""
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

    def _infer_slide(self, slide_path: str, true_label=None):
        """ Infer one slide once and cache the full result. """
        if slide_path in self._slide_cache:
            return self._slide_cache[slide_path]

        # Load features and tile information for one slide.
        dataset = self._single_slide_dataset(slide_path, label=true_label)
        feats, _, _ = dataset[0]

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device)
        with torch.no_grad():
            logits, attention = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        attention = attention.squeeze(1).detach().cpu().numpy()

        # Normalize attention scores to [0, 1]
        if attention.size and attention.max() > attention.min():
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(slide_path, zarr_path)
        wsi.tables[self.feature_key].obs["attention"] = attention

        # Build a tile table with attention scores and geometries
        attention_df = wsi.tables[self.feature_key].obs[["tile_id", "attention"]].copy()
        tile_df = wsi.shapes[self.tile_key][["tile_id", "geometry"]].copy()
        tile_table = pd.merge(attention_df, tile_df, on="tile_id", how="inner")

        slide_data = {
            "slide_path": slide_path,
            "zarr_path": zarr_path,
            "feature_key": self.feature_key,
            "tile_key": self.tile_key,
            "attention": attention,
            "tile_table": tile_table,
            "pred_idx": pred_idx,
            "pred_label": self.idx_to_label.get(pred_idx, pred_idx),
            "confidence": float(probs[pred_idx]),
            "true_idx": true_idx,
            "true_label": true_label,
            "is_correct": None if true_idx is None else pred_idx == true_idx,
        }
        slide_data.update({f"prob_{self.idx_to_label.get(i, i)}": float(prob) for i, prob in enumerate(probs)})

        self._slide_cache[slide_path] = slide_data
        self.save_cache()
        return slide_data

    def save_cache(self):
        """ Save the full slide cache to a pickle file. """
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._slide_cache, f)
        return self._slide_cache

    @staticmethod
    def load_cache(input_path: str):
        """ Load a pickle cache file. """
        with open(input_path, "rb") as f:
            return pickle.load(f)

    def process_slides(self, true_labels: list | None = None, top_k: int = 10):
        """Batch process slides, save cache after each slide, and return a success message."""
        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        for idx, slide_path in enumerate(self.slides):
            true_label = None if true_labels is None else true_labels[idx]
            row = self.infer_slide(slide_path, true_label=true_label)
            top_tiles_df = self.top_k_tiles(slide_path, k=top_k)
            top_tiles = top_tiles_df.to_dict(orient="records") if not top_tiles_df.empty else []
            rows.append({**row, "n_top_tiles": len(top_tiles), "top_tiles": top_tiles})

        results_df = pd.DataFrame(rows)
        valid = results_df["true_idx"].notna() if "true_idx" in results_df.columns else None
        accuracy = float(results_df.loc[valid, "is_correct"].mean()) if valid is not None and valid.any() else None

        results_df["run_n_slides"] = len(self.slides)
        results_df["run_n_labeled_slides"] = int(valid.sum()) if valid is not None else np.nan
        results_df["run_accuracy"] = accuracy

        return {
            "message": "Batch processing complete.",
            "results_df": results_df,
            "summary": {
                "n_slides": len(self.slides),
                "n_labeled_slides": int(valid.sum()) if valid is not None else 0,
                "accuracy": accuracy,
            },
            "accuracy": accuracy,
        }

    def top_k_tiles(self, slide_path: str, k: int = 10):
        """Return the top-k tiles from cache for later use."""
        tile_table = self._get_cached_slide(slide_path)["tile_table"]
        if tile_table.empty:
            return pd.DataFrame()
        return tile_table.sort_values("attention", ascending=False).head(k).copy()

    def attention_heatmap(self, slide_path: str):
        """Plot the attention heatmap for one cached slide."""
        slide_data = self._get_cached_slide(slide_path)
        wsi = open_wsi(slide_path, slide_data["zarr_path"])
        wsi.tables[slide_data["feature_key"]].obs["attention"] = slide_data["attention"]

        fig, ax = plt.subplots(figsize=(10, 10))
        zs.pl.tiles(
            wsi,
            tile_key=slide_data["tile_key"],
            feature_key=slide_data["feature_key"],
            color="attention",
            cmap="hot",
            alpha=0.8,
            show_contours=False,
            ax=ax,
        )
        ax.set_title(f"Attention heatmap: {os.path.basename(slide_path)}")
        ax.axis("off")
        plt.show()
        return fig

    def zoom_view(self, slide_path: str, k: int = 5, margin: int = 0):
        """Show a zoomed view of the top-k tiles on the slide."""
        slide_data = self._get_cached_slide(slide_path)
        tile_table = slide_data["tile_table"]
        wsi = open_wsi(slide_path, slide_data["zarr_path"])

        if tile_table.empty:
            print("No tiles available for zoom view.")
            return None

        top_tiles = tile_table.sort_values("attention", ascending=False).head(k).copy()
        cols = 2
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.asarray(axes).flatten()

        for index, (_, tile_row) in enumerate(top_tiles.iterrows()):
            geometry = tile_row["geometry"]
            if not hasattr(geometry, "bounds"):
                axes[index].axis("off")
                continue

            minx, miny, maxx, maxy = geometry.bounds
            xmin = minx - margin
            ymin = miny - margin
            xmax = maxx + margin
            ymax = maxy + margin

            zs.pl.tiles(wsi, tile_key=slide_data["tile_key"], zoom=(xmin, xmax, ymin, ymax), ax=axes[index])
            axes[index].set_title(f"tile {tile_row.get('tile_id', index)}")

        for index in range(len(top_tiles), len(axes)):
            axes[index].axis("off")

        plt.tight_layout()
        plt.show()
        return fig

    def assessment_report(self, true_labels: list | None = None, display: bool = True, output_path: str | None = None):
        """Compute confusion matrix and classification report from cached slide predictions."""
        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        for idx, slide_path in enumerate(self.slides):
            cached = self._get_cached_slide(slide_path)
            row = dict(cached["row"])
            if true_labels is not None:
                row["true_label"] = true_labels[idx]
                row["true_idx"] = self._label_to_index(true_labels[idx])
                row["is_correct"] = None if row["true_idx"] is None else row["pred_idx"] == row["true_idx"]
            rows.append(row)

        results_df = pd.DataFrame(rows)
        if results_df.empty or "true_idx" not in results_df.columns or results_df["true_idx"].isna().all():
            if display:
                print("No true labels available to compute confusion matrix.")
            report = {"cm": None, "labels": None, "report_str": None, "results_df": results_df}
            if output_path is not None:
                self.save_assessment(output_path, report)
            return report

        valid = results_df["true_idx"].notna()
        y_true = results_df.loc[valid, "true_idx"].astype(int).tolist()
        y_pred = results_df.loc[valid, "pred_idx"].astype(int).tolist()

        from sklearn.metrics import classification_report, confusion_matrix

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_str = classification_report(y_true, y_pred, labels=labels)

        if display:
            from sklearn.metrics import ConfusionMatrixDisplay

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap="Blues", colorbar=True)
            ax.set_title("Confusion Matrix")
            plt.show()
            print("\nClassification report:")
            print(report_str)

        report = {"cm": cm, "labels": labels, "report_str": report_str, "results_df": results_df}
        if output_path is not None:
            self.save_assessment(output_path, report)
        return report

    def save_assessment(self, output_path: str, report: dict):
        """Save assessment results to a pickle file."""
        payload = {
            "cm": report.get("cm"),
            "labels": report.get("labels"),
            "report_str": report.get("report_str"),
            "results_df": report.get("results_df", pd.DataFrame()),
        }
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)
        return payload

