import os
import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
import torch
from wsidata import open_wsi
from abmil import ZarrSlideDataset, load_checkpoint, confusion_matrix_report, auc_score, per_class_auc, plot_roc_curve, per_class_pr_curves


class ABMILInference:
    """ ABMIL inference: predicts slide-level labels from tile features. """
    def __init__(self, checkpoint_path: str, zarr_dir: str, slides: list[str], cache_path: str):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)
        self.cache_path = cache_path
        self._slide_cache = {}
        self._skipped_slides = []

        # Load cache
        if self.cache_path and os.path.exists(self.cache_path):
            loaded_cache = self.load_cache(self.cache_path)
            if isinstance(loaded_cache, dict) and "slide_cache" in loaded_cache:
                self._slide_cache = loaded_cache.get("slide_cache", {})
            elif isinstance(loaded_cache, dict):
                self._slide_cache = loaded_cache

        # Load the trained model
        self.model, self.config, self.label_mapping = load_checkpoint(self.checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.feature_key = self.config.get("feature_key")
        self.tile_key = self.config.get("tile_key")
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()} if self.label_mapping else {}

    def _single_slide_dataset(self, slide_path: str):
        """ Build a one-row dataset for a single slide. """
        df = pd.DataFrame({"slide_path": [slide_path], "label": [None]})
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

    def _infer_slide(self, slide_path: str):
        """ Infer one slide once and cache the full result. """
        if slide_path in self._slide_cache:
            return self._slide_cache[slide_path]

        # Load features and tile information for one slide.
        dataset = self._single_slide_dataset(slide_path)
        feats, _, _ = dataset[0]

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device)
        with torch.no_grad():
            logits, attention = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        attention = attention.squeeze(1).detach().cpu().numpy()

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

    def process_slides(self):
        """ Batch process slides, save cache after each slide. """
        self._skipped_slides = []
        processed_count = 0
        for slide_path in self.slides:
            try:
                self._infer_slide(slide_path)
                processed_count += 1
            except Exception as e:
                self._skipped_slides.append({
                    "slide_path": slide_path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
                print(f"Skipping slide {slide_path}: {type(e).__name__}: {e}")
        print(
            f"Processed {processed_count}/{len(self.slides)} slides. Cache saved to {self.cache_path}."
        )
        if self._skipped_slides:
            print(f"Skipped {len(self._skipped_slides)} slides due to errors.")

    def attention_heatmap(self, slide_path: str):
        """ Plot the attention heatmap for one cached slide. """
        slide_data = self._slide_cache.get(slide_path)
        if not slide_data:
            print(f"No cached data found for slide: {slide_path}")
            return None
        
        wsi = open_wsi(slide_path, slide_data["zarr_path"])
        attention = np.asarray(slide_data["attention"], dtype=float)
        if attention.size:
            low, high = np.percentile(attention, [5, 99])
            if high > low:
                attention_display = np.clip((attention - low) / (high - low + 1e-8), 0.0, 1.0)
            else:
                attention_display = attention.copy()
        else:
            attention_display = attention

        print(
            f"Attention stats for {os.path.basename(slide_path)}: "
            f"min={attention.min():.4f}, p5={np.percentile(attention, 5):.4f}, "
            f"median={np.median(attention):.4f}, p99={np.percentile(attention, 99):.4f}, "
            f"max={attention.max():.4f}"
        )

        wsi.tables[slide_data["feature_key"]].obs["attention_display"] = attention_display

        fig, ax = plt.subplots(figsize=(10, 10))
        zs.pl.tiles(
            wsi,
            tile_key=slide_data["tile_key"],
            feature_key=slide_data["feature_key"],
            color="attention_display",
            cmap="hot",
            vmin=0,
            vmax=1,
            show_contours=True,
            ax=ax,
        )
        ax.set_title(f"Attention heatmap: {os.path.basename(slide_path)}, Predicted: {slide_data['pred_label']} (Conf: {slide_data['confidence']:.2f})")
        ax.axis("off")
        plt.show()

    def results_dataframe(self):
        """ Get a DataFrame of cached results for all processed slides. """
        if not self._slide_cache:
            print("No cached results found. Run process_slides() first.")
            return pd.DataFrame()
        
        rows = []
        for slide_path, cached in self._slide_cache.items():
            rows.append({
                "slide_path": slide_path,
                "pred_idx": cached.get("pred_idx"),
                "pred_label": cached.get("pred_label"),
                "confidence": cached.get("confidence"),
                **{col: cached.get(col) for col in cached if col.startswith("prob_")},
            })
        return pd.DataFrame(rows)

class ABMILEvaluation:
    """ Evaluates ABMIL inference results by matching predictions with true labels via filenames."""
    def __init__(self, results_df: pd.DataFrame, metadata_df: pd.DataFrame, true_label_col: str):
        self.results_df = results_df.copy()
        self.metadata_df = metadata_df.copy()
        self.true_label_col = true_label_col
        
        self.y_true = None
        self.y_pred = None
        self.y_probs = None
        self.matched_df = None
    
    def _extract_slide_id(self, slide_path: str) -> str:
        """ Extract slide identifier from slide_path (basename without extension). """
        return os.path.basename(slide_path).replace(".mrxs", "")
    
    def match_true_labels(self, slide_id_col: str = "filename", results_path_col: str = "slide_path"):
        """ Match predicted results with true labels from metadata. """
        # Extract slide IDs from results paths
        self.results_df["_slide_id_results"] = self.results_df[results_path_col].apply(self._extract_slide_id)
        
        # Extract slide IDs from metadata paths
        self.metadata_df["_slide_id_metadata"] = self.metadata_df[slide_id_col].apply(self._extract_slide_id)
        
        # Merge results with metadata on slide_id
        self.matched_df = pd.merge(
            self.results_df,
            self.metadata_df[["_slide_id_metadata", self.true_label_col]],
            left_on="_slide_id_results",
            right_on="_slide_id_metadata",
            how="inner"
        )
        
        if self.matched_df.empty:
            raise ValueError("No matches found between results and metadata.")
        
        # Clean up temporary columns
        self.matched_df = self.matched_df.drop(columns=["_slide_id_results", "_slide_id_metadata"])
        
        # Extract labels and probabilities
        self.y_true = self.matched_df[self.true_label_col].values
        self.y_pred = self.matched_df["pred_label"].values
        
        # Extract probability columns (all columns starting with "prob_")
        prob_cols = sorted([col for col in self.matched_df.columns if col.startswith("prob_")])
        if prob_cols:
            self.y_probs = self.matched_df[prob_cols].values
        else:
            # Fallback: create one-hot encoded probs from predictions
            unique_labels = sorted(set(self.y_pred) | set(self.y_true))
            self.y_probs = np.zeros((len(self.matched_df), len(unique_labels)))
            for i, label in enumerate(self.y_pred):
                label_idx = unique_labels.index(label)
                self.y_probs[i, label_idx] = 1.0
        
        print(
            f"Matched {len(self.matched_df)} slides. "
            f"y_true shape: {self.y_true.shape}, y_pred shape: {self.y_pred.shape}, "
            f"y_probs shape: {self.y_probs.shape}"
        )
        return self.matched_df
    
    def assessment_report(self):
        """ Compute confusion matrix and classification report from matched results. """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")

        confusion_matrix_report(self.y_true, self.y_pred)
        return {"results_df": self.results_df}

    def compute_metrics(self):
        """ Compute AUC and per-class AUC scores. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        
        auc = auc_score(self.y_true, self.y_probs)
        per_class_aucs = per_class_auc(self.y_true, self.y_probs)
        return {"auc": auc, "per_class_aucs": per_class_aucs}

    def group_by_metrics(self, group_col: str):
        """Compute metrics grouped by a metadata column."""
        if self.matched_df is None or self.matched_df.empty:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        if group_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{group_col}' not found in metadata_df.")

        grouped_df = self.matched_df.copy()
        grouped_df["_slide_id"] = grouped_df["slide_path"].apply(self._extract_slide_id)
        metadata = self.metadata_df[[group_col]].copy()
        metadata["_slide_id"] = self.metadata_df["filename"].apply(self._extract_slide_id)
        grouped_df = grouped_df.merge(metadata, on="_slide_id", how="left")

        prob_cols = [col for col in grouped_df.columns if col.startswith("prob_")]
        rows = []
        for group_val, group_data in grouped_df.groupby(group_col, dropna=False):
            probs = group_data[prob_cols].to_numpy() if prob_cols else None
            rows.append({
                group_col: group_val,
                "n_samples": len(group_data),
                "accuracy": np.mean(group_data[self.true_label_col].values == group_data["pred_label"].values),
                "auc": auc_score(group_data[self.true_label_col].values, probs) if probs is not None and probs.shape[1] > 1 else np.nan,
            })

        group_metrics_df = pd.DataFrame(rows)
        print(f"\nPer-group metrics grouped by '{group_col}':")
        print(group_metrics_df.to_string(index=False))
        return group_metrics_df
    
    def roc_curve(self):
        """ Plot ROC curves. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        
        plot_roc_curve(self.y_true, self.y_probs)

    def pr_curves(self):
        """ Plot precision-recall curves. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        
        return per_class_pr_curves(self.y_true, self.y_probs)
    
    # TODO method to plot all roc curves ovr as the pr curves (same plot, different colors)



if __name__ == "__main__":
    """Example usage for running this module directly.

    Update the file paths and column names below to match your data.
    """
    # Example: inference on a batch of slides
    # inference = ABMILInference(
    #     checkpoint_path="/path/to/checkpoint.pt",
    #     zarr_dir="/path/to/zarr_cache",
    #     slides=[
    #         "/path/to/slide_001.mrxs",
    #         "/path/to/slide_002.mrxs",
    #     ],
    #     cache_path="/path/to/abmil_cache.pkl",
    # )
    # inference.process_slides()
    # results_df = inference.results_dataframe()

    # Example: evaluation against metadata
    # metadata_df = pd.read_csv("/path/to/metadata.csv")
    # evaluator = ABMILEvaluation(results_df, metadata_df, true_label_col="diagnosis")
    # evaluator.match_true_labels(slide_id_col="filename", results_path_col="slide_path")
    # evaluator.assessment_report()
    # metrics = evaluator.compute_metrics()
    # print(metrics)