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
    """ABMIL-based disease classification pipeline for WSI slides.
    
    Handles single-slide and batch inference workflows with prediction,
    tile selection, visualization, and comprehensive reporting.
    """
    def __init__(self, checkpoint_path: str, zarr_dir: str, slides: list[str], cache_path: str):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)
        self.cache_path = cache_path
        self._slide_cache = {}

        # Load an existing cache file if exists
        if os.path.exists(self.cache_path):
            loaded_cache = self.load_cache(self.cache_path)
            self._slide_cache = loaded_cache.get("slide_cache", {})

        # Load model, config, and label mappings from checkpoint
        self.model, self.config, self.label_mapping = load_checkpoint(self.checkpoint_path)
        self.device = next(self.model.parameters()).device
        
        # Get feature and tile keys from config (used to access data in Zarr files)
        self.feature_key = self.config.get("feature_key")
        self.tile_key = self.config.get("tile_key")
        
        # Create mapping: prediction index -> label name
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()} if self.label_mapping else {}

    def save_cache(self):
        """ Save the current slide cache to a pickle file. """
        payload = {
            "slide_cache": self._slide_cache,
            "n_slides_cached": len(self._slide_cache), # TODO: remove this, not needed, but maybe add timestamp
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(payload, f)
        return payload

    @staticmethod
    def load_cache(input_path: str):
        """ Load a pickle cache file and restore slide data structures. """
        with open(input_path, "rb") as f:
            return pickle.load(f)

    def _get_cached_slide(self, slide_path: str):
        """ Return cached slide data or raise if inference has not been run yet. """
        cache_key = slide_path
        if cache_key not in self._slide_cache:
            raise RuntimeError("No cached inference found for this slide. Run inference first. ")
        return self._slide_cache[cache_key]

    def _infer_slide(self, slide_path: str, true_label=None):
        """ Run one forward pass and cache prediction, attention, and WSI data. """
        cache_key = slide_path
        if cache_key in self._slide_cache:
            return self._slide_cache[cache_key]

        df = pd.DataFrame({"slide_path": [slide_path], "label": [true_label]})
        dataset = ZarrSlideDataset(
            df=df,
            filename_col="slide_path",
            label_col="label",
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
            max_tiles=None,
            seed=None,
        )
        feats, _, _ = dataset[0]

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device)
        with torch.no_grad():
            logits, attention = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        if isinstance(true_label, (int, np.integer)):
            true_idx = int(true_label)
        elif true_label is not None and self.label_mapping and true_label in self.label_mapping:
            true_idx = int(self.label_mapping[true_label])
        else:
            true_idx = None

        attention = attention.squeeze(1).detach().cpu().numpy()

        # Normalize attention weights to [0, 1] for better visualization
        if attention.size and attention.max() > attention.min():
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(slide_path, zarr_path)
        wsi.tables[self.feature_key].obs["attention"] = attention
        attention_df = wsi.tables[self.feature_key].obs[["tile_id", "attention"]].copy()
        tile_df = wsi.shapes[self.tile_key][["tile_id", "geometry"]].copy()
        tile_table = pd.merge(attention_df, tile_df, on="tile_id", how="inner")

        slide_data = {
            "slide_path": slide_path,
            "zarr_path": zarr_path,
            "pred_idx": pred_idx,
            "pred_label": self.idx_to_label.get(pred_idx, str(pred_idx)),
            "probs": probs,
            "true_idx": true_idx,
            "true_label": self.idx_to_label.get(true_idx, str(true_idx)) if true_idx is not None else None,
            "attention": attention,
            "tile_table": tile_table,
        }
        self._slide_cache[cache_key] = slide_data

        # Save the cache after every slide so the file stays up to date.
        self.save_cache(self.cache_path)

        return slide_data

    def infer_slide(self, slide_path: str, true_label=None):
        """Infer one slide, cache the raw result, and return the prediction row."""
        slide_data = self._infer_slide(slide_path, true_label)

        # Build the row only when the caller asks for the slide prediction.
        return self._build_row(slide_path, slide_data, true_label)

    def predict(self, slide_path: str, true_label=None):
        """ Run model inference on a single slide. Only returns the prediction row. """
        return self.infer_slide(slide_path, true_label)

    def attention_heatmap(self, slide_path: str, true_label=None):
        """ Visualize tile attention weights as a heatmap on the slide. """
        # Reuse the cached attention from the same forward pass used for prediction.
        slide_data = self._get_cached_slide(slide_path)
        zarr_path = slide_data["zarr_path"]
        wsi = open_wsi(slide_path, zarr_path)
        wsi.tables[self.feature_key].obs["attention"] = slide_data["attention"]

        fig, ax = plt.subplots(figsize=(10, 10))
        zs.pl.tiles(
            wsi,
            tile_key=self.tile_key,
            feature_key=self.feature_key,
            color="attention",  # Color by attention weights
            cmap="hot",  # Hot colormap: cold (low attention) -> hot (high attention)
            alpha=0.8,
            show_contours=False,
            ax=ax,
        )
        ax.set_title(f"Attention heatmap: {os.path.basename(slide_path)}")
        ax.axis("off")
        plt.show()

    def top_k_tiles(self, slide_path: str, k: int = 10, true_label=None):
        """ Extract top-k tiles by attention weight from a slide. """
        # Reuse the cached merged tile table from the single forward pass.
        tile_table = self._get_cached_slide(slide_path)["tile_table"]

        if tile_table.empty:
            return pd.DataFrame()

        return tile_table.sort_values("attention", ascending=False).head(k).copy()

    def zoom_view(self, slide_path: str, k: int = 5, margin: int = 0):
        """Show a zoomed view of the top-k tiles on the slide."""
        # Read the cached tiles only so the zoom view never recomputes inference.
        top_tiles = self.top_k_tiles(slide_path=slide_path, k=k)
        slide_data = self._get_cached_slide(slide_path)
        wsi = open_wsi(slide_path, slide_data["zarr_path"])

        if top_tiles.empty:
            print("No tiles available for zoom view.")
            return None

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

            zs.pl.tiles(wsi, tile_key=self.tile_key, zoom=(xmin, xmax, ymin, ymax), ax=axes[index])
            axes[index].set_title(f"tile {tile_row.get('tile_id', index)}")

        for index in range(len(top_tiles), len(axes)):
            axes[index].axis("off")

        plt.tight_layout()
        plt.show()
        return fig

    def run_predictions(self, true_labels: list | None = None, top_k: int = 10):
        """Run batch predictions on all slides in self.slides.
        
        Args:
            true_labels: Optional list of true labels (must match len(self.slides))
            top_k: Number of top tiles to extract per slide
            
        Returns:
            Dict with results_df, summary, and overall accuracy
        """
        # Validate input
        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        
        # Process each slide
        for idx, slide_path in enumerate(self.slides):
            true_label = None if true_labels is None else true_labels[idx]
            # Read cached slide data only; batch mode must not recompute inference.
            pred_row = self.infer_slide(slide_path, true_label=true_label)
            slide_data = self._get_cached_slide(slide_path)
            top_tiles_df = slide_data["tile_table"].sort_values("attention", ascending=False).head(top_k).copy()
            top_tiles = top_tiles_df.to_dict(orient="records") if not top_tiles_df.empty else []

            # Combine prediction with top tiles
            rows.append({
                **pred_row,
                "n_top_tiles": len(top_tiles),
                "top_tiles": top_tiles,
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(rows)
        slide_level_df = results_df.copy()

        # Compute overall accuracy if labels are provided
        accuracy = None
        valid = slide_level_df["is_correct"].notna() if "is_correct" in slide_level_df.columns else None
        if valid is not None and valid.any():
            accuracy = float(slide_level_df.loc[valid, "is_correct"].mean())

        # Add run-level summary columns to each row
        results_df["run_n_slides"] = len(self.slides)
        results_df["run_n_labeled_slides"] = int(valid.sum()) if valid is not None else np.nan
        results_df["run_accuracy"] = accuracy

        return {
            "results_df": results_df,
            "summary": {
                "n_slides": len(self.slides),
                "n_labeled_slides": int(valid.sum()) if valid is not None else 0,
                "accuracy": accuracy,
            },
            "accuracy": accuracy,
            "message": "Batch processing complete.",
        }

    def save_results(self, output_path: str, results_df: pd.DataFrame, summary: dict | None = None):
        """Save prediction results and summary to a pickle file."""
        payload = {
            "summary": summary or {},
            "results_df": results_df,
        }

        with open(output_path, "wb") as f:
            pickle.dump(payload, f)

        return payload

    def save_assessment(self, output_path: str, report: dict):
        """Save confusion-matrix assessment output to a pickle file."""
        payload = {
            "cm": report.get("cm"),
            "labels": report.get("labels"),
            "report_str": report.get("report_str"),
            "results_df": report.get("results_df", pd.DataFrame()),
        }

        with open(output_path, "wb") as f:
            pickle.dump(payload, f)

        return payload

    @staticmethod
    def load_results(input_path: str):
        """Load saved prediction results from a pickle file.
        
        Returns:
            Dict with 'summary' dict and 'results_df' DataFrame
        """
        with open(input_path, "rb") as f:
            return pickle.load(f)

    def pipeline(self, output_path: str, true_labels: list | None = None, top_k: int = 10):
        """Run full batch prediction pipeline: predict, extract top tiles, and save results.
        
        Returns:
            Dict with results_df, summary, accuracy, and output_path
        """
        # Run predictions on all slides
        preds = self.run_predictions(true_labels=true_labels, top_k=top_k)
        results_df = preds["results_df"]

        # Save results to JSON file
        self.save_results(output_path, results_df=results_df, summary=preds.get("summary"))

        # Return results and metadata
        return {
            "results_df": results_df,
            "summary": preds.get("summary"),
            "accuracy": preds.get("accuracy"),
            "output_path": output_path,
        }

    def assessment_report(self, true_labels: list | None = None, display: bool = True, output_path: str | None = None):
        """Compute the overall assessment using cached slide predictions only."""
        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        for idx, slide_path in enumerate(self.slides):
            true_label = None if true_labels is None else true_labels[idx]
            slide_data = self._get_cached_slide(slide_path)
            rows.append(self._build_row(slide_path, slide_data, true_label))

        results_df = pd.DataFrame(rows)

        if "true_idx" not in results_df.columns or results_df["true_idx"].isna().all():
            if display:
                print("No true labels available to compute confusion matrix.")
            report = {"cm": None, "labels": None, "report_str": None, "results_df": results_df}
            if output_path is not None:
                self.save_assessment(output_path, report)
            return report

        valid = results_df["true_idx"].notna()
        y_true = results_df.loc[valid, "true_idx"].astype(int).tolist()
        y_pred = results_df.loc[valid, "pred_idx"].astype(int).tolist()

        from sklearn.metrics import confusion_matrix, classification_report

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_str = classification_report(y_true, y_pred, labels=labels)

        if display:
            import matplotlib.pyplot as plt
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

    def confusion_report(self, true_labels: list | None = None, display: bool = True):
        """Backwards-compatible alias for the cached assessment report."""
        return self.assessment_report(true_labels=true_labels, display=display)

    def full_report(self, output_path: str, true_labels: list | None = None, top_k: int = 10, display: bool = True):
        """One-call workflow: predict all, save results, compute metrics, and visualize.
        
        Returns:
            Dict with results, accuracy, confusion matrix, and classification report
        """
        # Step 1: Predict all slides and save results to disk
        preds = self.pipeline(output_path, true_labels=true_labels, top_k=top_k)
        
        # Step 2: Compute confusion matrix and classification report from the cache.
        confusion = self.assessment_report(true_labels=true_labels, display=display)
        
        # Step 3: Display attention heatmap for first slide (if labels provided)
        if true_labels and display:
            try:
                self.attention_heatmap(self.slides[0])
            except Exception as e:
                print(f"Could not display attention heatmap: {e}")
        
        # Return comprehensive results
        return {
            "results_df": preds["results_df"],
            "summary": preds["summary"],
            "accuracy": preds["accuracy"],
            "output_path": preds["output_path"],
            "confusion_matrix": confusion["cm"],
            "confusion_labels": confusion["labels"],
            "classification_report": confusion["report_str"],
        }

    def infer_single_slide(self, slide_path: str, true_label=None, top_k: int = 10):
        """Single-slide inference: predict and extract top-k tiles.
        
        Args:
            slide_path: Path to WSI slide
            true_label: Optional true label for accuracy comparison
            top_k: Number of top tiles to extract
            
        Returns:
            Dict with prediction, confidence, probabilities, and top tiles
        """
        # Run one cached inference pass and reuse it for prediction plus tile selection.
        pred_row = self.infer_slide(slide_path, true_label)
        top_tiles_df = self.top_k_tiles(slide_path=slide_path, k=top_k, true_label=true_label)
        top_tiles = top_tiles_df.to_dict(orient="records") if not top_tiles_df.empty else []

        # Combine prediction and tiles into single result
        result = {
            **pred_row,
            "n_top_tiles": len(top_tiles),
            "top_tiles": top_tiles,
        }

        return result