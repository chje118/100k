import json
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

    def _single_slide_dataset(self, slide_path: str, label=None):
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

    def _build_slide_attention(self, slide_path: str, label=None):
        dataset = self._single_slide_dataset(slide_path, label)
        return SlideAttention(
            model=self.model,
            dataset=dataset,
            slide_idx=0, # TODO should be idx in list
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            filename_col="slide_path",
            zarr_dir=self.zarr_dir,
        )

    def _encode_label(self, label):
        if isinstance(label, (int, np.integer)):
            return int(label)
        if self.label_mapping and label in self.label_mapping:
            return int(self.label_mapping[label])
        return None

    def predict(self, slide_path: str, true_label=None):
        dataset = self._single_slide_dataset(slide_path, label=true_label)
        feats, _, _ = dataset[0]

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        true_idx = self._encode_label(true_label) if true_label is not None else None

        row = {
            "slide_path": slide_path,
            "pred_idx": pred_idx,
            "pred_label": self.idx_to_label.get(pred_idx, pred_idx),
            "confidence": float(probs[pred_idx]),
            "true_idx": true_idx,
            "true_label": true_label,
            "is_correct": None if true_idx is None else pred_idx == true_idx,
        }

        # Add probabilities for all classes
        row.update({f"prob_{self.idx_to_label.get(i, i)}": float(p) for i, p in enumerate(probs)})
        return row

    def attention_heatmap(self, slide_path: str, true_label=None):
        slide_attention = self._build_slide_attention(slide_path, label=true_label)

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

    def top_k_tiles(self, slide_path: str, k: int = 10, true_label=None):
        slide_attention = self._build_slide_attention(slide_path, label=true_label)
        top_df = slide_attention.select_top_tiles(n_tiles=k).copy()

        if "geometry" in top_df.columns:
            bounds = top_df["geometry"].apply(lambda g: g.bounds if g is not None else (np.nan, np.nan, np.nan, np.nan))
            bounds_df = pd.DataFrame(bounds.tolist(), columns=["minx", "miny", "maxx", "maxy"])
            top_df = pd.concat([top_df.reset_index(drop=True), bounds_df], axis=1)

        return top_df

    def run_predictions(self, true_labels: list | None = None, top_k: int = 10):
        if true_labels is not None and len(true_labels) != len(self.slides):
            raise ValueError("true_labels must have same length as slides")

        rows = []
        for idx, slide_path in enumerate(self.slides):
            true_label = None if true_labels is None else true_labels[idx]
            pred_row = self.predict(slide_path=slide_path, true_label=true_label)

            top_tiles_df = self.top_k_tiles(slide_path=slide_path, k=top_k, true_label=true_label)
            tile_cols = [c for c in ["tile_id", "attention", "minx", "miny", "maxx", "maxy"] if c in top_tiles_df.columns]
            top_tiles_df = top_tiles_df[tile_cols].reset_index(drop=True)

            top_tiles = []
            if not top_tiles_df.empty:
                for rank, tile_row in enumerate(top_tiles_df.to_dict(orient="records"), start=1):
                    top_tiles.append({"tile_rank": rank, **tile_row})

            rows.append({
                **pred_row,
                "n_top_tiles": len(top_tiles),
                "top_tiles": top_tiles,
            })

        results_df = pd.DataFrame(rows)
        slide_level_df = results_df.copy()

        accuracy = None
        valid = slide_level_df["is_correct"].notna() if "is_correct" in slide_level_df.columns else None
        if valid is not None and valid.any():
            accuracy = float(slide_level_df.loc[valid, "is_correct"].mean())

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
        }

    def save_results(self, output_path: str, results_df: pd.DataFrame, summary: dict | None = None):
        payload = {
            "summary": summary or {},
            "results": results_df.to_dict(orient="records"),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return payload

    @staticmethod
    def load_results(input_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["results_df"] = pd.DataFrame(payload.get("results", []))
        return payload

    def pipeline(self, output_path: str, true_labels: list | None = None, top_k: int = 10):
        # Run predictions in-memory and write a single structured payload to `output_path`.
        preds = self.run_predictions(true_labels=true_labels, top_k=top_k)
        results_df = preds["results_df"]

        self.save_results(output_path, results_df=results_df, summary=preds.get("summary"))

        # Return the single searchable results table and the saved path
        return {
            "results_df": results_df,
            "summary": preds.get("summary"),
            "accuracy": preds.get("accuracy"),
            "output_path": output_path,
        }


    def heatmap_by_index(self, index: int):
        """ Render attention heatmap for slide at `self.slides[index]`."""
        if index < 0 or index >= len(self.slides):
            raise IndexError("slide index out of range")
        slide_path = self.slides[index]
        return self.attention_heatmap(slide_path)

    def top_k_tiles_by_index(self, index: int, k: int = 5):
        """Return top-k tiles DataFrame for slide at `self.slides[index]`."""
        if index < 0 or index >= len(self.slides):
            raise IndexError("slide index out of range")
        slide_path = self.slides[index]
        return self.top_k_tiles(slide_path, k=k)

    def confusion_report(self, true_labels: list | None = None, display: bool = True):
        """Compute confusion matrix and classification report across `self.slides`.

        If `true_labels` is None, the method will attempt to use `None` (no report).
        Returns a dict with `cm`, `labels`, `report_str`, and `results_df`.
        """
        # Run predictions in-memory
        preds = self.run_predictions(true_labels=true_labels, top_k=0)
        results_df = preds["results_df"]
        slide_level_df = results_df

        if "true_idx" not in slide_level_df.columns or slide_level_df["true_idx"].isna().all():
            if display:
                print("No true labels available to compute confusion matrix.")
            return {"cm": None, "labels": None, "report_str": None, "results_df": results_df}

        # Extract valid rows
        valid = slide_level_df["true_idx"].notna()
        y_true = slide_level_df.loc[valid, "true_idx"].astype(int).tolist()
        y_pred = slide_level_df.loc[valid, "pred_idx"].astype(int).tolist()

        from sklearn.metrics import confusion_matrix, classification_report

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_str = classification_report(y_true, y_pred, labels=labels)

        if display:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap='Blues', colorbar=True)
            ax.set_title('Confusion Matrix')
            plt.show()
            print('\nClassification report:')
            print(report_str)

        return {"cm": cm, "labels": labels, "report_str": report_str, "results_df": results_df}
    

# FEATURES
def load_features(wsi):
    adata = wsi.tables[FEATURE_KEY]

    feats = torch.from_numpy(adata.X).float()
    tile_ids = np.array(adata.obs["tile_id"])

    return feats, tile_ids


# MODEL INFERENCE
def abmil_inference(model, feats, device):
    feats = feats.to(device)

    with torch.no_grad():
        logits, attention = model(feats)

    probs = torch.softmax(logits.squeeze().float(), dim=0).cpu().numpy()
    pred = int(np.argmax(probs))

    return logits, probs, attention.squeeze().cpu().numpy(), pred

# POSTPROCESSING
def get_top_k_tiles(wsi, tile_ids, attention, k=10):
    tile_table = wsi.tables[FEATURE_KEY].obs.copy()

    df = pd.DataFrame({
        "tile_id": tile_ids,
        "attention": attention
    })

    merged = df.merge(
        tile_table[["tile_id", "geometry"]],
        on="tile_id",
        how="inner"
    )

    if merged.empty:
        return merged

    return merged.sort_values("attention", ascending=False).head(k)


# PIPELINE ENTRYPOINT
def run_inference(wsi, checkpoint_path, slide_path, top_k=10):
    model, config, label_mapping, device = load_checkpoint(checkpoint_path)

    feats, tile_ids = load_features(wsi)

    if feats.shape[0] == 0:
        raise ValueError(f"No tiles found for slide: {slide_path}")

    logits, probs, attention, pred = abmil_inference(model, feats, device)

    top_tiles = get_top_k_tiles(
        wsi=wsi,
        tile_ids=tile_ids,
        attention=attention,
        k=top_k
    )

    return {
        "slide": slide_path,
        "prediction": pred,
        "probabilities": probs,
        "attention": attention,
        "tile_ids": tile_ids,
        "top_k_tiles": top_tiles,
    }