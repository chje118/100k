import os
import pickle
import matplotlib.pyplot as plt
import lazyslide as zs
import numpy as np
import pandas as pd
import torch
from wsidata import open_wsi
from abmil import ZarrSlideDataset, load_checkpoint, confusion_matrix_report, auc_score, per_class_auc, plot_roc_curve, per_class_pr_curves
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns


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
    
    def roc_curves_all(self):
        """ Plot all per-class ROC curves (one-vs-rest) on one figure with distinct colors. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        
        n_classes = self.y_probs.shape[1]
        y_true_onehot = np.eye(n_classes)[self.y_true]  # one-hot encoding
                
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("pastel", n_classes)
        
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, c], self.y_probs[:, c])
            roc_auc = roc_auc_score(y_true_onehot[:, c], self.y_probs[:, c])
            plt.plot(fpr, tpr, color=colors[c], linewidth=2, label=f"Class {c} (AUC={roc_auc:.4f})")
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-class ROC Curves (One-vs-Rest)")
        plt.legend(title="Classes", loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.show()



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