"""
ABMIL evaluation utilities: AUC + ROC plot.
Designed to compare ABMIL runs trained on features from different foundation models.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

try:
    # when repository root is on PYTHONPATH
    from notebooks.abmil import ABMIL, ZarrSlideDataset 
except Exception:
    # when running from inside `notebooks/` directory
    from abmil import ABMIL, ZarrSlideDataset


# For each ABMIL checkpoint we want to evaluate, define an ABMILRun with the necessary info to load the model and dataset.
@dataclass(frozen=True)
class ABMILRun:
    name: str # name of run
    model_path: str # path to saved model .pth file
    feature_key: str 
    tile_key: str = "tiles_224"
    hidden_dim: int = 256


def _parse_dims_from_model_path(model_path: str) -> Tuple[int, int, int]:
    """
    Parse `..._<in_dim>_<n_classes>_<hidden_dim>.pth` from `save_model` naming.
    """
    base = os.path.basename(model_path)
    parsed = re.search(r"_(\d+)_(\d+)_(\d+)\.pth$", base)
    if not parsed:
        raise ValueError(
            f"Could not parse dims from model filename '{base}'. "
            "Expected suffix like '_<in_dim>_<n_classes>_<hidden_dim>.pth'."
        )
    return int(parsed.group(1)), int(parsed.group(2)), int(parsed.group(3))


def load_abmil_from_checkpoint(model_path: str) -> Tuple[ABMIL, Dict[str, int]]:
    in_dim, n_classes, hidden_dim = _parse_dims_from_model_path(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = ABMIL(in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim).to(device)
    
    # Load state dict
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    
    # Set to eval mode (no dropout, etc.)
    model.eval()

    return model, {"in_dim": in_dim, "n_classes": n_classes, "hidden_dim": hidden_dim}


@torch.no_grad() # disable gradient computation (faster, less memory)
def predict_proba_abmil(
    model: ABMIL,
    dataset: ZarrSlideDataset,
    device: Optional[str] = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - y_true: shape [n_slides]
    - y_proba: shape [n_slides, n_classes]
    """
    if device is None:
        device = next(model.parameters()).device

    y_true: List[int] = []
    y_proba: List[np.ndarray] = []

    for i in range(len(dataset)):
        feats, tile_ids, label = dataset[i]
        if feats.dim() == 3:
            feats = feats.squeeze(0)
        if feats.shape[0] == 0:
            continue

        feats = feats.to(device)
        # normalize per slide (same as training/validation)
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

        if (isinstance(device, str) and str(device).startswith("cuda") and use_amp and torch.cuda.is_available()):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(feats)
        else:
            logits, _ = model(feats)

        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        y_true.append(int(label))
        y_proba.append(probs)

    return np.asarray(y_true), np.asarray(y_proba)


def auc_summary(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC AUC.
    - Binary: returns auc_roc (class 1 vs rest)
    - Multiclass: macro OVR AUC
    """
    n_classes = y_proba.shape[1]
    if n_classes == 2:
        auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        return {"auc_roc": auc, "n_classes": 2}

    auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return {"auc_roc_ovr_macro": auc, "n_classes": int(n_classes)}


def plot_roc_curves(
    runs_results: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    ax: Optional[plt.Axes] = None,
    title: str = "ABMIL ROC curves",
):
    """
    runs_results: list of (run_name, y_true, y_proba)
    Binary ROC only (for now). For multiclass, use summary AUCs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    for name, y_true, y_proba in runs_results:
        if y_proba.shape[1] != 2:
            continue
        RocCurveDisplay.from_predictions(y_true, y_proba[:, 1], name=name, ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return fig, ax


def evaluate_runs(
    runs: Sequence[ABMILRun],
    df: pd.DataFrame,
    filename_col: str,
    label_col: str,
    zarr_dir: str,
    device: Optional[str] = None,
    use_amp: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple ABMIL checkpoints on the same slide list (df).

    Returns a DataFrame with one row per run.
    """
    rows = []
    roc_payload = []

    for run in runs:
        model, dims = load_abmil_from_checkpoint(run.model_path, device=device)

        dataset = ZarrSlideDataset(
            df=df,
            filename_col=filename_col,
            label_col=label_col,
            feature_key=run.feature_key,
            tile_key=run.tile_key,
            zarr_dir=zarr_dir,
        )

        y_true, y_proba = predict_proba_abmil(model, dataset, device=device, use_amp=use_amp)
        summ = auc_summary(y_true, y_proba)

        row = {"run": run.name, "model_path": run.model_path, "feature_key": run.feature_key, **dims, **summ}
        rows.append(row)
        roc_payload.append((run.name, y_true, y_proba))

    out = pd.DataFrame(rows)

    # Optional ROC plot (binary only)
    if any(p[2].shape[1] == 2 for p in roc_payload):
        plot_roc_curves(roc_payload, title="ABMIL ROC (binary)")
        plt.show()

    return out


if __name__ == "__main__":
    # Example usage:
    # - Prepare a dataframe `df_val` with columns: filename_col, label_col
    # - Define one ABMILRun per foundation-model feature set you trained on
    #
    # NOTE: model_path filenames must match save_model() pattern:
    #   <name>_<in_dim>_<n_classes>_<hidden_dim>.pth
    #
    df_val = pd.DataFrame()  # fill in
    filename_col = "filename"
    label_col = "label"
    zarr_dir = "path/to/zarr_dir"
    runs = [
        # ABMILRun(name="conch", model_path="models/abmil_conch_512_2_256.pth", feature_key="features_conch"),
        # ABMILRun(name="uni",   model_path="models/abmil_uni_1024_2_256.pth", feature_key="features_uni"),
    ]

    if len(df_val) and runs:
        summary = evaluate_runs(
            runs=runs,
            df=df_val,
            filename_col=filename_col,
            label_col=label_col,
            zarr_dir=zarr_dir,
        )
        print(summary)

