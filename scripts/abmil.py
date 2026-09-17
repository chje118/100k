import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
import lazyslide as zs
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import glob
import copy
import random

# ----------------------------------------
# Dataset definition
# ----------------------------------------

class ZarrSlideDataset(Dataset):
    """PyTorch Dataset for loading WSI features from Zarr files."""

    def __init__(self, df, filename_col, label_col, feature_key, zarr_dir, max_tiles=None, seed=None, require_labels=True):
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.zarr_dir = zarr_dir
        self.max_tiles = max_tiles  # Maximum number of tiles per slide (None = no limit)
        self.seed = seed    # Seed for deterministic tile sampling (if max_tiles is set)
        self.require_labels = require_labels # labels required for training/evaluation; if False, label can be None (e.g., inference)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        slide_path = row[self.filename_col]
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))

        wsi = open_wsi(slide_path, zarr_path)
        adata = wsi.tables[self.feature_key]

        feats = torch.from_numpy(adata.X).float() # tile features as a PyTorch tensor
        tile_ids = np.array(adata.obs['tile_id']) # save tile IDs for visualization

        # Apply max_tiles limit with deterministic, uniform sampling (equal probability)
        if self.max_tiles is not None and feats.shape[0] > self.max_tiles:
            if self.seed is not None:
                local_rng = np.random.RandomState(self.seed + idx)
            else:
                local_rng = np.random.RandomState(idx)

            indices = local_rng.choice(feats.shape[0], self.max_tiles, replace=False)
            feats = feats[indices]
            tile_ids = tile_ids[indices]

        # Handle missing labels (inference uses label=None)
        if self.require_labels:
            label_val = row[self.label_col]
            if pd.isna(label_val):
                raise ValueError(f"Missing label for slide: {slide_path}")
            label = torch.tensor(int(label_val), dtype=torch.long)
        else:
            label = None

        return feats, tile_ids, label

# ----------------------------------------
# Model definition
# ----------------------------------------

class ABMIL(nn.Module):
    """
    Single-head gated ABMIL for binary classification:
        Class 0 = healthy-ish
        Class 1 = non-healthy

    This uses the standard gated attention mechanism from Ilse et al. (2018):
        a_i ∝ exp(w^T (tanh(V x_i) ⊙ sigmoid(U x_i)))
    and MIL pooling:
        z = Σ_i a_i x_i
    followed by a linear classifier on the slide embedding z.

    Keeps n_heads=1 for interpretability: one attention weight per tile.
    Tile-level class contributions are derived from attention and classifier weights.
    """
    def __init__(self, in_dim, n_classes=2, hidden_dim=256, n_heads=1):
        super().__init__()

        if n_classes != 2:
            raise ValueError(f"ABMIL is currently restricted to binary classification (n_classes=2), got {n_classes}.")

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Attention mechanism, producing one attention score per tile
        # Gated attention: A = V * U (tanh * sigmoid)
        # Tanh allows positive and negative responses
        self.attn_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        # Sigmoid acts as a learned gate between 0 and 1.
        self.attn_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        # Maps hidden representation to raw attention scores
        self.attn_w = nn.Linear(hidden_dim, n_heads) 

        # Classifier layer, maps final slide embedding to class scores
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        """
        Forward pass of ABMIL.
        Parameters:
        - x: tile features for one slide, shape [n_tiles, feat_dim]
        """
        # Compute attention scores: A = V * U (gated attention)
        V = self.attn_V(x)      # [n_tiles, hidden_dim] (tanh)
        U = self.attn_U(x)      # [n_tiles, hidden_dim] (sigmoid)
        H = V * U               # elementwise gating

        # Raw attention logits
        attention_logits = self.attn_w(H)      # [n_tiles, n_heads]

        # Normalize attention within each head so weights sum to 1 over tiles
        attention = torch.softmax(attention_logits, dim=0)  # [n_tiles, n_heads]

        # MIL pooling: one pooled representation per head
        # pooled_heads[h, :] = Σ_i attention[i, h] * x[i, :]
        pooled_heads = torch.einsum("nh,nd->hd", attention, x)  # [n_heads, in_dim]

        # Aggregate heads by averaging (simple, symmetric aggregation)
        pooled = pooled_heads.mean(dim=0)  # [in_dim]; with n_heads=1, this is just pooled[0]

        # Slide-level class logits
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "attention_logits": attention_logits,
            "attention": attention,
            "pooled_heads": pooled_heads,
            "pooled": pooled,
        }

@torch.no_grad()
def get_tile_contributions(model: ABMIL, feats: torch.Tensor):
    """
    Compute tile-level scores for DVP ROI selection.

    Returns:
    1. attention: "How much did this tile influence the slide embedding?"
       Unsigned salience only.

    2. contrast_score: "Does this tile's own feature vector read as disease-like or healthy-like?"
       Independent of attention.

    3. contribution_score = attention * contrast_score:
       Exact per-tile contribution to the class contrast.
    """
    model.eval()

    out = model(feats)
    logits = out["logits"]
    attention = out["attention"]

    # n_heads=1 by design 
    attention = attention.mean(dim=1)

    # Raw, attention-independent linear read of each tile's own features:
    tile_class_scores = feats @ model.classifier.weight.T

    healthy_score = tile_class_scores[:, 0]       # evidence for healthy-ish
    disease_score = tile_class_scores[:, 1]       # evidence for non-healthy

    # Binary-only model: class 0 = healthy-ish, class 1 = non-healthy
    contrast_score = disease_score - healthy_score
    contribution_score = attention * contrast_score
    probabilities = torch.softmax(logits, dim=0)

    return {
        "logits": logits.cpu(),
        "probabilities": probabilities.cpu(),
        "attention": attention.cpu(),
        "healthy_score": healthy_score.cpu(),
        "disease_score": disease_score.cpu(),
        "contrast_score": contrast_score.cpu(),
        "contribution_score": contribution_score.cpu(),
    }

# ----------------------------------------
# Helper functions
# ----------------------------------------

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_dataset(dataset):
    """Filter out invalid slides from dataset."""
    valid_indices = []
    for i in tqdm(range(len(dataset)), desc="Validating slides"):
        try:
            _ = dataset[i]
            valid_indices.append(i)
        except Exception as e:
            print(f"Invalid slide at index {i}: {type(e).__name__}: {str(e)}")

    filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)
    print(f"Dataset validation complete: {len(valid_indices)}/{len(dataset)} valid slides")
    return filtered_dataset, valid_indices

def require_cuda():
    """Raise an error if CUDA is not available and return the CUDA device string."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ABMIL pipeline, but no CUDA device is available.")
    return "cuda"

def _configure_gpu_optimization():
    """ Enable optional CUDA performance settings for training/inference on supported NVIDIA GPUs."""
    if not torch.cuda.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

def get_amp_dtype():
    """
    Choose the mixed-precision dtype for CUDA.

    For NVIDIA H100, bfloat16 is the best default for training stability and
    throughput. It is also a good fit for ABMIL because attention/softmax over
    many tiles benefits from bf16's wider numeric range compared with fp16.
    """
    device = require_cuda()
    major, _ = torch.cuda.get_device_capability(device=device)
    if major >= 8:
        return torch.bfloat16
    return torch.float16

def autocast_context():
    """Return the correct CUDA autocast context for the current GPU."""
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=get_amp_dtype())

def save_checkpoint(model, config, label_mapping, path):
    """Save model checkpoint with configuration and label mapping."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "label_mapping": label_mapping,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path):
    """Load a checkpoint with model, config, and label mapping."""
    device = require_cuda()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    label_mapping = checkpoint["label_mapping"]

    model = ABMIL(
        in_dim=config["in_dim"],
        n_classes=config["n_classes"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from {path}")
    print(
        f"Model config: in_dim={config['in_dim']}, "
        f"n_classes={config['n_classes']}, "
        f"hidden_dim={config['hidden_dim']}, "
        f"n_heads={config['n_heads']}"
    )
    print(f"Label mapping: {label_mapping}")
    return model, config, label_mapping

def create_label_mapping(df, label_col):
    """Map class labels to integer indices."""
    return {label: i for i, label in enumerate(sorted(df[label_col].unique()))}

def _group_stratified_train_val_split(df, label_col, group_col, test_size, random_state=None):
    """Group-aware, approximately stratified train/val split."""
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf.split(df, y=df[label_col], groups=df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df

# ----------------------------------------
# Training and validation functions
# ----------------------------------------

def train_ABMIL(train_df, train_dataset, val_dataset=None, label_col=None, n_epochs=10, 
    early_stopping_patience=None, seed=None):
    """Train ABMIL model with optional early stopping."""
    if seed is not None:
        set_seed(seed)

    device = require_cuda()

    # DataLoader: handles shuffling and batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=(lambda worker_id: set_seed(seed + worker_id) if seed is not None else None),
    )

    # Extract feature dimension and number of classes
    sample_feats, _, _ = train_dataset[0]
    feat_dim = sample_feats.shape[1]
    n_classes = train_df[label_col].nunique()
    if n_classes != 2:
        raise ValueError(
            f"This training pipeline currently supports binary classification only, but found {n_classes} classes."
        )

    model = ABMIL(feat_dim, n_classes=2).to(device)

    _configure_gpu_optimization()
    amp_dtype = get_amp_dtype()
    print(f"Using CUDA mixed precision dtype: {amp_dtype}")

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Early stopping setup
    best_auc = -1.0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0

        for feats, tile_ids, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)

            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue

            optimizer.zero_grad() # clear gradients

            # Forward pass with CUDA mixed precision.
            # On H100 this will use bfloat16, which is the preferred balance of
            # speed and numerical stability for this ABMIL setup.
            with autocast_context():
                out = model(feats)
                logits = out["logits"]
                loss = loss_fn(logits.unsqueeze(0), label)

            loss.backward() # backpropagation
            optimizer.step() # update weights

            total_loss += loss.item() # accumulate loss

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss:.4f}", end="")
        # TODO: save training loss history for plotting

        # Early stopping based on validation AUC
        if val_dataset is not None and early_stopping_patience is not None:
            all_labels, _, all_probs = validate_ABMIL(model, val_dataset)
            val_auc = auc_score(all_labels, all_probs)
            print(f" | Val AUC: {val_auc:.4f}", end="")

            # Check if validation AUC improved
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_no_improve = 0
                print(" (improved)", end="")
            else:
                epochs_no_improve += 1
                print(f" (no improve: {epochs_no_improve}/{early_stopping_patience})", end="")

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        print()

    if val_dataset is not None and early_stopping_patience is not None and best_epoch == 0:
        best_epoch = n_epochs
    elif val_dataset is None or early_stopping_patience is None:
        best_epoch = n_epochs

    return model, best_model_state, best_epoch


def validate_ABMIL(model, val_dataset):
    device = require_cuda()

    # DataLoader for validation (no shuffling, batch_size=1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval() # evaluation mode disables dropout, batchnorm updates, etc.

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False):
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)

            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue

            # Forward pass with mixed precision for speed and memory efficiency
            with autocast_context():
                logits = model(feats)["logits"]

            # Compute predicted class and probabilities
            probs = torch.softmax(logits.float(), dim=0).cpu().numpy()
            pred = torch.argmax(logits, dim=0).item()

            all_labels.append(label.item())
            all_preds.append(pred)
            all_probs.append(probs)

    return all_labels, all_preds, np.array(all_probs)


# ----------------------------------------
# ABMIL evaluation methods
# ----------------------------------------

def plot_confusion_matrix(all_labels, all_preds, class_names=None):
    """Compute and plot confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return cm

def get_classification_report(all_labels, all_preds, class_names=None):
    """Return and print classification report."""
    if class_names is None:
        report = classification_report(all_labels, all_preds)
    else:
        report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    return report

def auc_score(all_labels, all_probs):
    """Compute ROC AUC for binary classification."""
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    if all_probs.shape[1] != 2:
        raise ValueError(
            f"auc_score currently supports binary classification only. Found {all_probs.shape[1]} classes."
        )

    try:
        return roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        return np.nan

def plot_roc_curve(all_labels, all_probs):
    """Plot ROC curve for binary classification."""
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    if all_probs.shape[1] != 2:
        raise ValueError(
            f"plot_roc_curve only supports binary classification. Found {all_probs.shape[1]} classes."
        )

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary Classification)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_pr_curve(all_labels, all_probs):
    """Plot precision-recall curve for binary classification."""
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    if all_probs.shape[1] != 2:
        raise ValueError(
            f"pr_curve_binary only supports binary classification. Found {all_probs.shape[1]} classes."
        )

    p, r, _ = precision_recall_curve(all_labels, all_probs[:, 1])
    ap = average_precision_score(all_labels, all_probs[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(r, p, linewidth=2, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Binary Classification)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# Training, inference and evaluation pipelines
# ----------------------------------------

class TrainABMILPipeline:
    """
    Train ABMIL with validation, then retrain on all valid slides for the
    best epoch count and save checkpoint.
    """
    def __init__(self, df, filename_col, label_col, patient_col, feature_key, tile_key, zarr_dir, save_path):
        self.df = df.copy()
        self.filename_col = filename_col
        self.label_col = label_col
        self.patient_col = patient_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.save_path = save_path
        self.label_mapping = create_label_mapping(self.df, self.label_col)
        self.device = require_cuda()

        self.model = None
        self.best_epoch = None

        print(f"TrainABMILPipeline initialized with {len(self.df)} slides on device: {self.device}")

    def _make_dataset(self, df, max_tiles=None, seed=None):
        return ZarrSlideDataset(
            df=df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            zarr_dir=self.zarr_dir,
            max_tiles=max_tiles,
            seed=seed,
        )

    def _map_labels(self, df):
        mapped = df.copy()
        mapped[self.label_col] = mapped[self.label_col].map(self.label_mapping)
        if mapped[self.label_col].isna().any():
            missing = mapped.loc[mapped[self.label_col].isna(), self.label_col].unique().tolist()
            raise ValueError(f"Unmapped labels found: {missing}")
        mapped[self.label_col] = mapped[self.label_col].astype(int)
        return mapped

    def validate_slides(self):
        """Filter out slides that cannot be loaded."""
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")

        temp_dataset = self._make_dataset(self.df)
        _, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        return self.df

    def train_abmil(self, max_tiles=50000, n_epochs=100, seed=42, validation_fraction=0.10, early_stopping_patience=5):
        """Fit on a train/validation split and record the best epoch."""
        df = self._map_labels(self.df)

        n_classes = df[self.label_col].nunique()
        if n_classes != 2:
            raise ValueError(
                f"TrainABMILPipeline currently supports binary classification only, but found {n_classes} classes."
            )

        # Group-aware, stratified train/validation split
        train_df, val_df = _group_stratified_train_val_split(
            df,
            label_col=self.label_col,
            group_col=self.patient_col,
            test_size=validation_fraction,
            random_state=seed,
        )

        train_dataset = self._make_dataset(train_df, max_tiles=max_tiles, seed=seed)
        val_dataset = self._make_dataset(val_df, max_tiles=max_tiles, seed=seed)

        self.model, _, self.best_epoch = train_ABMIL(
            train_df=train_df,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_col=self.label_col,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
        )

        return self.model, self.best_epoch

    def save_abmil(self, max_tiles=50000, seed=42, n_epochs=None):
        """Retrain on all valid slides for best_epoch epochs and save the checkpoint."""
        if self.best_epoch is None and n_epochs is None:
            raise ValueError("best_epoch is not set. Run train_abmil() first.")

        df = self._map_labels(self.df)
        full_dataset = self._make_dataset(df, max_tiles=max_tiles, seed=seed)
        epochs = self.best_epoch if n_epochs is None else n_epochs

        self.model, _, _ = train_ABMIL(
            train_df=df,
            train_dataset=full_dataset,
            val_dataset=None,
            label_col=self.label_col,
            n_epochs=epochs,
            early_stopping_patience=None,
            seed=seed,
        )

        self.config = {
            "in_dim": self.model.in_dim,
            "n_classes": self.model.n_classes,
            "hidden_dim": self.model.hidden_dim,
            "n_heads": self.model.n_heads,
            "feature_key": self.feature_key,
            "tile_key": self.tile_key,
            "max_tiles": max_tiles,
            "n_epochs": epochs,
            "seed": seed,
        }

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        save_checkpoint(self.model, self.config, self.label_mapping, self.save_path)

        return self.model, self.label_mapping, self.save_path

    def run_pipeline(self, max_tiles=50000, n_epochs=100, seed=42, validation_fraction=0.10, early_stopping_patience=5):
        self.validate_slides()
        self.train_abmil(
            max_tiles=max_tiles,
            n_epochs=n_epochs,
            seed=seed,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
        )
        return self.save_abmil(max_tiles=max_tiles, seed=seed)


class KFoldPipeline:
    """K-fold cross-validation pipeline for binary ABMIL model evaluation."""
    def __init__(self, df, filename_col, label_col, patient_col, feature_key, tile_key, zarr_dir):
        self.df = df.copy()
        self.filename_col = filename_col
        self.label_col = label_col
        self.patient_col = patient_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.device = require_cuda()
        self.results = None

        print(f"KFoldPipeline initialized on device: {self.device}")

    def _make_dataset(self, df, max_tiles=None, seed=None):
        return ZarrSlideDataset(
            df=df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            zarr_dir=self.zarr_dir,
            max_tiles=max_tiles,
            seed=seed,
        )

    def _map_labels(self, df):
        mapped = df.copy()
        label_mapping = create_label_mapping(self.df, self.label_col)
        mapped[self.label_col] = mapped[self.label_col].map(label_mapping)
        if mapped[self.label_col].isna().any():
            missing = mapped.loc[mapped[self.label_col].isna(), self.label_col].unique().tolist()
            raise ValueError(f"Unmapped labels found: {missing}")
        mapped[self.label_col] = mapped[self.label_col].astype(int)
        return mapped, label_mapping

    def _evaluate_fold(self, model, eval_dataset):
        all_labels, all_preds, all_probs = validate_ABMIL(model=model, val_dataset=eval_dataset)
        fold_auc = auc_score(all_labels, all_probs)
        fold_accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
        return all_labels, all_preds, all_probs, fold_auc, fold_accuracy

    def validate_slides(self):
        """Filter out invalid slides causing errors during loading."""
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")

        temp_dataset = self._make_dataset(self.df)
        _, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        return self.df

    def kfold_cross_validation(
        self,
        n_splits=5,
        n_epochs=10,
        early_stopping_patience=5,
        max_tiles=None,
        random_state=42,
        resume_from_checkpoints=False,
        checkpoint_dir="checkpoints/"
    ):
        """
        Run stratified group k-fold cross-validation with internal validation.
        """
        set_seed(random_state)
        print(f"Random seed set to {random_state} for reproducibility")

        df, label_mapping = self._map_labels(self.df)

        if df[self.label_col].nunique() != 2:
            raise ValueError(
                f"KFoldPipeline currently supports binary classification only, but found {df[self.label_col].nunique()} classes."
            )

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_auc_scores = []
        fold_accuracies = []
        fold_all_labels = []
        fold_all_preds = []
        fold_all_probs = []

        print(f"Starting {n_splits}-fold cross-validation with early stopping (patience={early_stopping_patience})...")
        print("Train/Test split is patient-grouped and label-stratified.")

        start_fold = 1
        checkpoint_by_fold = {}

        if resume_from_checkpoints:
            pattern = os.path.join(checkpoint_dir, "fold_*_auc_*.pt")
            checkpoint_files = glob.glob(pattern)
            fold_regex = re.compile(r"fold_(\d+)_auc_.*\.pt$")

            for checkpoint_file in checkpoint_files:
                checkpoint_name = os.path.basename(checkpoint_file)
                match = fold_regex.match(checkpoint_name)
                if not match:
                    continue
                fold_num = int(match.group(1))
                prev_path = checkpoint_by_fold.get(fold_num)
                if prev_path is None or os.path.getmtime(checkpoint_file) > os.path.getmtime(prev_path):
                    checkpoint_by_fold[fold_num] = checkpoint_file

            while start_fold in checkpoint_by_fold and start_fold <= n_splits:
                start_fold += 1

            print(f"Resume mode enabled. Found checkpoints for folds: {sorted(checkpoint_by_fold.keys())}")
            print(f"Will train from fold {start_fold}/{n_splits}")

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, y=df[self.label_col], groups=df[self.patient_col])):
            fold_num = fold_idx + 1
            print(f"\n{'='*60}")
            print(f"Fold {fold_num}/{n_splits}")
            print(f"{'='*60}")

            # Split fold into 80% train + 20% test
            fold_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            split_seed = random_state + fold_num
            train_subset_df, internal_val_df = _group_stratified_train_val_split(
                fold_df,
                label_col=self.label_col,
                group_col=self.patient_col,
                test_size=1/9, # 90% train, 10% internal val
                random_state=split_seed,
            )

            print(f"Train subset: {len(train_subset_df)} samples")
            print(f"Internal val: {len(internal_val_df)} samples")
            print(f"Test set: {len(test_df)} samples")

            # Use fold-specific seed for deterministic tile sampling
            fold_seed = random_state + fold_num
        
            # Create datasets with deterministic tile sampling
            train_dataset = self._make_dataset(train_subset_df, max_tiles=max_tiles, seed=fold_seed)
            internal_val_dataset = self._make_dataset(internal_val_df, max_tiles=max_tiles, seed=fold_seed)
            test_dataset = self._make_dataset(test_df, max_tiles=max_tiles, seed=fold_seed)

            if resume_from_checkpoints and fold_num < start_fold:
                checkpoint_path = checkpoint_by_fold.get(fold_num)
                if checkpoint_path is None:
                    raise FileNotFoundError(f"Missing checkpoint for fold {fold_num} in {checkpoint_dir}.")

                print(f"Using existing checkpoint for fold {fold_num}: {checkpoint_path}")
                model, _, _ = load_checkpoint(checkpoint_path)

                all_labels, all_preds, all_probs, fold_auc, fold_accuracy = self._evaluate_fold(model, test_dataset)

                fold_auc_scores.append(fold_auc)
                fold_accuracies.append(fold_accuracy)
                fold_all_labels.append(list(all_labels))
                fold_all_preds.append(list(all_preds))
                fold_all_probs.append(all_probs.tolist())

                print(f"Fold {fold_num} (checkpoint) - Test AUC: {fold_auc:.4f}, Test Accuracy: {fold_accuracy:.4f}")
                continue

            # Train model on train_subset with early stopping on internal_val
            # Use fold-specific seed derived from random_state for reproducibility
            model, _, _ = train_ABMIL(
                train_df=train_subset_df,
                train_dataset=train_dataset,
                val_dataset=internal_val_dataset,
                label_col=self.label_col,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
                seed=fold_seed
            )

            # Validate on test set for final evaluation of this fold
            all_labels, all_preds, all_probs, fold_auc, fold_accuracy = self._evaluate_fold(model, test_dataset)

            fold_auc_scores.append(fold_auc)
            fold_accuracies.append(fold_accuracy)
            fold_all_labels.append(list(all_labels))
            fold_all_preds.append(list(all_preds))
            fold_all_probs.append(all_probs.tolist())

            print(f"Fold {fold_num} - Test AUC: {fold_auc:.4f}, Test Accuracy: {fold_accuracy:.4f}")

            # Save checkpoint for this fold
            config = {
                "in_dim": model.in_dim,
                "hidden_dim": model.hidden_dim,
                "n_classes": model.n_classes,
                "n_heads": model.n_heads,
                "feature_key": self.feature_key,
                "tile_key": self.tile_key,
                "max_tiles": max_tiles,
                "n_epochs": n_epochs,
                "random_state": fold_seed,
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_num}_auc_{fold_auc:.4f}.pt")
            save_checkpoint(model, config, label_mapping, checkpoint_path)

        # Compute mean and std across folds
        results_dict = {
            "fold_auc_scores": fold_auc_scores,
            "mean_auc": float(np.mean(fold_auc_scores)) if fold_auc_scores else np.nan,
            "std_auc": float(np.std(fold_auc_scores)) if fold_auc_scores else np.nan,
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else np.nan,
            "std_accuracy": float(np.std(fold_accuracies)) if fold_accuracies else np.nan,
            "fold_all_labels": fold_all_labels,
            "fold_all_preds": fold_all_preds,
            "fold_all_probs": fold_all_probs,
            "n_splits": n_splits
        }

        self.results = results_dict
        return self.results

    def print_results(self):
        """Print cross-validation result summary."""
        if self.results is None:
            print("No results available. Run the pipeline first with .kfold_cross_validation()")
            return

        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Results ({self.results['n_splits']} folds)")
        print(f"{'='*60}")
        print(f"Mean AUC: {self.results['mean_auc']:.4f} ± {self.results['std_auc']:.4f}")
        print(f"Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"\nPer-fold AUC: {[f'{auc:.4f}' for auc in self.results['fold_auc_scores']]}")
        print(f"Per-fold Accuracy: {[f'{acc:.4f}' for acc in self.results['fold_accuracies']]}")


class ABMILInference:
    """Run slide-level ABMIL inference from cached tile features and store attention outputs."""
    def __init__(self, checkpoint_path, zarr_dir, slides, cache_path=None, heatmap_dir=None, save_heatmap=False):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)
        self.cache_path = cache_path
        self.heatmap_dir = heatmap_dir or os.path.join(os.path.dirname(cache_path) if cache_path else ".", "heatmaps")
        self.save_heatmap = save_heatmap
        self._slide_cache = {}
        self._skipped_slides = []

        # Load cached inference results if available
        if self.cache_path and os.path.exists(self.cache_path):
            loaded_cache = self.load_cache(self.cache_path)
            if isinstance(loaded_cache, dict):
                self._slide_cache = loaded_cache

        # Load the trained model
        self.model, self.config, self.label_mapping = load_checkpoint(self.checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.feature_key = self.config.get("feature_key")
        self.tile_key = self.config.get("tile_key")
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()} if self.label_mapping else {}

    def save_cache(self):
        """Save cached inference results to disk."""
        if not self.cache_path:
            return self._slide_cache

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp_path = f"{self.cache_path}.tmp"

        with open(tmp_path, "wb") as f:
            pickle.dump(self._slide_cache, f)

        os.replace(tmp_path, self.cache_path)
        return self._slide_cache

    @staticmethod
    def load_cache(input_path):
        """Load a pickle cache file."""
        with open(input_path, "rb") as f:
            return pickle.load(f)

    def _infer_slide(self, slide_path: str):
        """Infer one slide and cache result."""
        if slide_path in self._slide_cache:
            return self._slide_cache[slide_path]

        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(slide_path, zarr_path)

        if self.feature_key not in wsi.tables:
            raise KeyError(f"Feature key '{self.feature_key}' not found in zarr tables")
        if self.tile_key not in wsi.shapes:
            raise KeyError(f"Tile key '{self.tile_key}' not found in zarr shapes")

        adata = wsi.tables[self.feature_key]
        feats = torch.from_numpy(adata.X).float()
        tile_ids = np.asarray(adata.obs["tile_id"])

        if feats.shape[0] == 0:
            raise ValueError(f"No tiles found for slide: {slide_path}")

        feats = feats.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast_context():
                contrib = get_tile_contributions(self.model, feats)

        probs = contrib["probabilities"].numpy()
        pred_idx = int(np.argmax(probs))

        attention = contrib["attention"].numpy()               # salience only
        healthy_score = contrib["healthy_score"].numpy()
        disease_score = contrib["disease_score"].numpy()
        contrast_score = contrib["contrast_score"].numpy()          # + disease-like, - healthy-like (raw, unweighted - healthy/control arm)
        contribution_score = contrib["contribution_score"].numpy()  # attention * contrast_score (exact logit decomposition - disease arm)

        scores_df = pd.DataFrame({
            "tile_id": tile_ids,
            "attention": attention,
            "healthy_score": healthy_score,
            "disease_score": disease_score,
            "contrast_score": contrast_score,
            "contribution_score": contribution_score,
        })

        tile_df = wsi.shapes[self.tile_key][["tile_id", "geometry"]].copy()
        tile_table = pd.merge(scores_df, tile_df, on="tile_id", how="inner")

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
        return slide_data

    def process_slides(self):
        """
        Process slides sequentially, skipping failures and continuing.
        Optionally save one JPG heatmap per slide.
        """
        processed_count = 0

        for slide_path in tqdm(self.slides, desc="Running ABMIL inference..."):
            try:
                self._infer_slide(slide_path)

                if self.save_heatmap:
                    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                    heatmap_path = os.path.join(self.heatmap_dir, f"{slide_name}_heatmap.jpg")
                    self.attention_heatmap(slide_path, save_path=heatmap_path)

                self.save_cache()
                processed_count += 1

            except Exception as e:
                self._skipped_slides.append({
                    "slide_path": slide_path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
                print(f"Skipping slide {slide_path}: {type(e).__name__}: {e}")

        print(f"Processed {processed_count}/{len(self.slides)} slides.")
        if self.cache_path:
            print(f"Cache saved to {self.cache_path}.")
        if self.save_heatmap:
            print(f"Heatmaps saved to {self.heatmap_dir}.")
        if self._skipped_slides:
            print(f"Skipped {len(self._skipped_slides)} slides due to errors.")

        return self._slide_cache

    def attention_heatmap(self, slide_path: str, save_path: str = None):
        """Plot the attention heatmap for one cached slide."""
        slide_data = self._slide_cache.get(slide_path)
        if not slide_data:
            print(f"No cached data found for slide: {slide_path}")
            return None

        wsi = open_wsi(slide_path, slide_data["zarr_path"])
        attention = np.asarray(slide_data["attention"], dtype=float)

        if attention.ndim > 1:
            attention = attention.mean(axis=1)

        if attention.size:
            low, high = np.percentile(attention, [5, 99])
            if high > low:
                # Scale attention to [0, 1] for visualization, clipping extreme values
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
        ax.set_title(
            f"Attention heatmap: {os.path.basename(slide_path)}, "
            f"Predicted: {slide_data['pred_label']} "
            f"(Conf: {slide_data['confidence']:.2f})"
        )
        ax.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path

        plt.show()
        return fig

    def contrast_heatmap(self, slide_path: str, save_path: str = None):
        """
        Plot the contrast_score heatmap for one cached slide.
        """
        slide_data = self._slide_cache.get(slide_path)
        if not slide_data:
            print(f"No cached data found for slide: {slide_path}")
            return None

        tile_table = slide_data.get("tile_table")
        if tile_table is None or "contrast_score" not in tile_table.columns:
            print(f"No contrast_score found for slide: {slide_path}. Re-run inference to populate it.")
            return None

        wsi = open_wsi(slide_path, slide_data["zarr_path"])
        contrast = tile_table.set_index("tile_id")["contrast_score"]

        # Center the colormap at 0
        limit = float(np.abs(contrast).max()) if len(contrast) else 1.0
        limit = limit if limit > 0 else 1.0

        adata = wsi.tables[slide_data["feature_key"]]
        adata.obs["contrast_display"] = contrast.reindex(adata.obs["tile_id"]).to_numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        zs.pl.tiles(
            wsi,
            tile_key=slide_data["tile_key"],
            feature_key=slide_data["feature_key"],
            color="contrast_display",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            show_contours=True,
            ax=ax,
        )
        ax.set_title(
            f"Contrast score (blue=healthy-like, red=disease-like): {os.path.basename(slide_path)}, "
            f"Predicted: {slide_data['pred_label']} (Conf: {slide_data['confidence']:.2f})"
        )
        ax.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path

        plt.show()
        return fig

    def results_dataframe(self):
        """Return one row per cached slide with predictions and class probabilities."""
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

    def get_skipped_slides(self):
        """Return skipped slides as a DataFrame."""
        return pd.DataFrame(self._skipped_slides)


class ABMILEvaluation:
    """Evaluate slide-level ABMIL inference results against metadata labels."""

    def __init__(self, results_df: pd.DataFrame, metadata_df: pd.DataFrame, true_label_col: str):
        self.results_df = results_df.copy()
        self.metadata_df = metadata_df.copy()
        self.true_label_col = true_label_col

        self.y_true = None
        self.y_pred = None
        self.y_probs = None
        self.matched_df = None
        self.label_to_idx = None
        self.idx_to_label = None

    @staticmethod
    def _extract_slide_id(slide_path):
        """Extract slide identifier from a path or filename."""
        return os.path.basename(str(slide_path)).replace(".mrxs", "")

    def match_true_labels(self, slide_id_col="filename", results_path_col="slide_path"):
        """Match predictions with ground-truth labels using slide identifiers."""
        if results_path_col not in self.results_df.columns:
            raise ValueError(f"Column '{results_path_col}' not found in results_df.")
        if slide_id_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{slide_id_col}' not found in metadata_df.")
        if self.true_label_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{self.true_label_col}' not found in metadata_df.")

        results_df = self.results_df.copy()
        metadata_df = self.metadata_df.copy()

        # Extract slide identifiers for matching
        results_df["_slide_id_results"] = results_df[results_path_col].apply(self._extract_slide_id)
        metadata_df["_slide_id_metadata"] = metadata_df[slide_id_col].apply(self._extract_slide_id)

        # Merge results with metadata on slide identifiers
        matched_df = pd.merge(
            results_df,
            metadata_df[["_slide_id_metadata", self.true_label_col]],
            left_on="_slide_id_results",
            right_on="_slide_id_metadata",
            how="inner"
        )

        if matched_df.empty:
            raise ValueError("No matches found between results and metadata.")

        # Drop temporary slide identifier columns after matching
        matched_df = matched_df.drop(columns=["_slide_id_results", "_slide_id_metadata"])
        self.matched_df = matched_df

        # Prepare true and predicted labels for evaluation
        pred_labels = matched_df["pred_label"].astype(str).values
        true_labels = matched_df[self.true_label_col].astype(str).values

        all_labels = sorted(set(pred_labels) | set(true_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.y_true = np.array([self.label_to_idx[label] for label in true_labels], dtype=int)
        self.y_pred = np.array([self.label_to_idx[label] for label in pred_labels], dtype=int)

        prob_cols = [col for col in matched_df.columns if col.startswith("prob_")]
        if prob_cols:
            y_probs = np.zeros((len(matched_df), len(all_labels)), dtype=float)
            for col in prob_cols:
                label = col.replace("prob_", "", 1)
                if label in self.label_to_idx:
                    y_probs[:, self.label_to_idx[label]] = matched_df[col].to_numpy(dtype=float)
            self.y_probs = y_probs
        else:
            y_probs = np.zeros((len(matched_df), len(all_labels)), dtype=float)
            for i, pred_idx in enumerate(self.y_pred):
                y_probs[i, pred_idx] = 1.0
            self.y_probs = y_probs

        print(
            f"Matched {len(self.matched_df)} slides. "
            f"y_true shape: {self.y_true.shape}, "
            f"y_pred shape: {self.y_pred.shape}, "
            f"y_probs shape: {self.y_probs.shape}"
        )
        return self.matched_df

    def assessment_report(self):
        """Print confusion matrix and classification report."""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")

        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        cm = plot_confusion_matrix(self.y_true, self.y_pred, class_names=class_names)
        report = get_classification_report(self.y_true, self.y_pred, class_names=class_names)
        return {"confusion_matrix": cm, "classification_report": report}

    def compute_metrics(self):
        """Compute binary AUC."""
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")

        return {
            "auc": auc_score(self.y_true, self.y_probs),
        }

    def group_by_metrics(self, group_col, slide_id_col="filename", results_path_col="slide_path"):
        """Compute accuracy and AUC grouped by a metadata column."""
        if self.matched_df is None or self.matched_df.empty:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        if group_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{group_col}' not found in metadata_df.")
        if slide_id_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{slide_id_col}' not found in metadata_df.")
        if results_path_col not in self.matched_df.columns:
            raise ValueError(f"Column '{results_path_col}' not found in matched_df.")

        grouped_df = self.matched_df.copy()
        grouped_df["_slide_id"] = grouped_df[results_path_col].apply(self._extract_slide_id)

        metadata_subset = self.metadata_df[[slide_id_col, group_col]].copy()
        metadata_subset["_slide_id"] = metadata_subset[slide_id_col].apply(self._extract_slide_id)

        grouped_df = grouped_df.merge(
            metadata_subset[["_slide_id", group_col]],
            on="_slide_id",
            how="left",
        )

        rows = []
        for group_val, group_data in grouped_df.groupby(group_col, dropna=False):
            y_true_group = np.array([self.label_to_idx[str(label)] for label in group_data[self.true_label_col].astype(str).values], dtype=int)
            y_pred_group = np.array([self.label_to_idx[str(label)] for label in group_data["pred_label"].astype(str).values], dtype=int)

            prob_cols = [col for col in group_data.columns if col.startswith("prob_")]
            if prob_cols:
                y_probs_group = np.zeros((len(group_data), len(self.label_to_idx)), dtype=float)
                for col in prob_cols:
                    label = col.replace("prob_", "", 1)
                    if label in self.label_to_idx:
                        y_probs_group[:, self.label_to_idx[label]] = group_data[col].to_numpy(dtype=float)
            else:
                y_probs_group = None

            auc_val = np.nan
            if y_probs_group is not None and y_probs_group.shape[1] == 2:
                try:
                    auc_val = float(auc_score(y_true_group, y_probs_group))
                except ValueError:
                    auc_val = np.nan

            rows.append({
                group_col: group_val,
                "n_samples": len(group_data),
                "accuracy": float(np.mean(y_true_group == y_pred_group)),
                "auc": auc_val,
            })

        group_metrics_df = pd.DataFrame(rows)
        group_metrics_df = group_metrics_df.sort_values(
            by=["accuracy", "n_samples"],
            ascending=[False, False],
            na_position="last"
        ).reset_index(drop=True)

        print(f"\nPer-group metrics grouped by '{group_col}':")
        print(group_metrics_df.to_string(index=False))
        return group_metrics_df

    def roc_curve(self):
        """Plot ROC curve for binary classification."""
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        return plot_roc_curve(self.y_true, self.y_probs)

    def pr_curve(self):
        """Plot precision-recall curve for binary classification."""
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        return plot_pr_curve(self.y_true, self.y_probs)


# ----------------------------------------
# Deletion curve evaluation
# ----------------------------------------

def _find_fold_checkpoint(checkpoint_dir, fold_num):
    """Find the checkpoint file for a given fold."""
    pattern = os.path.join(checkpoint_dir, f"fold_{fold_num}_auc_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for fold {fold_num} in {checkpoint_dir} (pattern: {pattern})")
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]

def _load_slide_feats(slide_path, zarr_dir, feature_key):
    """Load the full (unsampled) tile feature matrix for one slide."""
    zarr_path = os.path.join(zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
    wsi = open_wsi(slide_path, zarr_path)
    if feature_key not in wsi.tables:
        raise KeyError(f"Feature key '{feature_key}' not found in zarr tables for {slide_path}")
    adata = wsi.tables[feature_key]
    feats = torch.from_numpy(adata.X).float()
    return feats

@torch.no_grad()
def _slide_deletion_curve(model, feats, frac_grid, device, n_random=5, rng=None):
    """
    Compute deletion curve for a single slide.
    """
    feats = feats.to(device)
    n_tiles = feats.shape[0]
    if rng is None:
        rng = np.random.RandomState(0)

    contrib = get_tile_contributions(model, feats)
    pred_idx = int(torch.argmax(contrib["probabilities"]).item())
    baseline_prob = float(contrib["probabilities"][pred_idx].item())
    importance = contrib["attention"].numpy()

    order = np.argsort(-importance)  # descending: most important tile first

    def _prob_after_deleting(keep_idx):
        if len(keep_idx) == 0:
            return np.nan
        sub_feats = feats[keep_idx]
        with autocast_context():
            out = model(sub_feats)
        probs = torch.softmax(out["logits"], dim=0)
        return float(probs[pred_idx].item())

    importance_curve = []
    random_curve = []

    for frac in frac_grid:
        n_delete = min(int(round(frac * n_tiles)), max(n_tiles - 1, 0))

        # Importance-guided deletion: remove the top n_delete by rank
        keep_idx = order[n_delete:]
        importance_curve.append(_prob_after_deleting(keep_idx))

        # Random deletion baseline, averaged over n_random permutations
        random_probs = []
        for _ in range(n_random):
            perm = rng.permutation(n_tiles)
            keep_idx_r = perm[n_delete:]
            random_probs.append(_prob_after_deleting(keep_idx_r))
        random_curve.append(float(np.nanmean(random_probs)))

    return {
        "pred_idx": pred_idx,
        "baseline_prob": baseline_prob,
        "importance_curve": np.array(importance_curve),
        "random_curve": np.array(random_curve),
    }


def run_deletion_curve_evaluation(
    df,
    filename_col,
    label_col,
    patient_col,
    feature_key,
    zarr_dir,
    checkpoint_dir,
    n_splits=5,
    random_state=42,
    frac_grid=None,
    n_random=5,
    seed=0
):
    """
    Run out-of-fold deletion curves across an entire dataset, reusing an
    existing K-fold checkpoint set.
    """
    if frac_grid is None:
        frac_grid = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100%

    set_seed(random_state)
    df2 = df.copy().reset_index(drop=True)
    label_mapping = create_label_mapping(df2, label_col)
    df2[label_col] = df2[label_col].map(label_mapping).astype(int)

    if df2[label_col].nunique() != 2:
        raise ValueError(
            f"run_deletion_curve_evaluation currently supports binary classification only, but found {df2[label_col].nunique()} classes."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.RandomState(seed)

    rows = []

    # Iterate over the CV folds, only using held-out indices for evaluation
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(df2, df2[label_col], groups=df2[patient_col])): 
        fold_num = fold_idx + 1
        test_df = df2.iloc[test_idx].reset_index(drop=True)

        checkpoint_path = _find_fold_checkpoint(checkpoint_dir, fold_num)
        model, config, _ = load_checkpoint(checkpoint_path)
        device = next(model.parameters()).device
        model.eval()

        # Iterate through held-out slides
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Deletion curves (fold {fold_num})"):
            slide_path = row[filename_col]

            try:
                feats = _load_slide_feats(slide_path, zarr_dir, feature_key)
            except Exception as e:
                print(f"Skipping {slide_path}: {type(e).__name__}: {e}")
                continue

            # Compute targeted-vs-random deletion curves for this slide
            result = _slide_deletion_curve(
                model, feats, frac_grid, device, n_random=n_random, rng=rng,
            )

            for frac, imp_prob, rand_prob in zip(frac_grid, result["importance_curve"], result["random_curve"]):
                rows.append({
                    "slide_path": slide_path,
                    "fold": fold_num,
                    "true_label": row[label_col],
                    "pred_idx": result["pred_idx"],
                    "baseline_prob": result["baseline_prob"], # original class probability before deletion
                    "frac_deleted": frac, # fraction of tiles removed at this step
                    "importance_prob": imp_prob, # probability after deleting highest-importance tiles
                    "random_prob": rand_prob, # probability after deleting random tiles
                })

    return pd.DataFrame(rows)

def deletion_auc_summary(results_df):
    """
    Compute per-slide area under the deletion curve (AUDC) for both the
    importance-guided and random curves.
    """
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    audc_rows = []
    for slide_path, g in results_df.groupby("slide_path"):
        g = g.sort_values("frac_deleted") # ensure x-axis order
        audc_importance = _trapz(g["importance_prob"], g["frac_deleted"]) # area under the importance-guided deletion curve
        audc_random = _trapz(g["random_prob"], g["frac_deleted"]) # area under the random-deletion baseline curve
        audc_rows.append({
            "slide_path": slide_path,
            "audc_importance": audc_importance,
            "audc_random": audc_random,
            "audc_gap": audc_random - audc_importance,  # positive = importance ranking is more faithful
        })
    audc_df = pd.DataFrame(audc_rows)

    print(f"Mean AUDC (importance-guided): {audc_df['audc_importance'].mean():.4f} ± {audc_df['audc_importance'].std():.4f}")
    print(f"Mean AUDC (random baseline):   {audc_df['audc_random'].mean():.4f} ± {audc_df['audc_random'].std():.4f}")
    print(f"Mean gap (random - importance): {audc_df['audc_gap'].mean():.4f} ± {audc_df['audc_gap'].std():.4f}")
    print("(Positive gap = importance-guided deletion collapses confidence faster than random = importance ranking is faithful)")

    return audc_df


def plot_deletion_curves(results_df, title=None):
    """
    Plot mean ± SEM deletion curves (importance-guided vs. random) across all slides.
    SEM = standard error of the mean, computed as std / sqrt(n).
    """
    summary = results_df.groupby("frac_deleted").agg(
        importance_mean=("importance_prob", "mean"),
        importance_sem=("importance_prob", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        random_mean=("random_prob", "mean"),
        random_sem=("random_prob", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the importance-guided deletion curve with mean and SEM shading
    ax.plot(
        summary["frac_deleted"],
        summary["importance_mean"],
        color="#b30000",
        label="Importance-guided deletion",
        marker="o"
    )
    ax.fill_between(
        summary["frac_deleted"],
        summary["importance_mean"] - summary["importance_sem"],
        summary["importance_mean"] + summary["importance_sem"],
        color="#b30000",
        alpha=0.2,
    )

    # Plot the random deletion baseline curve with mean and SEM shading
    ax.plot(
        summary["frac_deleted"],
        summary["random_mean"],
        color="#555555",
        label="Random deletion",
        marker="o",
        linestyle="--"
    )
    ax.fill_between(
        summary["frac_deleted"],
        summary["random_mean"] - summary["random_sem"],
        summary["random_mean"] + summary["random_sem"],
        color="#555555",
        alpha=0.2,
    )

    ax.set_xlabel("Fraction of tiles deleted")
    ax.set_ylabel("Probability of original predicted class")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or "Deletion curve: attention faithfulness")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


# ----------------------------------------
# Five-slide overfit check
# ----------------------------------------

def run_five_slide_overfit_check(
    df,
    filename_col,
    label_col,
    feature_key,
    zarr_dir,
    patient_col=None,
    max_tiles=None,
    seed=42,
    n_epochs=100
):
    """
    Sanity check: can the model memorize 5 slides?
    Success criterion:
      - near-perfect training accuracy
      - near-perfect AUC
      - very low loss
    """
    set_seed(seed)
    require_cuda()

    # Keep only labeled rows
    df = df.dropna(subset=[label_col]).copy()

    # Binary only
    classes = sorted(df[label_col].unique())
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, found {classes}")

    # Pick a tiny balanced subset
    class0 = df[df[label_col] == classes[0]]
    class1 = df[df[label_col] == classes[1]]

    if len(class0) < 2 or len(class1) < 2:
        raise ValueError("Need at least 2 slides from each class for a 5-slide overfit check.")

    if patient_col is not None:
        class0 = class0.drop_duplicates(subset=[patient_col])
        class1 = class1.drop_duplicates(subset=[patient_col])

    tiny_df = pd.concat([
        class0.sample(n=2, random_state=seed),
        class1.sample(n=3, random_state=seed)
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print("Five-slide overfit subset:")
    display_cols = [c for c in [filename_col, label_col, patient_col] if c is not None]
    print(tiny_df[display_cols])

    tiny_dataset = ZarrSlideDataset(
        df=tiny_df,
        filename_col=filename_col,
        label_col=label_col,
        feature_key=feature_key,
        zarr_dir=zarr_dir,
        max_tiles=max_tiles,
        seed=seed,
        require_labels=True,
    )

    tiny_dataset, valid_indices = validate_dataset(tiny_dataset)
    if len(tiny_dataset) != 5:
        raise RuntimeError(f"Expected 5 valid slides, got {len(tiny_dataset)}")

    model, _, _ = train_ABMIL(
        train_df=tiny_df.iloc[valid_indices].reset_index(drop=True),
        train_dataset=tiny_dataset,
        val_dataset=None,
        label_col=label_col,
        n_epochs=n_epochs,
        early_stopping_patience=None,
        seed=seed,
    )

    all_labels, all_preds, all_probs = validate_ABMIL(model, tiny_dataset)
    train_acc = (np.array(all_labels) == np.array(all_preds)).mean()
    train_auc = auc_score(all_labels, all_probs)

    print(f"Five-slide overfit results | Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
    print(classification_report(all_labels, all_preds))

    return model, tiny_df, {
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "train_acc": train_acc,
        "train_auc": train_auc,
    }