"""
ABMIL (Attention-Based Multiple Instance Learning) for WSI Classification.

ABMIL pipeline for pathology benchmarking:
Stratified k-fold cross validation with proper train/internal-val/test split
Computes macro AUC (one-vs-rest, class-balanced)
"""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
import lazyslide as zs
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    """ PyTorch Dataset for loading WSI features from Zarr files. """

    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir, max_tiles=None, seed=None, require_labels = True):
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
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

        # Apply max_tiles limit with deterministic, norm-biased sampling
        if self.max_tiles is not None and feats.shape[0] > self.max_tiles:
            if self.seed is not None:
                local_rng = np.random.RandomState(self.seed + idx)
            else:
                local_rng = np.random.RandomState(idx)

            # Slight sampling bias toward high-information tiles
            norms = torch.linalg.norm(feats, dim=1).cpu().numpy()
            probs = norms / norms.sum() if norms.sum() > 0 else None

            indices = local_rng.choice(feats.shape[0], self.max_tiles, replace=False, p=probs)
            feats = feats[indices]
            tile_ids = tile_ids[indices] 
        
        # Handle missing labels (inference uses label=none)
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
    """ Attention-Based Multiple Instance Learning for WSI Classification. """
    def __init__(self, in_dim, n_classes, hidden_dim=256, n_heads=4):
        """
        in_dim: feature size per tile (e.g. 512, 768, 1024)
        n_classes: number of output classes
        hidden_dim: size of attention hidden layer
        n_heads: number of attention heads
        """
        super().__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Attention mechanism, producing one attention score per tile
        # Gated attention: A = V * U (tanh * sigmoid)
        self.attn_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, n_heads) 

        # Classifier layer, maps final slide embedding to class scores
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        """
        Forward pass of ABMIL:

        Parameters: 
        - x: slide with shape, [n_tiles, feat_dim]
        
        Returns:
        - logits: used for training (CrossEntropyLoss)
        - A: attention weights (for interpretability)
        """
        # Compute attention scores: A = V * U (gated attention)
        V = self.attn_V(x)      # [n_tiles, hidden_dim] (tanh)
        U = self.attn_U(x)      # [n_tiles, hidden_dim] (sigmoid)
        H = V * U               # elementwise gating
        A = self.attn_w(H)      # [n_tiles, n_heads] (attention)

        # Normalize with softmax (sums to 1)
        A = torch.softmax(A, dim=0)

        # MIL pooling, for multiple attention heads
        M = A.T @ x         # [n_heads, feat_dim]
        M = M.mean(dim=0)   

        # Multiclass predictions
        logits = self.classifier(M)

        return logits, A

# ----------------------------------------
# Helper functions
# ----------------------------------------

def set_seed(seed):
    """ Set seeds for reproducibility.
    
    Parameters:
    - seed: Random seed value (integer)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_dataset(dataset):
    """
    Filter out invalid slides from dataset.
    
    Parameters:
    - dataset: PyTorch Dataset instance to validate
    
    Returns:
    - filtered_dataset: torch.utils.data.Subset with only valid slides
    - valid_indices: List of valid slide indices
    """
    valid_indices = []
    for i in tqdm(range(len(dataset)), desc="Validating slides"):
        try:
            _ = dataset[i]
            valid_indices.append(i)
        except Exception as e:
            print(f"Invalid slide at index {i}: {type(e).__name__}")
    
    filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)
    print(f"Dataset validation complete: {len(valid_indices)}/{len(dataset)} valid slides")
    
    return filtered_dataset, valid_indices

def require_cuda():
    """ Raise an error if CUDA is not available. """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ABMIL pipeline, but no CUDA device is available.")
    return "cuda"

def _configure_gpu_optimization():
    """
    Enable optional CUDA performance settings for training/inference on 
    supported NVIDIA GPUs.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:  
        pass
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, "cudnn"):
        torch.backends.cuda.cudnn.allow_tf32 = True

def save_checkpoint(model, config, label_mapping, path):
    """ Save model checkpoint with configuration and label mapping. """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "label_mapping": label_mapping,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path):
    """ Load a checkpoint with model, config, and label mapping. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    label_mapping = checkpoint["label_mapping"]

    # Reconstruct model from config
    model = ABMIL(
        in_dim=config["in_dim"],
        n_classes=config["n_classes"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from {path}")
    print(f"Model config: in_dim={config['in_dim']}, n_classes={config['n_classes']}, hidden_dim={config['hidden_dim']}, n_heads = {config['n_heads']}")
    print(f"Label mapping: {label_mapping}")
    
    return model, config, label_mapping

def create_label_mapping(df, label_col):
    """ Map class labels to integer indices. """
    return {label: i for i, label in enumerate(sorted(df[label_col].unique()))}

# ----------------------------------------
# Training and validation functions
# ----------------------------------------

def train_ABMIL(train_df, train_dataset, val_dataset=None, label_col=None, n_epochs=10, 
    early_stopping_patience=None, seed=None):
    """ 
    Train ABMIL model with optional early stopping for best AUC comparability.
    
    Parameters:
    - train_df: DataFrame with training data
    - train_dataset: ZarrSlideDataset instance
    - val_dataset: Optional ZarrSlideDataset for early stopping.
    - label_col: Column name for labels (required if using early stopping)
    - n_epochs: Maximum number of training epochs
    - early_stopping_patience: Number of epochs with no improvement to wait before stopping.
    - seed: Random seed for reproducibility. If None, uses current random state.
    
    Returns:
    - model: Trained ABMIL model
    - best_model_state: Best model state dict (for restoring best model when using early stopping)
    """
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

    # Create the ABMIL model
    model = ABMIL(feat_dim, n_classes).to(device)

    # Enable TF32 / high matmul precision on modern GPUs (e.g. H100)
    _configure_gpu_optimization()
     
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss function: CrossEntropyLoss for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Early stopping setup
    best_auc = -1.0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    
    # Training Loop
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0

        # Loop over slides
        for feats, tile_ids, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            # Preprocess batch
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)

            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue

            optimizer.zero_grad() # Clear gradients

            # Forward pass with mixed precision on CUDA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, A = model(feats)
                ce_loss = loss_fn(logits.unsqueeze(0), label)

                # Diversity loss to encourage different attention maps
                G = A.T @ A                                 # off-diagonal entries tell how similar two heads are
                I = torch.eye(G.size(0), device=A.device)   # compare with identity matrix
                div_loss = ((G - I) ** 2).mean()            # computes loss, smaller when more independent
                loss = ce_loss + 0.05 * div_loss    # 0.05 can be tuned

            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            total_loss += loss.item() # Accumulate loss

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss:.4f}", end="")
        
        # Early stopping: evaluate on validation set if provided
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
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    model.eval() # Evaluation mode (disables dropout, batch norm updates)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False):
            # Preprocess batch 
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)

            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue

            # Forward pass with mixed precision on CUDA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(feats)
            
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
    """ Compute and plot confusion matrix. """
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
    """ Return and print classification report. """
    if class_names is None:
        report = classification_report(all_labels, all_preds)
    else:
        report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    return report

def auc_score(all_labels, all_probs):
    """
    Compute macro AUC score for multi-class classification, as the unweighted mean 
    of one-vs-rest AUC scores across all classes. This metric is class-balanced and 
    appropriate for multi-class problems where all classes are equally important.
    
    Formula: macro AUC = mean(AUC_class_i for i in 1..n_classes)
    where each AUC_class_i is computed using the one-vs-rest approach.    
    """
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    # Convert true labels to one-hot encoding for multi-class AUC computation
    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    # Compute macro AUC: unweighted mean of one-vs-rest AUC for each class 
    try:
        return roc_auc_score(y_true, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        return np.nan

def per_class_auc(all_labels, all_probs):
    """ Compute one-vs-rest AUC for each class. """
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    try:
        return roc_auc_score(y_true, all_probs, average=None, multi_class="ovr")
    except ValueError:
        return np.full(n_classes, np.nan)

def plot_roc_curve(all_labels, all_probs):
    """ Plot ROC curve for binary classification. """
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    if all_probs.shape[1] != 2:
        raise ValueError(
            f"plot_roc_curve only supports binary classification. "
            f"Found {all_probs.shape[1]} classes."
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

def per_class_roc_curves(all_labels, all_probs):
    """ Plot one-vs-rest ROC curves for all classes. """
    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("pastel", n_classes)

    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, c], all_probs[:, c])
        auc_c = roc_auc_score(y_true[:, c], all_probs[:, c])
        plt.plot(fpr, tpr, color=colors[c], lw=2, label=f"Class {c} (AUC={auc_c:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def per_class_pr_curves(all_labels, all_probs):
    """ Plot one-vs-rest precision-recall curves for all classes. """
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("pastel", n_classes)

    for c in range(n_classes):
        p, r, _ = precision_recall_curve(y_true[:, c], all_probs[:, c])
        ap = average_precision_score(y_true[:, c], all_probs[:, c])
        plt.plot(r, p, color=colors[c], linewidth=2, label=f"Class {c} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision–Recall curves (OvR)")
    plt.legend(title="Classes")
    plt.tight_layout()
    plt.show()    

# ----------------------------------------
# Training, inference and evaluation pipelines
# ----------------------------------------

class TrainABMILPipeline:
    """ 
    Train ABMIL with validation, then retrain on all valid slides for the best epoch count and save checkpoint. 
    
    Usage: 
    pipeline = TrainABMILPipeline(df, 'path_col', 'label_col', 'features_key', 'tiles_key', 'zarr_dir', 'save_path')
    pipeline.run_pipeline(max_tiles=50000, n_epochs=100, seed=42, validation_fraction=0.10, early_stopping_patience=5)
    """

    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir, save_path):
        self.df = df.copy()
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.save_path = save_path
        self.label_mapping = create_label_mapping(self.df, self.label_col)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.best_epoch = None

        print(f"TrainABMILPipeline initialized with {len(self.df)} slides on device: {self.device}")

    def _make_dataset(self, df, max_tiles=None, seed=None):
        return ZarrSlideDataset(
            df=df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
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
        """ Filter out slides that cannot be loaded. """
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")
        
        temp_dataset = self._make_dataset(self.df)
        _, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        return self.df

    def train_abmil(self, max_tiles=50000, n_epochs=100, seed=42, validation_fraction=0.10, early_stopping_patience=5):
        """ Fit on a train/validation split and record the best epoch. """
        df = self._map_labels(self.df)
        
        train_df, val_df = train_test_split(
            df,
            test_size=validation_fraction,
            stratify=df[self.label_col],
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
        """ Retrain on all valid slides for best_epoch epochs and save the checkpoint. """
        
        if self.best_epoch is None:
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
    """
    K-fold cross-validation pipeline for ABMIL model evaluation.
    
    Usage:
    """
    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir):
        """
        Parameters:
        - df: DataFrame with slide metadata
        - filename_col: Column name for slide paths
        - label_col: Column name for class labels
        - feature_key: Key for features in zarr (e.g., 'features_h-optimus-0')
        - tile_key: Key for tiles in zarr (e.g., 'tiles_224')
        - zarr_dir: Directory containing zarr files
        """
        self.df = df.copy()
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = None

        print(f"KFoldPipeline initialized on device: {self.device}")

    def _make_dataset(self, df, max_tiles=None, seed=None):
        return ZarrSlideDataset(
            df=df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
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
        fold_per_class_aucs = per_class_auc(all_labels, all_probs)
        return all_labels, all_preds, all_probs, fold_auc, fold_accuracy, fold_per_class_aucs

    def validate_slides(self):
        """ Filter out invalid slides causing errors during loading. """
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")
        
        temp_dataset = self._make_dataset(self.df)
        _, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        return self.df

    def kfold_cross_validation(self, n_splits=5, n_epochs=10, early_stopping_patience=5, max_tiles=None, 
            random_state=42, resume_from_checkpoints=False, checkpoint_dir="checkpoints/"):
        """ 
        Run stratified k-fold cross-validation with internal validation.
        
        Parameters:
        - n_splits: number of folds
        - n_epochs: maximum epochs per fold
        - early_stopping_patience: patience for early stopping
        - max_tiles: maximum tiles per slide (None = no limit)
        - random_state: random seed for reproducibility
        - resume_from_checkpoints: if True, detect completed folds from checkpoint files and continue from next fold
        - checkpoint_dir: directory containing fold checkpoints (fold_{n}_auc_*.pt)
        """
        set_seed(random_state)
        print(f"Random seed set to {random_state} for reproducibility")
    
        df, label_mapping = self._map_labels(self.df)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_ovr_auc_scores = []
        fold_per_class_aucs = []
        fold_accuracies = []
        fold_all_labels = []
        fold_all_preds = []
        fold_all_probs = []

        print(f"Starting {n_splits}-fold cross-validation with early stopping (patience={early_stopping_patience})...")
        print(f"Train (80%) → 90% training / 10% internal val | Test (20%) → final evaluation")

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

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[self.label_col])):
            fold_num = fold_idx + 1
            print(f"\n{'='*60}")
            print(f"Fold {fold_num}/{n_splits}")
            print(f"{'='*60}")

            # Split fold into 80% train + 20% test
            fold_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Further split train into 90% train_subset + 10% internal_val (use fold-specific seed)
            split_seed = random_state + fold_num
            train_subset_df, internal_val_df = train_test_split(
                fold_df,
                test_size=1/9, # 90% train, 10% internal val
                stratify=fold_df[self.label_col],
                random_state=split_seed
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

                all_labels, all_preds, all_probs, fold_auc, fold_accuracy, fold_per_class_aucs = self._evaluate_fold(model, test_dataset)

                fold_ovr_auc_scores.append(fold_auc)
                fold_per_class_aucs.append(fold_per_class_aucs)
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
            all_labels, all_preds, all_probs, fold_auc, fold_accuracy, fold_per_class_aucs = self._evaluate_fold(model, test_dataset)

            fold_ovr_auc_scores.append(fold_auc)
            fold_per_class_aucs.append(fold_per_class_aucs)
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
        per_class_aucs = np.array(fold_per_class_aucs, dtype=float)
        results_dict = {
            "fold_ovr_auc_scores": fold_ovr_auc_scores,
            "mean_ovr_auc": float(np.mean(fold_ovr_auc_scores)) if fold_ovr_auc_scores else np.nan,
            "std_ovr_auc": float(np.std(fold_ovr_auc_scores)) if fold_ovr_auc_scores else np.nan,        
            'fold_accuracies': fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else np.nan,
            "std_accuracy": float(np.std(fold_accuracies)) if fold_accuracies else np.nan,
            'fold_per_class_aucs': fold_per_class_aucs,
            "mean_per_class_auc": np.nanmean(per_class_aucs, axis=0).tolist() if len(per_class_aucs) else [],
            "std_per_class_auc": np.nanstd(per_class_aucs, axis=0).tolist() if len(per_class_aucs) else [],
            'fold_all_labels': fold_all_labels,
            'fold_all_preds': fold_all_preds,
            'fold_all_probs': fold_all_probs,
            'n_splits': n_splits
        }    
        self.results = results_dict
        return self.results

    def print_results(self):
        """ Print cross-validation result summary. """
        if self.results is None:
            print("No results available. Run the pipeline first with .kfold_cross_validation()")
            return
        
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Results ({self.results['n_splits']} folds)")
        print(f"{'='*60}")
        print(f"Mean AUC: {self.results['mean_ovr_auc']:.4f} ± {self.results['std_ovr_auc']:.4f}")
        print(f"Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"\nPer-fold AUC: {[f'{auc:.4f}' for auc in self.results['fold_ovr_auc_scores']]}")
        print(f"Per-fold Accuracy: {[f'{acc:.4f}' for acc in self.results['fold_accuracies']]}")
        print("Per-class AUC (mean ± std):")
        for class_idx, (mean_auc, std_auc) in enumerate(zip(self.results['mean_per_class_auc'], self.results['std_per_class_auc'])):
            print(f"  Class {class_idx}: {mean_auc:.4f} ± {std_auc:.4f}")

class ABMILInference:
    """ Run slide-level ABMIL inference from cached tile features and store attention outputs. """
    def __init__(self, checkpoint_path, zarr_dir, slides, cache_path=None):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.slides = list(slides)
        self.cache_path = cache_path
        self._slide_cache = {}
        self._skipped_slides = []

        # Ensure cache file exists (create empty cache if missing) and load it
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
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, attention = self.model(feats)
            else:
                logits, attention = self.model(feats)

        probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        attention = attention.detach().cpu().numpy()

        if attention.ndim == 2 and attention.shape[1] == 1:
            attention = attention.squeeze(1)

        if attention.ndim == 1:
            attention_for_table = attention
        else:
            attention_for_table = attention.mean(axis=1)

        # Build a tile table with attention scores and geometries
        attention_df = pd.DataFrame({
            "tile_id": tile_ids,
            "attention": attention_for_table,
        })
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
        return slide_data

    def process_slides(self):
        """Process slides sequentially, skipping failures and continuing."""
        processed_count = 0

        for slide_path in tqdm(self.slides, desc="Running ABMIL inference..."):
            if slide_path in self._slide_cache:
                processed_count += 1
                continue

            try:
                self._infer_slide(slide_path)
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
        if self._skipped_slides:
            print(f"Skipped {len(self._skipped_slides)} slides due to errors.")

        return self._slide_cache

    def attention_heatmap(self, slide_path: str):
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
        plt.show()

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
    """ Evaluate slide-level ABMIL inference results against metadata labels. """
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
        """ Extract slide identifier from a path or filename. """
        return os.path.basename(str(slide_path)).replace(".mrxs", "")

    def match_true_labels(self, slide_id_col: str = "filename", results_path_col: str = "slide_path"):
        """ Match predictions with ground-truth labels using slide identifiers. """
        if results_path_col not in self.results_df.columns:
            raise ValueError(f"Column '{results_path_col}' not found in results_df.")
        if slide_id_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{slide_id_col}' not found in metadata_df.")
        if self.true_label_col not in self.metadata_df.columns:
            raise ValueError(f"Column '{self.true_label_col}' not found in metadata_df.")

        results_df = self.results_df.copy()
        metadata_df = self.metadata_df.copy()

        # Extract slide IDs from results paths and metadata paths
        results_df["_slide_id_results"] = results_df[results_path_col].apply(self._extract_slide_id)
        metadata_df["_slide_id_metadata"] = metadata_df[slide_id_col].apply(self._extract_slide_id)

        # Merge results with metadata on slide_id
        matched_df = pd.merge(
            results_df,
            metadata_df[["_slide_id_metadata", self.true_label_col]],
            left_on="_slide_id_results",
            right_on="_slide_id_metadata",
            how="inner"
        )
        
        if matched_df.empty:
            raise ValueError("No matches found between results and metadata.")

        # Clean up temporary columns
        matched_df = matched_df.drop(columns=["_slide_id_results", "_slide_id_metadata"])
        self.matched_df = matched_df
        
        # True and predicted labels as strings
        pred_labels = matched_df["pred_label"].astype(str).values
        true_labels = matched_df[self.true_label_col].astype(str).values

        all_labels = sorted(set(pred_labels) | set(true_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # True and predicted labels as indices
        self.y_true = np.array([self.label_to_idx[label] for label in true_labels], dtype=int)
        self.y_pred = np.array([self.label_to_idx[label] for label in pred_labels], dtype=int)

        # Extract probability columns (all columns starting with "prob_")
        prob_cols = [col for col in matched_df.columns if col.startswith("prob_")]
        if prob_cols:
            # Converts each column name into its class label
            prob_label_map = {col: col.replace("prob_", "", 1) for col in prob_cols}
            # Creates empty array of shape (n_samples, n_classes) to hold probabilities
            y_probs = np.zeros((len(matched_df), len(all_labels)), dtype=float)
            
            # Copies probabilities from matched_df into y_probs based on label mapping
            for col in prob_cols:
                label = prob_label_map[col]
                if label in self.label_to_idx:
                    y_probs[:, self.label_to_idx[label]] = matched_df[col].to_numpy(dtype=float)
            self.y_probs = y_probs

        else:
            # Create empty array of shape (n_samples, n_classes) to hold probabilities
            y_probs = np.zeros((len(matched_df), len(all_labels)), dtype=float)
            
            # Set the predicted class index to 1.0 for each sample (one-hot encoding)
            # Placeholder probabilities if no probability columns are present
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
        """ Print confusion matrix and classification report using shared helper. """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")

        class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        cm = plot_confusion_matrix(self.y_true, self.y_pred, class_names=class_names)
        report = get_classification_report(self.y_true, self.y_pred, class_names=class_names)
        return {"confusion_matrix": cm, "classification_report": report}

    def compute_metrics(self):
        """ Compute macro AUC and per-class AUC using shared helpers. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")

        return {
            "auc": auc_score(self.y_true, self.y_probs),
            "per_class_aucs": per_class_auc(self.y_true, self.y_probs),
        }

    def group_by_metrics(self, group_col, slide_id_col="filename", results_path_col="slide_path"):
        """ Compute accuracy and AUC grouped by a metadata column. """
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

            rows.append({
                group_col: group_val,
                "n_samples": len(group_data),
                "accuracy": float(np.mean(y_true_group == y_pred_group)),
                "auc": float(auc_score(y_true_group, y_probs_group)) if y_probs_group is not None and y_probs_group.shape[1] > 1 else np.nan,
            })
        
        group_metrics_df = pd.DataFrame(rows)
        print(f"\nPer-group metrics grouped by '{group_col}':")
        print(group_metrics_df.to_string(index=False))
        return group_metrics_df

    def roc_curve(self):
        """ Plot ROC curve for binary classification using shared helper. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        return plot_roc_curve(self.y_true, self.y_probs)

    def pr_curves(self):
        """ Plot per-class precision-recall curves using shared helper. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        return per_class_pr_curves(self.y_true, self.y_probs)

    def ovr_roc_curves(self):
        """ Plot one-vs-rest ROC curves for all classes using shared helper. """
        if self.y_true is None or self.y_probs is None:
            raise ValueError("No matched labels found. Call match_true_labels() first.")
        return per_class_roc_curves(self.y_true, self.y_probs)