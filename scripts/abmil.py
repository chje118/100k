"""
ABMIL (Attention-Based Multiple Instance Learning) for WSI Classification.

ABMIL pipeline for pathology benchmarking:
Stratified k-fold cross validation with proper train/internal-val/test split
Computes macro AUC (one-vs-rest, class-balanced)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
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
                model.load_state_dict(best_model_state)
                break
        
        print()

    if val_dataset is not None and early_stopping_patience is not None and best_epoch == 0:
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

def confusion_matrix_report(all_labels, all_preds):
    """
    Generate a confusion matrix report.
 
    Parameters:
    - all_labels: list of true labels
    - all_preds: list of predicted labels
    """
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def auc_score(all_labels, all_probs):
    """
    Compute macro AUC score for multi-class classification (pathology benchmark).
    
    For the pathology benchmark, macro AUC is computed as the unweighted mean of 
    one-vs-rest AUC scores across all classes. This metric is class-balanced and 
    appropriate for multi-class problems where all classes are equally important.
    
    Formula: macro AUC = mean(AUC_class_i for i in 1..n_classes)
    where each AUC_class_i is computed using the one-vs-rest approach.
    
    Parameters:
    - all_labels: list of true labels (class indices)
    - all_probs: array of predicted probabilities (shape: [n_samples, n_classes])
    
    Returns:
    - macro AUC score (float, range [0, 1])
    """
    # Convert true labels to one-hot encoding for multi-class AUC computation
    n_classes = all_probs.shape[1]
    onehot_labels = np.eye(n_classes)[all_labels]

    # Compute macro AUC: unweighted mean of one-vs-rest AUC for each class
    try:
        auc = roc_auc_score(onehot_labels, all_probs, average="macro", multi_class="ovr")
    except ValueError as e:
        print(f"Error computing AUC: {e}")
        auc = np.nan  # Return NaN if AUC cannot be computed (e.g., only one class present)
    return auc

def per_class_auc(all_labels, all_probs):
    n_classes = all_probs.shape[1]
    onehot_labels = np.eye(n_classes)[all_labels]
    
    try:
        aucs = roc_auc_score(onehot_labels, all_probs, average=None, multi_class="ovr")
    except ValueError:
        aucs = [np.nan] * n_classes
    
    return aucs

def plot_roc_curve(all_labels, all_probs):
    """
    Plot ROC curve for binary classification only.
    
    ROC curves are most interpretable for binary classification. For multi-class problems,
    use macro AUC (computed via auc_score) instead.
    
    Parameters:
    - all_labels: list of true labels (must be binary: 0 and 1)
    - all_probs: array of predicted probabilities (shape: [n_samples, 2])
    """
    n_classes = all_probs.shape[1]
    
    if n_classes != 2:
        raise ValueError(
            f"plot_roc_curve only supports binary classification. "
            f"Found {n_classes} classes. Use auc_score() with macro averaging for multi-class."
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

def per_class_pr_curves(all_labels, all_probs):
    """ Plot all per-class PR curves on one figure with distinct colors and class legend. """
    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]  # one-hot

    precisions = {}
    recalls = {}
    ap_scores = {}

    plt.figure(figsize=(8, 6))
    # Use seaborn 'pastel' palette for consistent visuals across plots
    colors = sns.color_palette("pastel", n_classes)

    for c in range(n_classes):
        p, r, _ = precision_recall_curve(y_true[:, c], all_probs[:, c])
        ap = average_precision_score(y_true[:, c], all_probs[:, c])
        precisions[c] = p
        recalls[c] = r
        ap_scores[c] = ap

        plt.plot(r, p, color=colors[c], linewidth=2, label=f"Class {c} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision–Recall curves (OvR)")
    plt.legend(title="Classes")
    plt.tight_layout()
    plt.show()

# ========== Training and Evaluation Pipelines ==========

class TrainABMILPipeline:
    """ Train ABMIL with validation, then retrain on 100% of slides for the best epoch count. """

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

        self.train_df = None
        self.model = None
        self.best_epoch = None
        self.max_tiles = None
        self.n_epochs = None
        self.seed = None
        self.validation_fraction = None
        self.early_stopping_patience = None

        print(f"TrainABMILPipeline initialized with {len(self.df)} slides on device: {self.device}")

    def validate_slides(self):
        """ Filter out slides that cannot be loaded. """
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")
        
        temp_dataset = ZarrSlideDataset(
            df=self.df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
        )

        _, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        return self.df

    def train_abmil(self, max_tiles=50000, n_epochs=100, seed=42, validation_fraction=0.10, early_stopping_patience=5):
        """ Fit on a 90/10 split and record the best epoch. """
        self.max_tiles = max_tiles
        self.n_epochs = n_epochs
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience

        df = self.df.copy()
        df[self.label_col] = df[self.label_col].map(self.label_mapping).astype(int)

        train_df, val_df = train_test_split(
            df,
            test_size=validation_fraction,
            stratify=df[self.label_col],
            random_state=seed,
        )

        train_dataset = ZarrSlideDataset(
            df=train_df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
            max_tiles=max_tiles,
            seed=seed,
        )

        val_dataset = ZarrSlideDataset(
            df=val_df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
            max_tiles=max_tiles,
            seed=seed,
        )

        self.model, _, self.best_epoch = train_ABMIL(
            train_df=train_df,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_col=self.label_col,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            return_best_epoch=True,
        )

        return self.model, self.best_epoch

    def save_abmil(self, max_tiles=50000, n_epochs=100, seed=42):
        """ Retrain on 100% of slides for best_epoch epochs and save the checkpoint. """
        
        full_dataset = ZarrSlideDataset(
            df=self.df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir,
            max_tiles=max_tiles,
            seed=seed,
        )

        self.model, _ = train_ABMIL(
            train_df=self.df,
            train_dataset=full_dataset,
            val_dataset=None,
            label_col=self.label_col,
            n_epochs=self.best_epoch if n_epochs is None else n_epochs,
            early_stopping_patience=None,
            seed=seed,
        )

        self.config = {
            "in_dim": self.model.classifier.in_features,
            "n_classes": self.model.classifier.out_features,
            "hidden_dim": self.model.attn[0].out_features,
            "feature_key": self.feature_key,
            "tile_key": self.tile_key,
            "max_tiles": max_tiles,
            "n_epochs": self.best_epoch if n_epochs is None else n_epochs,
            "seed": seed,
        }

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        save_checkpoint(self.model, self.config, self.label_mapping, self.save_path)
       
        return self.model, self.label_mapping, self.save_path

    def run_pipeline(self, max_tiles=50000, n_epochs=10, seed=42, validation_fraction=0.10, early_stopping_patience=5):
        self.validate_slides()
        self.train_abmil(
            max_tiles=max_tiles,
            n_epochs=n_epochs,
            seed=seed,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
        )
        return self.save_abmil()

class KFoldPipeline:
    """
    K-fold cross-validation pipeline for ABMIL model evaluation.
    
    Usage:
        pipeline = KFoldPipeline(df, 'path_col', 'label_col', 'features_key', 'tiles_key', 'zarr_dir')
        results = pipeline.kfold_cross_validation(n_splits=5, n_epochs=100, max_tiles=5000)
        pipeline.print_results()
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
        print(f"KFoldPipeline initialized on device: {self.device}")
        self.results = None
    
    def validate_slides(self):
        """ Filter out invalid slides causing errors during loading. """
        len_before = len(self.df)
        print(f"\nValidating {len_before} slides...")
        
        temp_dataset = ZarrSlideDataset(
            df=self.df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir
        )
        
        filtered_dataset, valid_indices = validate_dataset(temp_dataset)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Validation complete: {len(self.df)} valid slides (removed {len_before - len(self.df)})")
        
        return self.df
    
    def kfold_cross_validation(self, n_splits=5, n_epochs=10, early_stopping_patience=5, max_tiles=None, 
            random_state=42, resume_from_checkpoints=False, checkpoint_dir="checkpoints/"):
        """ Run k-fold cross-validation.
        
        Parameters:
        - n_splits: Number of folds
        - n_epochs: Maximum epochs per fold
        - early_stopping_patience: Patience for early stopping
        - max_tiles: Maximum tiles per slide (None = no limit)
        - random_state: Random seed for reproducibility
        - resume_from_checkpoints: If True, detect completed folds from checkpoint files and continue from next fold
        - checkpoint_dir: Directory containing fold checkpoints (fold_{n}_auc_*.pt)
        
        Returns:
        - Dictionary with cross-validation results
        """
        set_seed(random_state)
        print(f"Random seed set to {random_state} for reproducibility")
    
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
        fold_auc_scores = []
        fold_accuracies = []
        fold_class_aucs = []
        fold_all_labels = []
        fold_all_preds = []
        fold_all_probs = []
    
        print(f"Starting {n_splits}-fold cross-validation with early stopping (patience={early_stopping_patience})...")
        print(f"Data leakage prevention: Train (80%) (90% Training | 10% Internal Val → Early stopping) | Test (20%) → Final evaluation")

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
    
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.df, self.df[self.label_col])):
            fold_num = fold_idx + 1
            print(f"\n{'='*60}")
            print(f"Fold {fold_num}/{n_splits}")
            print(f"{'='*60}")
        
            # Split fold into 80% train + 20% test
            fold_df = self.df.iloc[train_idx].reset_index(drop=True)
            test_df = self.df.iloc[test_idx].reset_index(drop=True)
        
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
            train_dataset = ZarrSlideDataset(
                df=train_subset_df,
                filename_col=self.filename_col,
                label_col=self.label_col,
                feature_key=self.feature_key,
                tile_key=self.tile_key,
                zarr_dir=self.zarr_dir,
                max_tiles=max_tiles,
                seed=fold_seed
            )
        
            internal_val_dataset = ZarrSlideDataset(
                df=internal_val_df,
                filename_col=self.filename_col,
                label_col=self.label_col,
                feature_key=self.feature_key,
                tile_key=self.tile_key,
                zarr_dir=self.zarr_dir,
                max_tiles=max_tiles,
                seed=fold_seed
            )
        
            test_dataset = ZarrSlideDataset(
                df=test_df,
                filename_col=self.filename_col,
                label_col=self.label_col,
                feature_key=self.feature_key,
                tile_key=self.tile_key,
                zarr_dir=self.zarr_dir,
                max_tiles=max_tiles,
                seed=fold_seed
            )

            if resume_from_checkpoints and fold_num < start_fold:
                checkpoint_path = checkpoint_by_fold.get(fold_num)
                if checkpoint_path is None:
                    raise FileNotFoundError(
                        f"Missing checkpoint for fold {fold_num} in {checkpoint_dir}."
                    )
                print(f"Using existing checkpoint for fold {fold_num}: {checkpoint_path}")
                model, _, _ = load_checkpoint(checkpoint_path)

                all_labels, all_preds, all_probs = validate_ABMIL(
                    model=model,
                    val_dataset=test_dataset,
                )

                fold_auc = auc_score(all_labels, all_probs)
                fold_accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
                fold_per_class_aucs = per_class_auc(all_labels, all_probs)

                fold_auc_scores.append(fold_auc)
                fold_accuracies.append(fold_accuracy)
                fold_class_aucs.append(fold_per_class_aucs)
                fold_all_labels.append(list(all_labels))
                fold_all_preds.append(list(all_preds))
                fold_all_probs.append(all_probs.tolist())

                print(f"Fold {fold_num} (checkpoint) - Test AUC: {fold_auc:.4f}, Test Accuracy: {fold_accuracy:.4f}")
                continue
        
            # Train model on train_subset with early stopping on internal_val
            # Use fold-specific seed derived from random_state for reproducibility
            model, _ = train_ABMIL(
                train_df=train_subset_df,
                train_dataset=train_dataset,
                val_dataset=internal_val_dataset, 
                label_col=self.label_col,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
                seed=fold_seed
            )

            # Validate on test set for final evaluation of this fold
            all_labels, all_preds, all_probs = validate_ABMIL(
                model=model,
                val_dataset=test_dataset,
            )
        
            # Compute AUC and accuracy for this fold
            fold_auc = auc_score(all_labels, all_probs)
            fold_accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
            fold_per_class_aucs = per_class_auc(all_labels, all_probs)
        
            fold_auc_scores.append(fold_auc)
            fold_accuracies.append(fold_accuracy)
            fold_class_aucs.append(fold_per_class_aucs)
            fold_all_labels.append(list(all_labels))
            fold_all_preds.append(list(all_preds))
            fold_all_probs.append(all_probs.tolist())
        
            print(f"Fold {fold_num} - Test AUC: {fold_auc:.4f}, Test Accuracy: {fold_accuracy:.4f}")
            
            # Save checkpoint for this fold
            config = {
                "in_dim": model.classifier.in_features,
                "hidden_dim": model.attn[0].out_features,
                "n_classes": model.classifier.out_features,
                "feature_key": self.feature_key,
                "tile_key": self.tile_key,
                "max_tiles": max_tiles,
            }
            label_mapping = create_label_mapping(self.df, self.label_col)
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_num}_auc_{fold_auc:.4f}.pt")
            save_checkpoint(model, config, label_mapping, checkpoint_path)

        # Compute mean and std across folds
        mean_auc = np.mean(fold_auc_scores)
        std_auc = np.std(fold_auc_scores)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        class_aucs = np.array(fold_class_aucs, dtype=float)
        mean_per_class_auc = np.nanmean(class_aucs, axis=0)
        std_per_class_auc = np.nanstd(class_aucs, axis=0)
    
        results_dict = {
            'fold_auc_scores': fold_auc_scores,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_class_aucs': fold_class_aucs,
            'mean_per_class_auc': mean_per_class_auc.tolist(),
            'std_per_class_auc': std_per_class_auc.tolist(),
            'fold_all_labels': fold_all_labels,
            'fold_all_preds': fold_all_preds,
            'fold_all_probs': fold_all_probs,
            'n_splits': n_splits
        }    
        self.results = results_dict
        return self.results
    
    def print_results(self):
        """ Print cross-validation results summary."""
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
        print("Per-class AUC (mean ± std):")
        for class_idx, (mean_auc, std_auc) in enumerate(
            zip(self.results['mean_per_class_auc'], self.results['std_per_class_auc'])
        ):
            print(f"  Class {class_idx}: {mean_auc:.4f} ± {std_auc:.4f}")

    def _get_evaluation_data(self, fold_idx=None):
        """Return labels, predictions, and probabilities for one fold or all folds."""
        if self.results is None:
            raise ValueError("No results available. Run .kfold_cross_validation() first.")

        fold_all_labels = self.results.get('fold_all_labels', [])
        fold_all_preds = self.results.get('fold_all_preds', [])
        fold_all_probs = self.results.get('fold_all_probs', [])

        if not fold_all_labels or not fold_all_preds or not fold_all_probs:
            raise ValueError("Evaluation outputs are missing from self.results.")

        if fold_idx is None:
            all_labels = [label for fold_labels in fold_all_labels for label in fold_labels]
            all_preds = [pred for fold_preds in fold_all_preds for pred in fold_preds]
            all_probs = np.concatenate([np.asarray(fold_probs) for fold_probs in fold_all_probs], axis=0)
            return all_labels, all_preds, all_probs

        if fold_idx < 1 or fold_idx > len(fold_all_labels):
            raise IndexError(f"fold_idx must be between 1 and {len(fold_all_labels)}")

        index = fold_idx - 1
        return list(fold_all_labels[index]), list(fold_all_preds[index]), np.asarray(fold_all_probs[index])

    def plot_confusion_matrix(self, fold_idx=None):
        """Plot a confusion matrix for one fold or all folds combined."""
        all_labels, all_preds, _ = self._get_evaluation_data(fold_idx=fold_idx)
        confusion_matrix_report(all_labels, all_preds)

    def plot_roc_curve(self, fold_idx=None):
        """Plot ROC curve for binary classification for one fold or all folds combined."""
        all_labels, _, all_probs = self._get_evaluation_data(fold_idx=fold_idx)
        n_classes = all_probs.shape[1]
        title_suffix = f"Fold {fold_idx}" if fold_idx is not None else "All folds"

        if n_classes != 2:
            raise ValueError(
                f"plot_roc_curve only supports binary classification. Found {n_classes} classes."
            )

        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({title_suffix})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
