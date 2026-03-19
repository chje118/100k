"""
ABMIL (Attention-Based Multiple Instance Learning) for WSI Classification.

ABMIL pipeline for pathology benchmarking:
Stratified k-fold cross validation with proper train/internal-val/test split
Computes macro AUC (one-vs-rest, class-balanced)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import random
import copy


# ========== Helper Functions ==========

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

def _create_dataloader(dataset, batch_size=1, shuffle=False, device=None, worker_init_fn=None):
    """Create DataLoader with standard configuration."""
    kwargs = {"batch_size": batch_size, "shuffle": shuffle}
    if worker_init_fn is not None:
        kwargs["worker_init_fn"] = worker_init_fn
    if isinstance(device, str) and device.startswith("cuda"):
        kwargs["pin_memory"] = True
    return DataLoader(dataset, **kwargs)

def _configure_gpu_optimization():
    """Configure GPU for modern CUDA GPUs."""
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

def _preprocess_batch(feats, tile_ids, label, device):
    """Preprocess batch tensors before forward pass."""
    if feats.dim() == 3:
        feats = feats.squeeze(0)
    if tile_ids.ndim == 2:
        tile_ids = tile_ids.squeeze(0)
    feats = feats.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)
    return feats, tile_ids, label

# TODO: remove this function, call logic inline
def _should_use_amp(device, use_amp):
    """Check if AMP should be enabled."""
    return (isinstance(device, str) and device.startswith("cuda") and 
            use_amp and torch.cuda.is_available())

def _extract_model_dims(model):
    """Extract dimensions from ABMIL model."""
    return {
        'in_dim': model.classifier.in_features,
        'n_classes': model.classifier.out_features,
        'hidden_dim': model.attn[0].out_features
    }

def _compute_metrics(all_labels, all_preds, all_probs):
    """Compute accuracy and AUC metrics."""
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    auc = auc_score(all_labels, all_probs)
    return {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'accuracy': accuracy,
        'auc': auc
    }

def validate_dataset(dataset, verbose=True):
    """
    Filter out invalid slides from dataset.
    
    Parameters:
    - dataset: PyTorch Dataset instance to validate
    - verbose: If True, print validation progress
    
    Returns:
    - filtered_dataset: torch.utils.data.Subset with only valid slides
    - valid_indices: List of valid slide indices
    """
    valid_indices = []
    
    for i in range(len(dataset)):
        try:
            _ = dataset[i]
            valid_indices.append(i)
        except Exception as e:
            if verbose:
                print(f"Invalid slide at index {i}: {type(e).__name__}")
    
    filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)
    if verbose:
        print(f"Dataset validation complete: {len(valid_indices)}/{len(dataset)} valid slides")
    
    return filtered_dataset, valid_indices

# ========== Dataset and Model Definitions ==========

class ZarrSlideDataset(Dataset):
    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir, max_tiles=None, seed=None):
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.max_tiles = max_tiles  # Maximum number of tiles per slide (None = no limit)
        self.seed = seed  # Seed for deterministic tile sampling (if max_tiles is set)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        slide_path = row[self.filename_col]
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))

        wsi = open_wsi(slide_path, zarr_path)
        adata = wsi.tables[self.feature_key]

        feats = torch.tensor(adata.X[:]).float() # tile features as a PyTorch tensor
        tile_ids = np.array(adata.obs['tile_id']) # save tile IDs for visualization
        
        # Apply max_tiles limit if specified with deterministic sampling
        if self.max_tiles is not None and feats.shape[0] > self.max_tiles:
            if self.seed is not None:
                local_rng = np.random.RandomState(self.seed + idx)
            else:
                local_rng = np.random.RandomState(idx)
            
            indices = local_rng.choice(feats.shape[0], self.max_tiles, replace=False)
            feats = feats[indices]
            tile_ids = tile_ids[indices]
        
        label = torch.tensor(row[self.label_col]).long() # slide label as a long integer (for classification)

        return feats, tile_ids, label

class ABMIL(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256):
        """
        in_dim: feature size per tile (e.g. 512, 768, 1024)
        n_classes: number of output classes
        hidden_dim: size of attention hidden layer
        """
        super().__init__()

        # Attention mechanism, producing one attention score per tile
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ) 

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
        # Compute attention scores
        A = self.attn(x)

        # Normalize with softmax (sums to 1)
        A = torch.softmax(A, dim=0)

        # Weighted sum of features (MIL posoling)
        M = torch.sum(A * x, dim=0)

        # Multiclass predictions
        logits = self.classifier(M)

        return logits, A


# ========== Training and Evaluation Pipelines ==========

class KFoldPipeline:
    """
    K-fold cross-validation pipeline for ABMIL model evaluation.
    
    Usage:
        pipeline = KFoldPipeline(df, 'path_col', 'label_col', 'features_key', 'tiles_key', 'zarr_dir')
        results = pipeline.run(n_splits=5, n_epochs=100)
        pipeline.print_results()
        pipeline.plot_fold_distribution()
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
        self.validate_slides()
        self.results = None
        print(f"KFoldPipeline initialized on device: {self.device}")
    
    def validate_slides(self):
        """
        Filter out invalid slides (those that cause errors during loading).
        """
        print(f"\nValidating {len(self.df)} slides...")
        
        temp_dataset = ZarrSlideDataset(
            df=self.df,
            filename_col=self.filename_col,
            label_col=self.label_col,
            feature_key=self.feature_key,
            tile_key=self.tile_key,
            zarr_dir=self.zarr_dir
        )
        
        filtered_dataset, valid_indices = validate_dataset(temp_dataset, verbose=True)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Validation complete: {len(self.df)} valid slides")
        
        return self.df
    
    # TODO: use_amp and compile_models should be set at configuration level
    def kfold_cross_validation(self, n_splits=5, n_epochs=10, early_stopping_patience=5, max_tiles=None, 
            random_state=42, use_amp=True, compile_model=False):
        """
        Run k-fold cross-validation.
        
        Parameters:
        - n_splits: Number of folds
        - n_epochs: Maximum epochs per fold
        - early_stopping_patience: Patience for early stopping
        - max_tiles: Maximum tiles per slide (None = no limit)
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with cross-validation results
        """
        set_seed(random_state)
        print(f"Random seed set to {random_state} for reproducibility")
    
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
        fold_auc_scores = []
        fold_accuracies = []
    
        print(f"Starting {n_splits}-fold cross-validation with early stopping (patience={early_stopping_patience})...")
        print(f"Data leakage prevention: Train (80%) (90% Training | 10% Internal Val → Early stopping) | Test (20%) → Final evaluation")
    
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.df, self.df[self.label_col])):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'='*60}")
        
            # Split fold into 80% train + 20% test
            fold_df = self.df.iloc[train_idx].reset_index(drop=True)
            test_df = self.df.iloc[test_idx].reset_index(drop=True)
        
            # Further split train into 90% train_subset + 10% internal_val
            # Use fold-specific seed for this split
            split_seed = random_state + fold_idx + 1
            train_subset_df, internal_val_df = train_test_split(
                fold_df,
                test_size=1/9, 
                stratify=fold_df[self.label_col],
            random_state=split_seed
            )
        
            print(f"Train subset: {len(train_subset_df)} samples")
            print(f"Internal val: {len(internal_val_df)} samples")
            print(f"Test set: {len(test_df)} samples")
            
            # Use fold-specific seed for deterministic tile sampling
            fold_seed = random_state + fold_idx + 1
        
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
        
            all_labels, all_preds, all_probs = validate_ABMIL(
                model=model,
                val_dataset=test_dataset,
            )
        
            # Compute AUC and accuracy for this fold
            fold_auc = auc_score(all_labels, all_probs)
            fold_accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
        
            fold_auc_scores.append(fold_auc)
            fold_accuracies.append(fold_accuracy)
        
            print(f"Fold {fold_idx + 1} - Test AUC: {fold_auc:.4f}, Test Accuracy: {fold_accuracy:.4f}")

        # Compute mean and std across folds
        mean_auc = np.mean(fold_auc_scores)
        std_auc = np.std(fold_auc_scores)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
    
        results_dict = {
            'fold_auc_scores': fold_auc_scores,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'n_splits': n_splits
        }    
        self.results = results_dict
        return self.results
    
    def print_results(self):
        """ Print cross-validation results summary."""
        if self.results is None:
            print("No results available. Run the pipeline first with .run_kfold_cross_validation()")
            return
        
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Results ({self.results['n_splits']} folds)")
        print(f"{'='*60}")
        print(f"Mean AUC: {self.results['mean_auc']:.4f} ± {self.results['std_auc']:.4f}")
        print(f"Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"\nPer-fold AUC: {[f'{auc:.4f}' for auc in self.results['fold_auc_scores']]}")
        print(f"Per-fold Accuracy: {[f'{acc:.4f}' for acc in self.results['fold_accuracies']]}")
    
    def plot_fold_distribution(self):
        """Plot fold AUC and accuracy distribution."""
        if self.results is None:
            print("No results available. Run the pipeline first with .run_kfold_cross_validation()")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # AUC plot
        axes[0].bar(range(len(self.results['fold_auc_scores'])), self.results['fold_auc_scores'], alpha=0.7)
        axes[0].axhline(self.results['mean_auc'], color='r', linestyle='--', label=f"Mean: {self.results['mean_auc']:.4f}")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("AUC")
        axes[0].set_title("AUC per Fold")
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        
        # Accuracy plot
        axes[1].bar(range(len(self.results['fold_accuracies'])), self.results['fold_accuracies'], alpha=0.7)
        axes[1].axhline(self.results['mean_accuracy'], color='r', linestyle='--', label=f"Mean: {self.results['mean_accuracy']:.4f}")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy per Fold")
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()

    # TODO save best model for later usage and interpretability (e.g. attention visualization)


# TODO add gpu config here
def train_ABMIL(
    train_df,
    train_dataset,
    val_dataset=None,
    label_col=None,
    n_epochs=10,
    early_stopping_patience=None,
    seed=None
):
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
    
    # HERTIL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DataLoader: decides when items are loaded, handles shuffling and batching
    # Use worker_init_fn to ensure reproducibility with multiple workers
    def worker_init_fn(worker_id):
        if seed is not None:
            set_seed(seed + worker_id)
    
    train_loader = _create_dataloader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        device=device,
        worker_init_fn=worker_init_fn if seed is not None else None
    )

    # TODO: use helper functions instead
    # Extract Feature Dimension and Number of Classes
    sample_feats, _, _ = train_dataset[0]
    feat_dim = sample_feats.shape[1]
    n_classes = train_df[label_col].nunique()

    # Create the ABMIL Model
    model = ABMIL(feat_dim, n_classes).to(device)

    # Enable TF32 / high matmul precision on modern GPUs (e.g. H100)
    _configure_gpu_optimization()
    
    # Why not just always compile
    # Compile model for additional speed (PyTorch 2.x+)
    if compile_model and hasattr(torch, "compile") and device.startswith("cuda"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")

    # Create Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss function: CrossEntropyLoss for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Early stopping setup
    best_auc = -1.0
    epochs_no_improve = 0
    best_model_state = None
    
    # Training Loop
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0

        # Loop over slides
        for feats, tile_ids, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            feats, tile_ids, label = _preprocess_batch(feats, tile_ids, label, device)
        
            if feats.shape[0] == 0:
                continue

            # As much config as possible shoudl be outside (or simplified with helper functions) to maximize readability 
            # Forward pass
            if _should_use_amp(device, use_amp):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(feats)
                    loss = loss_fn(logits.unsqueeze(0), label)
            else:
                logits, _ = model(feats)
                loss = loss_fn(logits.unsqueeze(0), label)

            # Clear gradients
            optimizer.zero_grad()
            
            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss:.4f}", end="")
        
        # Early stopping: evaluate on validation set if provided
        if val_dataset is not None and early_stopping_patience is not None:
            all_labels, all_preds, all_probs = validate_ABMIL(
                model=model,
                val_dataset=val_dataset,
                )
            val_auc = auc_score(all_labels, all_probs)
            print(f" | Val AUC: {val_auc:.4f}", end="")
            
            # Check if validation AUC improved
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                # Use deep copy to avoid shallow copy issues where tensors may drift during training
                best_model_state = copy.deepcopy(model.state_dict())
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

    return model, best_model_state


def validate_ABMIL(
    model,
    val_dataset,
    ):
    device = next(model.parameters()).device

    # Validation DataLoader
    val_loader = _create_dataloader(val_dataset, batch_size=1, shuffle=False, device=device)

    # Validation Loop
    model.eval()   # Disables training behaviors (dropout etc.)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():   # Disables gradient computation (save memory)
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False, disable=not verbose):
            feats, tile_ids, label = _preprocess_batch(feats, tile_ids, label, device)

            if feats.shape[0] == 0:
                continue

            # Forward pass (optionally with mixed precision on CUDA)
            if _should_use_amp(device, use_amp):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(feats)
            else:
                logits, _ = model(feats)

            # Compute predicted class and probabilities
            probs = torch.softmax(logits.float(), dim=0).cpu().numpy()
            pred = torch.argmax(logits, dim=0).item()
            
            all_labels.append(label.item())
            all_preds.append(pred)
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    
    return all_labels, all_preds, all_probs

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

def save_model(model, model_name):
    """
    Save the trained ABMIL model to disk with auto-generated filename including model parameters.
    
    Parameters:
    - model (ABMIL): Trained ABMIL model
    - model_name (str): Base name for the model (e.g., 'abmil_placenta')
    """
    dims = _extract_model_dims(model)
    filename = f"{model_name}_{dims['in_dim']}_{dims['n_classes']}_{dims['hidden_dim']}.pth"
    save_path = os.path.join("models/", filename)
    
    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def _parse_model_dimensions(model_path):
    """ Parse dimensions from model filename: `..._<in_dim>_<n_classes>_<hidden_dim>.pth`. """
    base = os.path.basename(model_path)
    parsed = re.search(r"_(\d+)_(\d+)_(\d+)\.pth$", base)
    if not parsed:
        raise ValueError(
            f"Could not parse dims from model filename '{base}'. "
            "Expected suffix like '_<in_dim>_<n_classes>_<hidden_dim>.pth'."
        )
    return int(parsed.group(1)), int(parsed.group(2)), int(parsed.group(3))

def load_model(model_path):
    """
    Load a trained ABMIL model. 
    Expects filename in format: `..._<in_dim>_<n_classes>_<hidden_dim>.pth`.
    """
    in_dim, n_classes, hidden_dim = _parse_model_dimensions(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} on device: {device}")

    # Initialize model and load state dict
    model = ABMIL(in_dim, n_classes, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model successfully loaded")
    return model, {"in_dim": in_dim, "n_classes": n_classes, "hidden_dim": hidden_dim}

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
    y_true = np.eye(n_classes)[all_labels]

    # Compute macro AUC: unweighted mean of one-vs-rest AUC for each class
    return roc_auc_score(y_true, all_probs, average="macro", multi_class="ovr")

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
    
    y_true = np.eye(n_classes)[all_labels]

    RocCurveDisplay.from_predictions(y_true, all_probs, average="macro")
    plt.title("ROC Curve (Binary Classification)")
    plt.show()

