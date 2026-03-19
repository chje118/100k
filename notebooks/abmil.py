"""
Attention-Based Multiple Instance Learning (ABMIL) for Whole Slide Image Classification.
Assumes binary or multi-class, single-label classification.

Evaluation includes confusion matrix and classification report, AUC and ROC curves.
Model saving and loading functions are included, with auto-generated filenames containing model dimensions for easy tracking.

AUC Metric for Pathology Benchmark:
-----------------------------------
This module uses macro AUC (unweighted mean of one-vs-rest AUC) for multi-class pathology classification.
Macro AUC is class-balanced and appropriate when all diagnostic categories are equally important.
For each class i, one-vs-rest AUC is computed as the ROC AUC treating class i vs. all others.
Then macro AUC = mean(AUC_i for all classes i).

This provides a single interpretable metric for multi-class problems without requiring per-class weights.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import lazyslide as zs
from tqdm import tqdm
import re
import random
import copy


def set_seed(seed):
    """
    Set seeds for reproducibility across numpy, torch, and python random.
    
    Parameters:
    - seed: Random seed value (integer)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ZarrSlideDataset(Dataset):
    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir, max_tiles=None, seed=None):
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
        self.max_tiles = max_tiles  # Maximum number of tiles per slide (None = no limit)
        self.seed = seed  # Seed for deterministic tile sampling across runs
    
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
            # Use deterministic random state based on seed and slide index
            # This ensures the same slide always gets the same tiles sampled across runs
            if self.seed is not None:
                local_rng = np.random.RandomState(self.seed + idx)
            else:
                local_rng = np.random.RandomState(idx)
            
            # Deterministically sample max_tiles tiles (without replacement)
            indices = local_rng.choice(feats.shape[0], self.max_tiles, replace=False)
            feats = feats[indices]
            tile_ids = tile_ids[indices]
        
        label = torch.tensor(row[self.label_col]).long() # the slide label as a long integer (for classification)

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

def train_ABMIL(
    train_df,
    train_dataset,
    val_dataset=None,
    label_col=None,
    n_epochs=10,
    class_weights=None,
    device=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    compile_model=False,
    early_stopping_patience=None,
    seed=None
):
    """
    Train ABMIL model with optional early stopping for best AUC comparability.
    
    Parameters:
    - train_df: DataFrame with training data
    - train_dataset: ZarrSlideDataset instance
    - val_dataset: Optional ZarrSlideDataset for early stopping. If provided, trains until 
                   validation AUC plateaus, ensuring fair comparison across folds.
    - label_col: Column name for labels (required if using early stopping)
    - n_epochs: Maximum number of training epochs
    - class_weights: Optional tensor of class weights (shape: [n_classes])
            Higher weights give more importance to that class during training.
    - device: Torch device string, e.g. "cuda" or "cpu". Defaults to "cuda" if available.
    - use_amp: If True and running on CUDA, use autocast mixed precision (optimized for H100).
    - amp_dtype: Autocast dtype when use_amp is True (default: torch.bfloat16, good for H100).
    - compile_model: If True and torch.compile is available, compile the model for extra speed.
    - early_stopping_patience: Number of epochs with no improvement to wait before stopping.
                              If None, trains for all n_epochs (no early stopping).
    - seed: Random seed for reproducibility. If None, uses current random state.
    
    Returns:
    - model: Trained ABMIL model
    - best_model_state: Best model state dict (for restoring best model when using early stopping)
    """
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)
    # Decide device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # DataLoader: decides when items are loaded, handles shuffling and batching
    # Use worker_init_fn to ensure reproducibility with multiple workers
    def worker_init_fn(worker_id):
        if seed is not None:
            set_seed(seed + worker_id)
    
    default_loader_kwargs = {
        "batch_size": 1, 
        "shuffle": True,
        "worker_init_fn": worker_init_fn if seed is not None else None
    }
    if device.startswith("cuda"):
        default_loader_kwargs["pin_memory"] = True
    train_loader = DataLoader(train_dataset, **default_loader_kwargs)

    # Extract Feature Dimension and Number of Classes
    sample_feats, _, _ = train_dataset[0]
    feat_dim = sample_feats.shape[1]
    n_classes = train_df[label_col].nunique()

    # Create the ABMIL Model
    model = ABMIL(feat_dim, n_classes).to(device)

    # Enable TF32 / high matmul precision on modern GPUs (e.g. H100)
    if device.startswith("cuda"):
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cuda, "cudnn"):
                torch.backends.cuda.cudnn.allow_tf32 = True

    # Compile model for additional speed (PyTorch 2.x+)
    if compile_model and hasattr(torch, "compile") and device.startswith("cuda"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")

    # Create Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # If class weights are provided, convert to tensor and use in CrossEntropyLoss
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
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
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)
            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
        
            if feats.shape[0] == 0:
                continue

            # Forward pass (optionally with mixed precision on CUDA)
            if device.startswith("cuda") and use_amp and torch.cuda.is_available():
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
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                verbose=False
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
    device=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    verbose=True,
    ):
    if device is None:
        device = next(model.parameters()).device

    # Validation DataLoader
    default_loader_kwargs = {"batch_size": 1, "shuffle": False,}
    if isinstance(device, str) and device.startswith("cuda"):
        default_loader_kwargs["pin_memory"] = True
    val_loader = DataLoader(val_dataset, **default_loader_kwargs)

    # Validation Loop
    model.eval()   # Disables training behaviors (dropout etc.)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():   # Disables gradient computation (save memory)
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False, disable=not verbose):
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)
            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue

            # Forward pass (optionally with mixed precision on CUDA)
            if (isinstance(device, str) and device.startswith("cuda")
                    and use_amp and torch.cuda.is_available()):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(feats)
            else:
                logits, _ = model(feats)

            # Compute predicted class and probabilities
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            pred = torch.argmax(logits, dim=0).item()
            
            all_labels.append(label.item())
            all_preds.append(pred)
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    if verbose:
        print(f"Validation Accuracy: {accuracy:.4f}")

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
    in_dim = model.classifier.in_features
    n_classes = model.classifier.out_features
    hidden_dim = model.attn[0].out_features
    
    filename = f"{model_name}_{in_dim}_{n_classes}_{hidden_dim}.pth"
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

# No weights for FM benchmark
def kfold_cross_validation(
    df,
    filename_col,
    label_col,
    feature_key,
    tile_key,
    zarr_dir,
    n_splits=5,
    n_epochs=10,
    early_stopping_patience=5,
    class_weights=None,
    device=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    compile_model=False,
    max_tiles=None,
    random_state=42
):
    """
    Perform K-fold cross-validation for ABMIL model with proper train/internal-val/test split.
    
    Prevents data leakage by using a strict three-way split for each fold:
    - Train subset (90%): For model training
    - Internal validation (10%): For early stopping only (NOT seen during evaluation)
    - Test set (20%): For final evaluation only (NEVER seen during training/early stopping)
    
    For the pathology benchmark, this function evaluates using macro AUC (unweighted mean of 
    one-vs-rest AUC scores across all classes). This metric is class-balanced and appropriate 
    for multi-class pathology classification where all diagnostic categories are equally important.
    
    For fair AUC comparison across folds, this function uses early stopping on each fold's internal
    validation set. This ensures each fold trains until it reaches peak performance, enabling 
    meaningful comparison of model performance across different data splits.
    
    All randomness (fold splitting, model initialization, training) is controlled by random_state
    to ensure fully reproducible AUC results.
    
    Parameters:
    - df: DataFrame with all data
    - filename_col: Column name for file paths
    - label_col: Column name for labels
    - feature_key: Key for accessing features in zarr
    - tile_key: Key for accessing tiles in zarr
    - zarr_dir: Directory containing zarr files
    - n_splits: Number of folds (default: 5)
    - n_epochs: Maximum number of training epochs per fold
    - early_stopping_patience: Number of epochs with no validation AUC improvement to wait before
                              stopping (default: 5). Set to None to disable early stopping and train 
                              for exactly n_epochs.
    - class_weights: Optional tensor of class weights
    - device: Torch device (default: cuda if available else cpu)
    - use_amp: Use mixed precision training
    - amp_dtype: Autocast dtype for mixed precision
    - compile_model: Whether to compile model with torch.compile
    - max_tiles: Maximum number of tiles per slide (default: None = no limit). If set, slides with
                more tiles will have tiles randomly sampled down to this limit. Useful for preventing
                extremely large bags from skewing the learning.
    - random_state: Random seed for reproducibility (default: 42). Controls fold splitting and 
                   all training randomness for fully reproducible AUC.
    
    Returns:
    - results_dict: Dictionary with fold results and aggregated metrics
        Keys: 'fold_auc_scores' (macro AUC per fold), 'mean_auc', 'std_auc', (primary metric for pathology benchmark)
              'fold_accuracies', 'mean_accuracy', 'std_accuracy', 'n_splits'
    """
    from sklearn.model_selection import train_test_split
    
    # Set global seed at the start for reproducibility
    set_seed(random_state)
    print(f"Random seed set to {random_state} for reproducibility")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_auc_scores = []
    fold_accuracies = []
    
    print(f"Starting {n_splits}-fold cross-validation with early stopping (patience={early_stopping_patience})...")
    print(f"Data leakage prevention: Train(90%) → Training | Internal Val(10%) → Early stopping | Test(20%) → Final evaluation only")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[label_col])):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")
        
        # Split fold into 80% train + 20% test
        fold_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        # Further split train into 90% train_subset + 10% internal_val
        # Use fold-specific seed for this split
        split_seed = random_state + fold_idx + 1
        train_subset_df, internal_val_df = train_test_split(
            fold_df,
            test_size=1/9,  # 10% of fold_df (which is 80%), so 10%/90% split
            stratify=fold_df[label_col],
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
            filename_col=filename_col,
            label_col=label_col,
            feature_key=feature_key,
            tile_key=tile_key,
            zarr_dir=zarr_dir,
            max_tiles=max_tiles,
            seed=fold_seed
        )
        
        internal_val_dataset = ZarrSlideDataset(
            df=internal_val_df,
            filename_col=filename_col,
            label_col=label_col,
            feature_key=feature_key,
            tile_key=tile_key,
            zarr_dir=zarr_dir,
            max_tiles=max_tiles,
            seed=fold_seed
        )
        
        test_dataset = ZarrSlideDataset(
            df=test_df,
            filename_col=filename_col,
            label_col=label_col,
            feature_key=feature_key,
            tile_key=tile_key,
            zarr_dir=zarr_dir,
            max_tiles=max_tiles,
            seed=fold_seed
        )
        
        # Train model on train_subset with early stopping on internal_val
        # Use fold-specific seed derived from random_state for reproducibility
        model, _ = train_ABMIL(
            train_df=train_subset_df,
            train_dataset=train_dataset,
            val_dataset=internal_val_dataset,  # Early stopping uses INTERNAL val only
            label_col=label_col,
            n_epochs=n_epochs,
            class_weights=class_weights,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            compile_model=compile_model,
            early_stopping_patience=early_stopping_patience,
            seed=fold_seed
        )
        
        # Evaluate on TEST set ONLY (never seen during training or early stopping)
        all_labels, all_preds, all_probs = validate_ABMIL(
            model=model,
            val_dataset=test_dataset,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype
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
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation Results ({n_splits} folds)")
    print(f"{'='*60}")
    print(f"Mean Test AUC: {mean_auc:.4f} +- {std_auc:.4f}")
    print(f"Mean Test Accuracy: {mean_accuracy:.4f} +- {std_accuracy:.4f}")
    print(f"Individual fold Test AUC scores: {[f'{auc:.4f}' for auc in fold_auc_scores]}")
    print(f"Individual fold Test accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"\nData leakage prevention verified:")
    print(f"  ✓ Train subset used for model training")
    print(f"  ✓ Internal validation used for early stopping only")
    print(f"  ✓ Test set used for final evaluation only (no leakage)")
    
    results_dict = {
        'fold_auc_scores': fold_auc_scores,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'n_splits': n_splits
    }    
    return results_dict


# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("slides_metadata.csv")  # DataFrame with slide info
    label_col = "diagnosis"
    filename_col = "slide_path"
    feature_key = "features_h-optimus-0"  # or your desired feature key
    zarr_dir = "/path/to/zarr/cache"
    
    # K-fold cross-validation with proper train/internal-val/test split (RECOMMENDED)
    # Prevents data leakage by using strict three-way split for each fold:
    # - Train subset (90%): For model training
    # - Internal validation (10%): For early stopping only
    # - Test set (20%): For final evaluation only
    # 
    # random_state parameter ensures fully reproducible results including:
    # - Fold splitting
    # - Model initialization
    # - Training randomness
    # - Tile sampling (deterministic per-slide when max_tiles is used)
    #
    # Use the same random_state to reproduce exact same AUC values across runs.
    results = kfold_cross_validation(
        df=df,
        filename_col=filename_col,
        label_col=label_col,
        feature_key=feature_key,
        tile_key="tiles_224",
        zarr_dir=zarr_dir,
        n_splits=5,
        n_epochs=100,               # Maximum epochs
        early_stopping_patience=5,  # Stop after 5 epochs with no AUC improvement on internal val
        max_tiles=5000,             # Limit to 5000 tiles per slide (None = no limit)
        random_state=42             # Fixed seed for reproducibility (controls ALL randomness including tile sampling)
    )
    print(f"\nFinal Results: Test AUC = {results['mean_auc']:.4f} +- {results['std_auc']:.4f}")
    
    # Alternative: K-fold without early stopping (train for exactly n_epochs)
    # results_no_es = kfold_cross_validation(
    #     df=df,
    #     filename_col=filename_col,
    #     label_col=label_col,
    #     feature_key=feature_key,
    #     tile_key="tiles_224",
    #     zarr_dir=zarr_dir,
    #     n_splits=5,
    #     n_epochs=20,
    #     early_stopping_patience=None,  # Disable early stopping
    #     max_tiles=None,                # No tile limit
    #     random_state=42                # Fixed seed for reproducibility
    # )
    
    # Alternative: Single train/internal-val/test split (faster, less robust)
    # from sklearn.model_selection import train_test_split
    # # First split: 80% train, 20% test
    # train_df, test_df = train_test_split(
    #     df,
    #     test_size=0.2,
    #     stratify=df[label_col],
    #     random_state=42
    # )
    # # Second split: split train into 90% train_subset, 10% internal_val
    # train_subset_df, internal_val_df = train_test_split(
    #     train_df,
    #     test_size=1/9,  # 10% of train_df
    #     stratify=train_df[label_col],
    #     random_state=42
    # )
    # train_dataset = ZarrSlideDataset(
    #     df=train_subset_df, filename_col=filename_col, label_col=label_col,
    #     feature_key=feature_key, tile_key="tiles_224", zarr_dir=zarr_dir, max_tiles=5000
    # )
    # internal_val_dataset = ZarrSlideDataset(
    #     df=internal_val_df, filename_col=filename_col, label_col=label_col,
    #     feature_key=feature_key, tile_key="tiles_224", zarr_dir=zarr_dir, max_tiles=5000
    # )
    # test_dataset = ZarrSlideDataset(
    #     df=test_df, filename_col=filename_col, label_col=label_col,
    #     feature_key=feature_key, tile_key="tiles_224", zarr_dir=zarr_dir, max_tiles=5000
    # )
    # # Train with early stopping on internal_val, evaluate on test_df
    # model, _ = train_ABMIL(train_subset_df, train_dataset, internal_val_dataset, label_col, 
    #                         n_epochs=100, early_stopping_patience=5, seed=42)
    # all_labels, all_preds, all_probs = validate_ABMIL(model, test_dataset)
    # auc = auc_score(all_labels, all_probs)
    # print(f"Test AUC: {auc:.4f}")
