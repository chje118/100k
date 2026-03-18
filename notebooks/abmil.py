"""
Attention-Based Multiple Instance Learning (ABMIL) for Whole Slide Image Classification.
Assumes binary or multi-class, single-label classification.

Evaluation includes confusion matrix and classification report, AUC and ROC curves.
Model saving and loading functions are included, with auto-generated filenames containing model dimensions for easy tracking.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import lazyslide as zs
from tqdm import tqdm
import re

class ZarrSlideDataset(Dataset):
    def __init__(self, df, filename_col, label_col, feature_key, tile_key, zarr_dir):
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.zarr_dir = zarr_dir
    
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

        # Weighted sum of features (MIL pooling)
        M = torch.sum(A * x, dim=0)

        # Multiclass predictions
        logits = self.classifier(M)

        return logits, A
    
def train_ABMIL(
    train_df,
    train_dataset,
    label_col,
    n_epochs=10,
    class_weights=None,
    device=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    compile_model=False
):
    """
    Train ABMIL model with optional class weights for handling class imbalance.
    
    Parameters:
    - train_df: DataFrame with training data
    - train_dataset: ZarrSlideDataset instance
    - label_col: Column name for labels
    - n_epochs: Number of training epochs
    - class_weights: Optional tensor of class weights (shape: [n_classes])
            Higher weights give more importance to that class during training.
    - device: Torch device string, e.g. "cuda" or "cpu". Defaults to "cuda" if available.
    - use_amp: If True and running on CUDA, use autocast mixed precision (optimized for H100).
    - amp_dtype: Autocast dtype when use_amp is True (default: torch.bfloat16, good for H100).
    - compile_model: If True and torch.compile is available, compile the model for extra speed.
    """
    # Decide device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # DataLoader: decides when items are loaded, handles shuffling and batching
    default_loader_kwargs = {"batch_size": 1, "shuffle": True,}
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
            
            # Normalize features per slide
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

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

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss:.4f}")

    return model


def validate_ABMIL(
    model,
    val_dataset,
    device=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
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
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False):
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)
            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if feats.shape[0] == 0:
                continue
            
            # Normalize features per slide
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

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
    Compute AUC score for multi-class classification.
    
    Parameters:
    - all_labels: list of true labels
    - all_probs: array of predicted probabilities (shape: [n_samples, n_classes])
    
    Returns:
    - AUC score (float)
    """
    # Convert true labels to one-hot encoding
    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    return roc_auc_score(y_true, all_probs, average="macro")

def plot_roc_curve(all_labels, all_probs):
    """
    Plot ROC curve for multi-class classification.
    
    Parameters:
    - all_labels: list of true labels
    - all_probs: array of predicted probabilities (shape: [n_samples, n_classes])
    """
    n_classes = all_probs.shape[1]
    y_true = np.eye(n_classes)[all_labels]

    RocCurveDisplay.from_predictions(y_true, all_probs, average="macro")
    plt.title("ROC Curve")
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame()
    label_col = "label"
    filename_col = "filename"
    feature_key = "features"
    zarr_dir = "zarr_dir"

    # Splitting Data into Training and Validation Sets
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,      # 20% of slides for validation
        stratify=df[label_col],  # preserve class distribution
        random_state=42
    )

    train_dataset = ZarrSlideDataset(
        df=train_df, 
        filename_col=filename_col, 
        label_col=label_col, 
        feature_key=feature_key,
        tile_key="tiles_224",
        zarr_dir=zarr_dir
    )

    val_dataset = ZarrSlideDataset(
        df=val_df,
        filename_col=filename_col,
        label_col=label_col,
        feature_key=feature_key,
        tile_key="tiles_224",
        zarr_dir=zarr_dir
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    model = train_ABMIL(train_df, train_dataset, label_col, n_epochs=10, class_weights=[1.0, 2.0, 3.0])
    all_labels, all_preds = validate_ABMIL(model, val_dataset)
    confusion_matrix_report(all_labels, all_preds)
