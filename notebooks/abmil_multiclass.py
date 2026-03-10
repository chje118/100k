"""
Attention-Based Multiple Instance Learning (ABMIL) for Whole Slide Image Classification.
Assumes multi-class, single-label classification.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from wsidata import open_wsi
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import lazyslide as zs
from tqdm import tqdm


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
    
def train_ABMIL(train_df, train_dataset, label_col, n_epochs=10, class_weights=None):
    """
    Train ABMIL model with optional class weights for handling class imbalance.
    
    Parameters:
    - train_df: DataFrame with training data
    - train_dataset: ZarrSlideDataset instance
    - label_col: Column name for labels
    - n_epochs: Number of training epochs
    - class_weights: Optional tensor of class weights (shape: [n_classes])
                   Higher weights give more importance to that class during training.
                   Useful for handling class imbalance or emphasizing severe classes.
    """

    # Extract Feature Dimension and Number of Classes
    sample_feats, _, _ = train_dataset[0]
    feat_dim = sample_feats.shape[1]
    n_classes = train_df[label_col].nunique()

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Using device: {device}")

    # Create the ABMIL Model
    model = ABMIL(feat_dim, n_classes).to(device)

    # Create Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if class_weights is not None:
        if isinstance(class_weights, (list, np.ndarray)):
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        elif isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.to(device)
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
            feats = feats.to(device)
            label = label.to(device)
        
            if feats.shape[0] == 0:
                continue
            
            # Normalize features per slide
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

            # Forward pass
            logits, _ = model(feats)
            
            # Compute loss
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


def compute_class_weights(train_df, label_col, method="balanced"):
    """
    Compute class weights for handling class imbalance in ABMIL training.
    
    Parameters:
    - train_df: DataFrame with training data
    - label_col: Column name for labels
    - method: Weighting strategy
        - "balanced": Inverse frequency weighting (sklearn-like)
        - "severity": Custom weights for different severity levels
        - "manual": Return template for manual specification
    
    Returns:
    - torch.Tensor: Class weights tensor
    """
    labels = train_df[label_col].values
    n_classes = len(np.unique(labels))
    class_counts = np.bincount(labels, minlength=n_classes)
    
    if method == "balanced":
        # Inverse frequency weighting (sklearn.utils.class_weight.compute_class_weight)
        weights = len(labels) / (n_classes * class_counts)
        weights = weights / weights.sum() * n_classes  # Normalize so mean weight = 1
        
    elif method == "severity":
        # Example: Assume higher class indices are more severe
        # Customize this based on your specific severity mapping
        severity_weights = np.array([1.0, 2.0, 3.0, 4.0])  # Example for 4 classes
        if len(severity_weights) != n_classes:
            print(f"WARNING: Severity weights ({len(severity_weights)}) don't match n_classes ({n_classes})")
            print("Using balanced weights instead")
            weights = len(labels) / (n_classes * class_counts)
        else:
            weights = severity_weights
            
    elif method == "manual":
        print("Manual class weights template:")
        print(f"Number of classes: {n_classes}")
        print(f"Class counts: {class_counts}")
        print("Example weights (modify as needed):")
        template_weights = np.ones(n_classes)
        for i in range(n_classes):
            print(f"  Class {i}: weight = {template_weights[i]:.2f} (count: {class_counts[i]})")
        return None
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced', 'severity', or 'manual'")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"Computed {method} class weights: {weights}")
    return weights_tensor


def validate_ABMIL(model, val_dataset):
    device = next(model.parameters()).device

    # Validation DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Validation Loop
    model.eval()   # Disables training behaviors (dropout etc.)

    all_labels = []
    all_preds = []

    with torch.no_grad():   # Disables gradient computation (save memory)
        for feats, tile_ids, label in tqdm(val_loader, desc="Validation", leave=False):
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            if tile_ids.ndim == 2:
                tile_ids = tile_ids.squeeze(0)
            feats = feats.to(device)
            label = label.to(device)

            if feats.shape[0] == 0:
                continue
            
            # Normalize features per slide
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

            logits, _ = model(feats)

            # Compute predicted class
            pred = torch.argmax(logits, dim=0).item()
            
            all_labels.append(label.item())
            all_preds.append(pred)

    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"Validation Accuracy: {accuracy:.4f}")

    return all_labels, all_preds

def confusion_matrix_report(all_labels, all_preds):
    """
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


def view_slide_attention(model, dataset, slide_idx, feature_key, filename_col, zarr_dir, tile_key='tiles_224',
                         zoom_top_k: int = 5, zoom_margin: float = 0, zoom_method="top_k"):
    """ Visualize a slide's attention heatmap and optionally zoom in on high-attention regions.

    Parameters:
        model (ABMIL): trained attention-based MIL model
        dataset: instance of ZarrSlideDataset or a Subset thereof
        slide_idx (int): index of slide within the dataset to visualize
        feature_key (str): key used to lookup features in the Zarr tables
        filename_col (str): column name in dataframe containing slide paths
        zarr_dir (str): directory where corresponding .zarr folders live
        tile_key (str): tile key to display (default: 'tiles_224')
        zoom_top_k (int): number of highest-attention tiles to bound for zooming.
            Set to ``None`` or ``0`` to disable automatic zooming.
        zoom_margin (float): extra padding (pixels) around computed bounding box.
        zoom_method (str): method for zooming ('top_k' or 'top_concentration').
            'top_k' zooms to the bounding box of the top K attention tiles.
            'top_concentration' to be implemented
    """
    model.eval()
    device = next(model.parameters()).device

    # Load slide features and tile IDs
    feats, tile_ids, label = dataset[slide_idx]
    if feats.shape[0] == 0:
        print("Slide has no tiles!")
        return None
    feats = feats.to(device)

    # Normalize features
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    # Forward pass to get attention
    with torch.no_grad():
        logits, A = model(feats)

    A = A.squeeze(1).cpu().numpy()
    print(f"DEBUG: Raw attention before normalization - shape: {A.shape}, min: {A.min():.6f}, max: {A.max():.6f}, std: {A.std():.6f}")
    
    # Normalize to [0,1] - only if there's variation
    if A.max() > A.min():
        A_normalized = (A - A.min()) / (A.max() - A.min() + 1e-8)
        print(f"DEBUG: Attention after normalization - shape: {A_normalized.shape}, min: {A_normalized.min():.6f}, max: {A_normalized.max():.6f}, std: {A_normalized.std():.6f}")
    else:
        print("WARNING: All attention weights are identical! Model may not be learning attention.")
        A_normalized = A  # Keep original uniform values for debugging

    # Handle both Subset and direct ZarrSlideDataset
    if hasattr(dataset, 'dataset'):
        base_dataset = dataset.dataset
        actual_idx = dataset.indices[slide_idx]
    else:  
        base_dataset = dataset
        actual_idx = slide_idx

    # Open the WSI
    slide_path = base_dataset.df.iloc[actual_idx][filename_col]
    zarr_path = os.path.join(zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
    wsi = open_wsi(slide_path, zarr_path)

    # Add attention to feature table
    adata = wsi.tables[feature_key]
    print(f"DEBUG: adata.obs shape: {adata.obs.shape}, adata.X shape: {adata.X.shape}, attention shape: {A_normalized.shape}")
    
    if len(A_normalized) != adata.n_obs:
        print(f"ERROR: Attention weights ({len(A_normalized)}) don't match observations ({adata.n_obs})!")
        return A_normalized
    
    adata.obs['attention'] = A_normalized

    # Plot with WSIViewer
    viewer = zs.pl.WSIViewer(wsi)
    viewer.add_tiles(
        key=tile_key,
        feature_key=feature_key,
        color_by='attention',
        style='heatmap',
        cmap='hot',
        alpha=0.8
    )

    # optionally add a zoom around the top attention tiles
    if zoom_top_k and zoom_top_k > 0:
        if zoom_method == "top_k":
            add_top_k_zoom(viewer, adata, A_normalized, tile_key, wsi, zoom_top_k, zoom_margin)
        else:
            add_top_concentration_zoom(viewer, adata, A_normalized, tile_key, wsi, zoom_margin)

    viewer.show()
    
    print(f"Slide {slide_idx}: True Label={label}, Predicted Class={torch.argmax(logits).item()}")
    print(f"Attention weights - Min: {A_normalized.min():.4f}, Max: {A_normalized.max():.4f}, Mean: {A_normalized.mean():.4f}, Std: {A_normalized.std():.4f}")
    
    return A_normalized

def add_top_k_zoom(viewer, adata, attention, tile_key, wsi, top_k, margin):
    """ Add a zoom window around the highest-attention tiles.

    Parameters:
        viewer (zs.pl.WSIViewer): instance of the slide viewer
        adata (anndata.AnnData): feature table with 'tile_id' column
        attention (np.ndarray): attention weights, one per tile
        tile_key (str): key for tile shapes in wsi.shapes
        wsi: WSI object with tables
        top_k (int): number of top tiles to include in the bounding box
        margin (float or int): extra padding (in pixels) around the computed box
    
    Data Assumptions:
    Attention weights in tables[feature_key].obs['attention'] with corresponding tile IDs in tables[feature_key].obs['tile_id']
    Tile shapes in wsi.shapes[tile_key] with 'tile_id' column to match with feature table
    Geometry of each tile in wsi.shapes[tile_key].obs['geometry'] as a Polygon
    """

    # select indices of top attention scores
    if top_k >= len(attention):
        top_idxs = np.arange(len(attention))
    else:
        top_idxs = np.argsort(attention)[-top_k:]

    # get tile_ids for top attention tiles
    top_tile_ids = adata.obs['tile_id'].iloc[top_idxs]

    # get tile shapes
    tile_adata = wsi.shapes[tile_key]
    if 'tile_id' not in tile_adata.obs.columns:
        print("add_top_k_zoom: 'tile_id' column not found in tile shapes; skipping zoom")
        return

    # find matching rows in tile table
    tile_mask = tile_adata.obs['tile_id'].isin(top_tile_ids)
    if not tile_mask.any():
        print("add_top_k_zoom: no matching tiles found; skipping zoom")
        return

    tiles = tile_adata.obs[tile_mask]

    # compute bounding box from tile geometries 
    if 'geometry' in tiles.columns:
        # Get bounds for each polygon: (minx, miny, maxx, maxy)
        bounds = tiles['geometry'].apply(lambda p: p.bounds)
        # bounds is a Series of tuples, unpack to find overall min/max
        xmin = bounds.apply(lambda b: b[0]).min() - margin
        ymin = bounds.apply(lambda b: b[1]).min() - margin
        xmax = bounds.apply(lambda b: b[2]).max() + margin
        ymax = bounds.apply(lambda b: b[3]).max() + margin
        viewer.add_zoom(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    else:
        print("add_top_k_zoom: 'geometry' column not found in tile shapes; skipping zoom")

def add_top_concentration_zoom(viewer, adata, attention, tile_key, wsi, margin):
    # to be implemented: compute zoom region based on attention concentration (e.g. using KDE or clustering)
    raise NotImplementedError("Top concentration zoom method not implemented yet")


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


def load_model(model_path, in_dim, n_classes, hidden_dim=256):
    """
    Load a trained ABMIL model from disk.
    
    Parameters:
    - model_path (str): Path to the saved model
    - in_dim (int): Feature size per tile (must match training)
    - n_classes (int): Number of output classes (must match training)
    - hidden_dim (int): Size of attention hidden layer (must match training)
    
    Returns:
        ABMIL: Loaded model ready for inference
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} on device: {device}")

    model = ABMIL(in_dim, n_classes, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model successfully loaded")
    return model

class ABMILTileSelector:
    def __init__(self, wsi, feature_key, tile_key='tiles_224', n_tiles=10, min_distance_px=500.0):
        self.wsi = wsi
        self.tile_key = tile_key
        self.feature_key = feature_key
        self.n_tiles = n_tiles
        self.min_distance_px = min_distance_px

    def _get_attention_df(self):
        feature_adata = self.wsi.tables[self.feature_key]
        if 'attention' not in feature_adata.obs.columns or 'tile_id' not in feature_adata.obs.columns:
            raise ValueError("Feature table must have 'attention' and 'tile_id' columns")

        attention_df = feature_adata.obs[['tile_id', 'attention']].copy()

        return attention_df
    
    def _get_tile_df(self):
        tile_adata = self.wsi.shapes[self.tile_key]
        if 'tile_id' not in tile_adata.columns or 'geometry' not in tile_adata.columns:
            raise ValueError("Tile shapes must have 'tile_id' and 'geometry' columns")
        
        tile_df = tile_adata.obs[['tile_id', 'geometry']].copy()

        return tile_df
    
    def select_tiles(self):
        """Select n tiles based on ABMIL attention scores with spatial diversity constraint.

        Returns:
            pd.DataFrame: Selected tiles with their metadata
        """
        # Merge attention scores with tile geometries
        attention_df = self._get_attention_df()
        tile_df = self._get_tile_df()        
        merged = pd.merge(attention_df, tile_df, on='tile_id', how='inner')

        if merged.empty:
            return pd.DataFrame()

        # Sort by attention descending
        merged = merged.sort_values('attention', ascending=False)

        # Compute centroids for distance calculations
        merged['centroid'] = merged['geometry'].apply(lambda p: p.centroid)

        selected = []
        for _, row in merged.iterrows():
            if len(selected) >= self.n_tiles:
                break
            
            # Check distance to already selected tiles
            for sel_row in selected:
                dist = row['centroid'].distance(sel_row['centroid'])
                if dist < self.min_distance_px:
                    break
            else:
                selected.append(row)

        if selected:
            return pd.DataFrame(selected).drop(columns=['centroid'])
        else:
            return pd.DataFrame()

# Example Usage
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
    
    model = train_ABMIL(train_df, train_dataset, label_col, n_epochs=10)
    all_labels, all_preds = validate_ABMIL(model, val_dataset)
    confusion_matrix_report(all_labels, all_preds)

    # show attention heatmap and automatically zoom into top-k tiles
    view_slide_attention(
        model,
        val_dataset,
        slide_idx=0,
        filename_col=filename_col,
        zarr_dir=zarr_dir,
        zoom_top_k=5,
        zoom_margin=50,
        zoom_kwargs={"edgecolor": "cyan", "alpha": 0.3}
    )