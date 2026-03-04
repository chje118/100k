"""
Attention-Based Multiple Instance Learning (ABMIL) for Whole Slide Image Classification.
Assumes multi-class, single-label classification.
"""
# TODO: tqdm progress bars
# TODO: check parameters for ABMIL (hidden_dim, learning rate, etc.) for loading/saving model

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
        adata = wsi.tables[self.tile_key]

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
        param: 
        x: slide with shape, [n_tiles, feat_dim]
        
        output:
        logits: used for training (CrossEntropyLoss)
        A: attention weights (for interpretability)
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
    
def train_ABMIL(train_df, train_dataset, label_col,n_epochs=10):
    # DataLoader: decides when items are loaded, handles shuffling and batching
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Extract Feature Dimension and Number of Classes
    sample_feats, _, _ = train_dataset[0]
    feat_dim = sample_feats.shape[1]
    n_classes = train_df[label_col].nunique()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the ABMIL Model
    model = ABMIL(feat_dim, n_classes).to(device)

    # Create Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        # Loop over slides
        for feats, tile_ids, label in train_loader:
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


def validate_ABMIL(model, val_dataset):
    device = next(model.parameters()).device

    # Validation DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Validation Loop
    model.eval()   # Disables training behaviors (dropout etc.)

    all_labels = []
    all_preds = []

    with torch.no_grad():   # Disables gradient computation (save memory)
        for feats, tile_ids, label in val_loader:
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
    all_labels: list of true labels
    all_preds: list of predicted labels

    Interpretation:
    If accuracy is low → model may overfit or you may have too few slides
    If accuracy is high → attention can now be visualized (next step)
    """
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def view_slide_attention(abmil_model, dataset, slide_idx, feature_key, filename_col, zarr_dir):
    abmil_model.eval()
    device = next(model.parameters()).device

    # Load slide features and tile IDs
    feats, tile_ids, label = dataset[slide_idx]
    if feats.shape[0] == 0:
        print("Slide has no tiles!")
        return
    feats = feats.to(device)

    # Normalize features
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    # Forward pass to get attention
    with torch.no_grad():
        logits, A = model(feats)

    A = A.squeeze(1).cpu().numpy()
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)  # normalize to [0,1]

    # Open the WSI
    slide_path = dataset.df.iloc[slide_idx][filename_col]
    zarr_path = os.path.join(zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
    wsi = open_wsi(slide_path, zarr_path)

    # Add attention to feature table (no subsampling!)
    adata = wsi.tables[feature_key]
    adata.obs['attention'] = A

    # Plot with WSIViewer
    viewer = zs.pl.WSIViewer(wsi)
    viewer.add_tiles(
        key='tiles_224',           # polygon table
        feature_key=feature_key,
        color_by='attention',  # use the column we just added
        style='heatmap',
        cmap='hot',
        alpha=0.8
    )
    viewer.show()


def save_model(model, save_path):
    """
    Save the trained ABMIL model to disk.
    
    Args:
        model (ABMIL): Trained ABMIL model
        save_path (str): Path to save the model (e.g., 'models/abmil_model.pth')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model_path, in_dim, n_classes, hidden_dim=256, device="cpu"):
    """
    Load a trained ABMIL model from disk.
    
    Args:
        model_path (str): Path to the saved model
        in_dim (int): Feature size per tile (must match training)
        n_classes (int): Number of output classes (must match training)
        hidden_dim (int): Size of attention hidden layer (must match training)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        ABMIL: Loaded model ready for inference
    """
    model = ABMIL(in_dim, n_classes, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


# Example Usage
if __name__ == "__main__":
    # update to fit changes made to the code
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
        feature_key=feature_key
    )

    val_dataset = ZarrSlideDataset(
        df=val_df,
        filename_col=filename_col,
        label_col=label_col,
        feature_key=feature_key
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    model = train_ABMIL(train_df, train_dataset, label_col, n_epochs=10)
    all_labels, all_preds = validate_ABMIL(model, val_dataset)
    confusion_matrix_report(all_labels, all_preds)

    view_slide_attention(model, val_dataset, slide_idx=0, feature_key=feature_key, filename_col=filename_col, zarr_dir=zarr_dir)