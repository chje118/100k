import torch
import numpy as np
import os
import lazyslide as zs
from wsidata import open_wsi
import pandas as pd
import matplotlib.pyplot as plt
import math


class SlideAttention:
    def __init__(self, model, dataset, slide_idx, feature_key, tile_key, filename_col, zarr_dir):
        self.model = model # trained ABMIL model
        self.dataset = dataset # ZarrSlideDataset or Subset
        self.slide_idx = slide_idx
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.filename_col = filename_col
        self.zarr_dir = zarr_dir
        self.attention = self._compute_attention()

    def _compute_attention(self):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Load slide features and tile IDs
        feats, tile_ids, label = self.dataset[self.slide_idx]
        if feats.shape[0] == 0:
            print("Slide has no tiles!")
            return None
        feats = feats.to(device)

        # Normalize features
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

        # Forward pass to get attention
        with torch.no_grad():
            logits, A = self.model(feats)
        A = A.squeeze(1).cpu().numpy()

        # Normalize to [0,1] - only if there's variation
        if A.max() > A.min():
            A_normalized = (A - A.min()) / (A.max() - A.min() + 1e-8)
        else:
            print("WARNING: All attention weights are identical! Model may not be learning attention.")
            A_normalized = A  # Keep original uniform values for debugging

        # Handle both Subset and direct ZarrSlideDataset
        if hasattr(self.dataset, 'dataset'):
            base_dataset = self.dataset.dataset
            actual_idx = self.dataset.indices[self.slide_idx]
        else:  
            base_dataset = self.dataset
            actual_idx = self.slide_idx

        # Open the WSI
        slide_path = base_dataset.df.iloc[actual_idx][self.filename_col]
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(slide_path).replace(".mrxs", ".zarr"))
        self.wsi = open_wsi(slide_path, zarr_path)

        # Add attention to feature table
        adata = self.wsi.tables[self.feature_key]
        adata.obs['attention'] = A_normalized
        
        print(f"Slide {self.slide_idx}: True Label={label}, Predicted Class={torch.argmax(logits).item()}")
        print(f"Attention weights - Min: {A_normalized.min():.4f}, Max: {A_normalized.max():.4f}, Mean: {A_normalized.mean():.4f}, Std: {A_normalized.std():.4f}")
    
        return A_normalized
    
    def _get_attention_df(self):
        adata = self.wsi.tables[self.feature_key]
        if 'tile_id' not in adata.obs.columns or 'attention' not in adata.obs.columns:
            raise ValueError("Feature table must have 'tile_id' and 'attention' columns")
        
        attention_df = adata.obs[['tile_id', 'attention']].copy()

        return attention_df

    def _get_tile_df(self):
        tile_adata = self.wsi.shapes[self.tile_key]
        if 'tile_id' not in tile_adata.columns or 'geometry' not in tile_adata.columns:
            raise ValueError("Tile shapes must have 'tile_id' and 'geometry' columns")
        
        tile_df = tile_adata[['tile_id', 'geometry']].copy()

        return tile_df
    
    def attention_heatmap(self):
        viewer = zs.pl.WSIViewer(self.wsi)
        viewer.add_tiles(
            key=self.tile_key,
            feature_key=self.feature_key,
            color_by='attention',
            style='heatmap',
            cmap='hot',
            alpha=0.8
        )
        viewer.show()
    
    def select_top_tiles(self, n_tiles=10):   
        # Merge attention scores with tile geometries
        attention_df = self._get_attention_df()
        tile_df = self._get_tile_df()        
        merged = pd.merge(attention_df, tile_df, on='tile_id', how='inner')

        if merged.empty:
            return pd.DataFrame()

        # Sort by attention descending
        merged = merged.sort_values('attention', ascending=False)
        
        # Select top N tiles
        selected_tiles = merged.head(n_tiles)
        
        return pd.DataFrame(selected_tiles)
                
    def top_k_zoom(self, top_k=5, margin=0):
        selected_tiles = self.select_top_tiles(n_tiles=top_k)
        
        cols = 2
        rows = math.ceil(top_k / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

        axes = axes.flatten()
        
        for i, (_, row) in enumerate(selected_tiles.iterrows()):
            if 'geometry' not in row:
                continue

            tile_geometry = row['geometry']

            if not hasattr(tile_geometry, "bounds"):
                print("Geometry does not have bounds attribute; skipping.")
                continue

            minx, miny, maxx, maxy = tile_geometry.bounds

            xmin = minx - margin
            ymin = miny - margin
            xmax = maxx + margin
            ymax = maxy + margin

            zs.pl.tiles(self.wsi, tile_key=self.tile_key, zoom = (xmin, xmax, ymin, ymax), ax=axes[i])

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
    
        plt.tight_layout()
        plt.show()