import torch
import numpy as np
import os
import lazyslide as zs
from wsidata import open_wsi
import pandas as pd


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
        
        tile_df = tile_adata.obs[['tile_id', 'geometry']].copy()

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
    
    def select_top_tiles(self, n_tiles=10, min_distance_px=500.0):    
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

        selected_tiles = []
        for _, row in merged.iterrows():
            if len(selected_tiles) >= n_tiles:
                break
            if not selected_tiles:
                selected_tiles.append(row)
            else:
                # Check distance to already selected tiles
                distances = [row['centroid'].distance(sel['centroid']) for sel in selected_tiles]
                if all(dist > min_distance_px for dist in distances):
                    selected_tiles.append(row)

        return pd.DataFrame(selected_tiles)
                
    def attention_top_k_zoom(self, top_k=5, margin=100):
        selected_tiles = self.select_top_tiles(n_tiles=top_k)
        
        viewer = zs.pl.WSIViewer(self.wsi)

        viewer.add_tiles(
            key=self.tile_key,
            feature_key=self.feature_key,
            color_by='attention',
            style='heatmap',
            cmap='hot',
            alpha=0.8
        )

        for _, row in selected_tiles.iterrows():
            print(f"Selected Tile ID: {row['tile_id']}, Attention: {row['attention']:.4f}")

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

            viewer.add_zoom(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        viewer.show()