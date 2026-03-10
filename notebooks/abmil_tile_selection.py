def view_slide_attention(model, dataset, slide_idx, feature_key, filename_col, zarr_dir, tile_key='tiles_224',
                         zoom_top_k: int = 5, zoom_margin: float = 0):
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
        add_top_k_zoom(viewer, adata, A_normalized, tile_key, wsi, zoom_top_k, zoom_margin)

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