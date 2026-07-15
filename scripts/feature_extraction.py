import gc
import os
from datetime import datetime
import geopandas as gpd
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import lazyslide as zs
from tissue_artifact_segmentation import is_empty_array, _load_cache, _save_cache, _get_processed_entries
from wsidata import open_wsi

# --------------------
# Nvidia H100 Optimizations
# --------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _configure_h100() -> None:
    """Enable GPU settings that improve throughput on H100-class hardware."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return
    try:
        torch.set_float32_matmul_precision("high")  # Prefer faster TF32-style matmul on H100
    except Exception:
        pass
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 matmuls for throughput
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True  # Let cuDNN use TF32 kernels
        torch.backends.cudnn.benchmark = True  # Pick fastest convolution kernels for this input shape

# Configure H100 optimizations at module level
_configure_h100()


# --------------------
# Feature extraction
# --------------------

class ExtractFeatures:
    """Extract tile embeddings and aggregate slide-level features for a single WSI."""

    TISSUE_CANDIDATES = ["tissue_default", "tissue_grandqc", "tissue_threshold"]

    def __init__(
        self,
        wsi_path,
        zarr_dir,
        foundation_model,
        remove_artifacts=False,
        feature_batch_size=512,
        feature_num_workers=None,
        feature_autocast_dtype=None,
    ):
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        os.makedirs(zarr_dir, exist_ok=True)
        self.wsi = open_wsi(self.wsi_path, self.zarr_path)
        self.foundation_model = foundation_model
        self.remove_artifacts = remove_artifacts

        cpu_count = os.cpu_count() or 8
        self.feature_batch_size = feature_batch_size
        self.feature_num_workers = (
            feature_num_workers if feature_num_workers is not None else max(4, min(16, cpu_count // 2))
        )
        self.feature_autocast_dtype = feature_autocast_dtype or torch.bfloat16

        self.TILE_KEY = "clean_tiles_224" if remove_artifacts else "tiles_224"
        self.FEATURE_KEY = (f"clean_features_{self.foundation_model}" if remove_artifacts else f"features_{self.foundation_model}")
        self.elapsed_time = None

        self.process_slide()

    def _get_tissue_key(self):
        """Return the first non-empty tissue key."""
        for key in self.TISSUE_CANDIDATES:
            if key in self.wsi.shapes:
                tissue = self.wsi.get(key)
                if not is_empty_array(tissue):
                    print(f"Using tissue key: {key}")
                    return key
        raise RuntimeError("No tissue key found")

    def _remove_artifact_tiles(self):
        """Remove tiles that intersect artifact polygons."""
        tiles = self.wsi.shapes.get(self.TILE_KEY)
        artifacts = self.wsi.shapes.get("artifacts_grandqc") # Default segementation key (7x)
        if tiles is None or artifacts is None:
            return
        overlapping = gpd.sjoin(tiles, artifacts, predicate="intersects")
        self.wsi.shapes[self.TILE_KEY] = tiles.drop(index=overlapping.index.unique())

    def process_slide(self):
        """Run validation, tiling, feature extraction, and aggregation."""
        try:
            self.TISSUE_KEY = self._get_tissue_key()

            if self.TILE_KEY not in self.wsi.shapes:
                self.tile_tissue()
                tiles = self.wsi.get(self.TILE_KEY)
                if is_empty_array(tiles):
                    raise RuntimeError("No tiles generated")
    
            self.extract_features()
            self.aggregate_features()
            
            print(f"Processing complete for {self.wsi.path}")
            return 
        
        except Exception as e:
            raise RuntimeError(str(e))
    
    def tile_tissue(self, tile_px=224):
        """Generate tissue tiles and optionally filter out artifact-overlapping tiles."""
        try:
            if self.TILE_KEY not in self.wsi.shapes:
                zs.pp.tile_tissues(self.wsi, tile_px=tile_px, key_added=self.TILE_KEY, tissue_key=self.TISSUE_KEY)
                
            if self.remove_artifacts:
                self._remove_artifact_tiles()
            
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            raise RuntimeError(f"Error during tiling: {e}") from e

    def extract_features(self):
        """ Extract tile embeddings for the foundation model. """
        try:
            if self.FEATURE_KEY not in self.wsi.tables:
                if self.foundation_model == "conch":
                    start_time = datetime.now()
                    zs.tl.feature_extraction(
                        self.wsi,
                        model=self.foundation_model,
                        tile_key=self.TILE_KEY,
                        key_added=self.FEATURE_KEY,
                        device='cpu',
                        num_workers=self.feature_num_workers,
                        )
                    self.elapsed_time = datetime.now() - start_time
                else:                 
                    start_time = datetime.now()
                    zs.tl.feature_extraction(
                        self.wsi,
                        model=self.foundation_model,
                        tile_key=self.TILE_KEY,
                        key_added=self.FEATURE_KEY,
                        device="cuda" if torch.cuda.is_available() else 'cpu',
                        amp=torch.cuda.is_available(),
                        autocast_dtype=self.feature_autocast_dtype,
                        batch_size=self.feature_batch_size,
                        num_workers=self.feature_num_workers,
                        )
                    self.elapsed_time = datetime.now() - start_time
            
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            raise RuntimeError(f"Error during feature extraction: {e}")

    def aggregate_features(self):
        """Aggregate tile features into slide-level features."""
        try:
            zs.tl.feature_aggregation(self.wsi, feature_key=self.FEATURE_KEY, tile_key=self.TILE_KEY)
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            raise RuntimeError(f"Error during feature aggregation: {e}")

    def get_features(self):
        """Return extracted features as AnnData."""
        return self.wsi.fetch.features_anndata(feature_key=self.FEATURE_KEY, tile_key=self.TILE_KEY)

# --------------------
# Batch processing
# --------------------

class ExtractMany:
    """Batch feature extraction with pickle caching and error recovery."""

    def __init__(
        self,
        wsi_paths,
        cache_path,
        zarr_dir,
        foundation_model,
        remove_artifacts=False,
        feature_batch_size=512,
        feature_num_workers=None,
        feature_autocast_dtype=None,
        retry_on_previous_errors=False,
    ):
        self.wsi_paths = wsi_paths
        self.cache_path = cache_path
        self.zarr_dir = zarr_dir
        self.foundation_model = foundation_model
        self.remove_artifacts = remove_artifacts
        self.feature_batch_size = feature_batch_size
        self.feature_num_workers = feature_num_workers
        self.feature_autocast_dtype = feature_autocast_dtype
        self.retry_on_previous_errors = retry_on_previous_errors
        
        # Load cache and track processed slides
        self.cache = _load_cache(cache_path)
        self.processed = _get_processed_entries(cache_path, self.foundation_model)
        
        self.extract_features()

    def extract_features(self):
        """Process all unprocessed slides, saving cache after each result."""
        skipped = 0
        for path in tqdm(self.wsi_paths, desc="feature extraction progress"):
            slide_name = os.path.basename(path)
            abs_path = os.path.abspath(path)
            
            # Check if already processed
            if abs_path in self.processed:
                status = self.processed[abs_path].get("status")
                if status == "complete":
                    print(f"Skipping {slide_name} — already processed")
                    skipped += 1
                    continue
                elif status and status.startswith("error:") and not self.retry_on_previous_errors:
                    print(f"Skipping {slide_name} — previously failed: {status}. "
                          f"Enable retry with retry_on_previous_errors=True.")
                    skipped += 1
                    continue

            print(f"\nProcessing {slide_name}...")

            try:
                wsiobj = ExtractFeatures(
                    path,
                    self.zarr_dir,
                    foundation_model=self.foundation_model,
                    remove_artifacts=self.remove_artifacts,
                    feature_batch_size=self.feature_batch_size,
                    feature_num_workers=self.feature_num_workers,
                    feature_autocast_dtype=self.feature_autocast_dtype,
                )
                features = wsiobj.get_features()
                elapsed_time = wsiobj.elapsed_time
                status = "complete"
                del wsiobj
            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                features = None
                elapsed_time = None
                status = f"error: {str(e)}"
            
            # Store result in cache indexed by model and absolute path
            slide_data = {
                "slide": slide_name,
                "wsi_path": path,
                "features": features,
                "elapsed_time": elapsed_time,
                "status": status,
            }
            self.cache.setdefault(self.foundation_model, {})[abs_path] = slide_data
            _save_cache(self.cache_path, self.cache)
            
            print(f"Saved result for {slide_name} → {self.cache_path}")
            gc.collect()
        
        print(f"\nExtraction complete. Skipped {skipped} already-processed slides.")


if __name__ == "__main__":
    all_slides = [
        "path/to/slide1.mrxs",
        "path/to/slide2.mrxs",
        "path/to/slide3.mrxs",
    ]
    cache_file = "path/to/cache.pkl"
    zarr_dir = "path/to/zarr/dir"

    extractor = ExtractMany(all_slides, cache_file, zarr_dir=zarr_dir, foundation_model="h-optimus-0")