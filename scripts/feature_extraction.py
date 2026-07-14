import os
from datetime import datetime
import geopandas as gpd
import pandas as pd
import torch
from tqdm import tqdm
import lazyslide as zs
import gc
from tissue_artifact_segmentation import is_empty_array
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

    def _find_tissue_key(self):
        """Locate available tissue segmentation key."""
        tissue_candidates = [key for key in ['tissue_default', 'tissue_grandqc', 'tissue_threshold'] if key in self.wsi.shapes]
        if not tissue_candidates:
            raise KeyError("No tissue key found. ")
        return tissue_candidates

    def process_slide(self):
        try:
            tissue_keys = self._find_tissue_key()
            for key in tissue_keys:
                tissue = self.wsi.get(key)
                if not is_empty_array(tissue):
                    self.tissue_key = key
                    print(f"Using tissue key: {self.tissue_key}")
                    break
            else:
                raise RuntimeError(f"No tissue key found")


            if self.TILE_KEY not in self.wsi.shapes:
                self.tile_tissue()
                tiles = self.wsi.get(self.TILE_KEY)
                if self.is_empty_array(tiles):
                    raise RuntimeError(f"No tiles generated")
                
            self.extract_features()
            self.aggregate_features()
            
            print(f"Processing complete for {self.wsi.path}")
            return True
        
        except Exception as e:
            raise RuntimeError(str(e))
    
    def tile_tissue(self, tile_px=224):
        try:
            if self.TILE_KEY not in self.wsi.shapes:
                zs.pp.tile_tissues(self.wsi, tile_px=tile_px, key_added=self.TILE_KEY, tissue_key=self.TISSUE_KEY)
                
            if self.remove_artifacts:
                tiles = self.wsi.shapes[self.TILE_KEY]
                artifacts = self.wsi.shapes.get("artifacts_grandqc")
                if tiles is not None and artifacts is not None:
                    overlapping = gpd.sjoin(tiles, artifacts, predicate="intersects")
                    clean_tiles = tiles.drop(index=overlapping.index.unique())
                    self.wsi.shapes[self.TILE_KEY] = clean_tiles
            
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            print(f"Error while extracting tiles: {e}")
            raise            

    def extract_features(self):
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
                        num_workers=self.feature_num_workers,  # Overlap host preprocessing with GPU work
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
                        batch_size=self.feature_batch_size,  # Larger batches keep the H100 busy
                        num_workers=self.feature_num_workers,  # Overlap host preprocessing with GPU work
                        )
                    self.elapsed_time = datetime.now() - start_time
            
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            raise RuntimeError(f"Error during feature extraction: {e}")

    def aggregate_features(self):
        try:
            zs.tl.feature_aggregation(self.wsi, feature_key=self.FEATURE_KEY, tile_key=self.TILE_KEY)
            self.wsi.write(self.zarr_path)
            return self.wsi
        
        except Exception as e:
            raise RuntimeError(f"Error during feature aggregation: {e}")

    def get_features(self):
        return self.wsi.fetch.features_anndata(feature_key=self.FEATURE_KEY, tile_key=self.TILE_KEY)


class ExtractMany:
    def __init__(
        self,
        wsi_paths,
        output_path,
        local_zarr_dir,
        model,
        remove_artifacts=False,
        feature_batch_size=512,
        feature_num_workers=None,
        feature_autocast_dtype=None,
    ):
        self.wsi_paths = wsi_paths
        self.local_zarr_dir = local_zarr_dir
        self.model = model
        self.remove_artifacts = remove_artifacts
        self.csv_path = output_path
        
        # Configure feature extraction parameters, with sensible defaults for H100
        self.feature_batch_size = feature_batch_size
        self.feature_num_workers = feature_num_workers
        self.feature_autocast_dtype = feature_autocast_dtype
        
        self.extract_features()

    def already_processed(self):
        if not os.path.exists(self.csv_path):
            return set()
        try:
            df_existing = pd.read_csv(self.csv_path, usecols=["wsi_path", "model"])
            df_existing = df_existing[df_existing["model"] == self.model]

            return {os.path.abspath(str(p)) for p in df_existing["wsi_path"].dropna()}
        except Exception:
            return set()

    def extract_features(self):
        processed_slides = self.already_processed()
        remaining_paths = [p for p in self.wsi_paths if os.path.abspath(p) not in processed_slides]

        print(f"Found {len(processed_slides)} slides already processed.")
        print(f"{len(remaining_paths)} remaining to process.")

        for path in tqdm(self.wsi_paths, desc="feature extraction progress"):
            slide_name = os.path.basename(path)
            print(f"\nProcessing {slide_name}...")
            try:
                wsiobj = ExtractFeatures(
                    path,
                    self.local_zarr_dir,
                    model=self.model,
                    remove_artifacts=self.remove_artifacts,
                    feature_batch_size=self.feature_batch_size,
                    feature_num_workers=self.feature_num_workers,
                    feature_autocast_dtype=self.feature_autocast_dtype,
                )
                features = wsiobj.get_features()
                elapsed_time = wsiobj.elapsed_time
                status = "feature extraction complete"
                del wsiobj
            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                features = None
                elapsed_time = None
                status = f"error: {str(e)}"

            new_row = pd.DataFrame([{
                "slide": slide_name,
                "wsi_path": path,
                "features": features,
                "elapsed_time": elapsed_time,
                "status": status,
                "model": self.model
            }])

            header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            new_row.to_csv(self.csv_path, mode='a', header=header, index=False)
            print(f"Appended result for {slide_name} → {self.csv_path}")
            
            gc.collect()



if __name__ == "__main__":
    df_path = "path/to/csv"
    df_subset = pd.read_csv(df_path)
    all_filenames = list(df_subset['wsi filenames'])

    output_dir = "path/to/output"
    local_dir = "path/to/local/zarr/dir"

    extractor = ExtractMany(all_filenames, output_dir, local_zarr_dir=local_dir, model="h-optimus-0")