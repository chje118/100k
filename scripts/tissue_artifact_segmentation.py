from wsidata import open_wsi
import lazyslide as zs
import os
from tqdm import tqdm
from datetime import datetime
import pickle
import shutil

# --------------------
# Cache handling
# --------------------

def _load_cache(cache_file):
    """Load segmentation cache from pickle file. Returns empty dict if file doesn't exist."""
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return {}

def _save_cache(data, cache_file):
    """Atomically save segmentation cache to pickle file using temp file."""
    tmp_file = cache_file + ".tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_file, cache_file)

def _get_processed_entries(cache):
    """Extract all processed (slide, category, version) entries from cache with their statuses.
    
    Returns:
        dict: {(slide_name, category, version): status}
    """
    processed = {}
    for slide, categories in cache.items():
        for category, versions in categories.items():
            for version, data in versions.items():
                status = data.get("status", "unknown")
                processed[(slide, category, version)] = status
    return processed

def _remove_zarr_path(zarr_path):
    """Safely remove Zarr directory or file."""
    if os.path.isdir(zarr_path):
        shutil.rmtree(zarr_path)
    elif os.path.exists(zarr_path):
        os.remove(zarr_path)

def validate_zarr_caches(wsi_paths, zarr_dir):
    """Validate and recover Zarr caches for all WSI paths. Removes corrupted caches."""
    for path in wsi_paths:
        zarr_path = os.path.join(zarr_dir, os.path.basename(path).replace(".mrxs", ".zarr"))
        if not os.path.exists(zarr_path):
            continue
        try:
            open_wsi(path, zarr_path)
            print(f"Opened WSI with zarr cache: {zarr_path}")
            continue
        except Exception as e:
            print(f"Existing zarr at {zarr_path} failed to open: {e}; removing it.")
            _remove_zarr_path(zarr_path)
            continue

def is_empty_array(arr) -> bool:
    """Check if array/geometry collection is empty. Handles None, empty geometries, and zero arrays."""
    if arr is None:
        return True
    if hasattr(arr, "is_empty"):
        return arr.is_empty.all()
    try:
        return arr.sum() == 0
    except Exception:
        return len(arr) == 0

# --------------------
# Tissue Segmentation
# --------------------
 
class SegmentTissue:
    """
    Segment tissue regions from WSI.

    Supports multiple tissue segmentation methods:
    - 'default': Basic morphological finding
    - 'grandqc': Deep learning-based segmentation
    - 'threshold': Threshold-based with HSV conversion (threshold=5)
    """
    VALID_VERSIONS = ["default", "grandqc", "threshold"]
    
    def __init__(self, wsi_path, zarr_dir, version="default"):
        if version not in self.VALID_VERSIONS:
            raise ValueError(f"version must be one of {self.VALID_VERSIONS}")
        
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        os.makedirs(zarr_dir, exist_ok=True)
        self.wsi = open_wsi(wsi_path, self.zarr_path) if os.path.exists(self.zarr_path) else open_wsi(wsi_path)
        self.version = version
        self.tissue_key = f"tissue_{version}"
        self.elapsed_time = None
        self.process_slide()

    def process_slide(self):
        """Main processing pipeline: segment tissue and validate."""
        self._segment_tissue()
        tissue = self.wsi.get(self.tissue_key)
        if is_empty_array(tissue):
            raise RuntimeError("No tissue detected")
        print(f"Tissue processing complete for {self.wsi.path}")

    def _segment_tissue(self):
        """Apply tissue segmentation using configured method."""
        if self.tissue_key in self.wsi.shapes:
            print(f"Tissue already segmented: {self.tissue_key}")
            return
        
        start = datetime.now()
        try:
            if self.version == "default":
                zs.pp.find_tissues(self.wsi, key_added=self.tissue_key)
            elif self.version == "grandqc":
                zs.seg._tissue.tissue(self.wsi, model='grandqc', key_added=self.tissue_key, device="cpu")
            elif self.version == "threshold":
                zs.pp.find_tissues(self.wsi, threshold=5, to_hsv=True, filter_artifacts=False, key_added=self.tissue_key)
            
            self.elapsed_time = datetime.now() - start
            self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during {self.version} tissue segmentation: {e}")

    def get_full_tissue_area(self):
        """Get total tissue area in pixels."""
        return self.wsi.shapes[self.tissue_key].area.sum()

# --------------------
# Artifact Segmentation
# --------------------

class SegmentArtifacts:
    """ Detect and classify artifacts (air bubbles, out of focus, etc.) in tissue tiles.
        
    Supports two detection variants:
    - 'default': Standard artifact detection (512px tiles, 1.5 mpp, 10% overlap)
    - '10x': Alternative for 10x magnification (512px tiles, 1.0 mpp, 10% overlap)
    """
    VALID_VERSIONS = ["default", "10x"]
    
    def __init__(self, wsi_path: str, zarr_dir: str, version="default"):
        if version not in self.VALID_VERSIONS:
            raise ValueError(f"version must be one of {self.VALID_VERSIONS}")
        
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        self.wsi = open_wsi(wsi_path, self.zarr_path) if os.path.exists(self.zarr_path) else open_wsi(wsi_path)
        os.makedirs(zarr_dir, exist_ok=True)
        
        self.version = version
        self.tile_key = "tiles_px512_mpp1.5_overlap0.1" if version == "default" else "tiles_px512_mpp1.0_overlap0.1"
        self.artifact_key = "artifacts_grandqc" if version == "default" else "artifacts_grandqc_10x"
        self.tissue_key = None
        self.elapsed_time = None
        self.process_slide()

    def _find_tissue_key(self):
        """Locate available tissue segmentation key."""
        tissue_candidates = [key for key in ['tissue_default', 'tissue_grandqc', 'tissue_threshold'] if key in self.wsi.shapes]
        if not tissue_candidates:
            raise KeyError("No tissue key found. ")
        return tissue_candidates

    def _tile_tissue(self, tile_px=512, mpp=1.5, overlap=0.1):
        """ Generate tiles from tissue regions. 
        Default: tile_px=512, mpp=1.5, overlap=0.1 for GrandQC 7x.
        """
        if self.tile_key not in self.wsi.shapes:
            zs.pp.tile_tissues(self.wsi, tile_px=tile_px, mpp=mpp, overlap=overlap, 
                              key_added=self.tile_key, tissue_key=self.tissue_key)
            self.wsi.write(self.zarr_path)

    def process_slide(self):
        """Main pipeline: validate tissue, generate tiles, detect artifacts."""
        try:
            tissue_keys = self._find_tissue_key()
            for key in tissue_keys:
                tissue = self.wsi.get(key)
                if not is_empty_array(tissue):
                    self.tissue_key = key
                    break
            
            if self.tile_key not in self.wsi.shapes:
                if self.version == "10x":
                    self._tile_tissue(tile_px=512, mpp=1.0, overlap=0.1)
                else:
                    self._tile_tissue(tile_px=512, mpp=1.5, overlap=0.1)
            
            self._segment_artifacts()
            artifacts = self.wsi.get(self.artifact_key)
            if is_empty_array(artifacts):
                raise RuntimeError("No artifacts detected")
            print(f"Artifact processing complete for {self.wsi.path}")
        except Exception as e:
            raise RuntimeError(f"Error processing slide: {e}")

    def _segment_artifacts(self):
        """Run artifact detection model."""
        try:
            if self.artifact_key not in self.wsi.shapes:
                start = datetime.now()
                if self.version == "default":
                    zs.seg._artifact.artifact(self.wsi, tile_key=self.tile_key, key_added=self.artifact_key)
                else:
                    zs.seg._artifact.artifact(self.wsi, tile_key=self.tile_key, variant="10x", key_added=self.artifact_key)
                self.elapsed_time = datetime.now() - start
                self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during artifact segmentation: {e}")

    def get_artifacts(self):
        """Get artifact geometries."""
        return self.wsi.shapes[self.artifact_key]

    def get_full_tissue_area(self):
        """Get total tissue area in pixels."""
        return self.wsi.shapes[self.tissue_key].area.sum()

    def get_artifact_dataframe(self):
        """Get artifact statistics grouped by class."""
        artifacts = self.get_artifacts()
        artifact_df = artifacts.groupby(["class"], as_index=True).agg(
            area=("geometry", lambda x: x.area.sum()),
            count=("geometry", "size")
        )
        full_area = self.get_full_tissue_area()
        artifact_df["percentage"] = (artifact_df["area"] / full_area) * 100
        return artifact_df

    def get_artifact_percentage(self):
        """Get total artifact coverage as percentage of tissue."""
        artifact_df = self.get_artifact_dataframe()
        artifact_sum = artifact_df["area"].sum()
        total_area = self.get_full_tissue_area()
        return (artifact_sum / total_area) * 100
    
    def view_artifacts(self):
        """Visualize artifacts overlaid on tissue."""
        viewer = zs.pl.WSIViewer(self.wsi)
        viewer.add_image('thumbnail')
        viewer.add_polygons(self.artifact_key, color_by='class', alpha=0.5)
        viewer.add_contours(self.tissue_key)
        viewer.show()

# --------------------
# Batch processing
# --------------------

class SegmentMany:
    """ Batch processing of tissue and artifact segmentation for multiple WSI with checkpointing. """
    
    def __init__(self, wsi_paths, cache_path, zarr_dir, segmentation_type, version="default", retry_on_previos_errors=False):
        if segmentation_type not in ["tissue", "artifact"]:
            raise ValueError("segmentation_type must be 'tissue' or 'artifact'")
        
        self.wsi_paths = wsi_paths
        self.cache_path = cache_path
        self.zarr_dir = zarr_dir
        self.segmentation_type = segmentation_type
        self.version = version
        self.retry_on_previos_errors = retry_on_previos_errors
        self.version_key = f"{segmentation_type}_{version}"
        self.cache = _load_cache(cache_path)
        self.processed = _get_processed_entries(self.cache)
        self.run_segmentation()

    def run_segmentation(self):
        """Process all slides with progress tracking."""
        for path in tqdm(self.wsi_paths, desc=f"{self.segmentation_type} segmentation"):
            slide_name = os.path.basename(path)
            cache_key = (slide_name, self.segmentation_type, self.version)

            if cache_key in self.processed and self.processed[cache_key] == "complete":
                print(f"Skipping {slide_name} — already processed")
                continue

            if cache_key in self.processed and self.processed[cache_key].startswith("error:") and not self.retry_on_previos_errors:
                print(f"Skipping {slide_name} — previously failed: {self.processed[cache_key]}. Enable retry with retry_on_previos_errors=True.")
                continue

            try:
                if self.segmentation_type == "tissue":
                    wsiobj = SegmentTissue(path, self.zarr_dir, version=self.version)
                    slide_data = {
                        "size": wsiobj.get_full_tissue_area(),
                        "elapsed_time": wsiobj.elapsed_time,
                        "status": "complete"
                    }
                elif self.segmentation_type == "artifact":
                    wsiobj = SegmentArtifacts(path, self.zarr_dir, version=self.version)
                    slide_data = {
                        "pct": wsiobj.get_artifact_percentage(),
                        "df": wsiobj.get_artifact_dataframe(),
                        "elapsed_time": wsiobj.elapsed_time,
                        "status": "complete"
                    }

                self.cache.setdefault(slide_name, {}).setdefault(self.segmentation_type, {})[self.version] = slide_data
                _save_cache(self.cache, self.cache_path)
                del wsiobj

            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                slide_data = {"status": f"error: {str(e)}"}
                self.cache.setdefault(slide_name, {}).setdefault(self.segmentation_type, {})[self.version] = slide_data
                _save_cache(self.cache, self.cache_path)


if __name__ == "__main__":
    slide_paths = [
        "path/to/wsi1.mrxs",
        "path/to/wsi2.mrxs"
    ]
    zarr_dir = "path/to/zarr/directory"
    cache_file = "path/to/cache.pkl"

    # Tissue segmentation
    SegmentMany(slide_paths, cache_file, zarr_dir, segmentation_type="tissue", version="threshold")
    
    # Artifact segmentation
    SegmentMany(slide_paths, cache_file, zarr_dir, segmentation_type="artifact", version="10x")