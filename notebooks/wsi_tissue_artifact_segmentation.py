from wsidata import open_wsi
import lazyslide as zs
import os
import pandas as pd
from tqdm import tqdm
import gc
from datetime import datetime
import pickle

def load_big_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return {}

def save_big_cache(big_cache, cache_file):
    tmp_file = cache_file + ".tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(big_cache, f)
    os.replace(tmp_file, cache_file)

def already_processed(big_cache):
    """Return a dict of {(slide_name, category, version): status} for all processed entries."""
    processed = {}
    for slide, categories in big_cache.items():
        for category, versions in categories.items():
            for version, data in versions.items():
                status = data.get("status", "unknown")
                processed[(slide, category, version)] = status
    return processed

def is_empty_array(arr):
        if arr is None:
            return True
        if hasattr(arr, "is_empty"):
            return arr.is_empty.all()
        try:
            return arr.sum() == 0
        except Exception:
            return len(arr) == 0

class SegmentTissue:
    def __init__(self, wsi_path, local_zarr_dir, version="default"):
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(local_zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        os.makedirs(local_zarr_dir, exist_ok=True)
        self.wsi = open_wsi(wsi_path, self.zarr_path) if os.path.exists(self.zarr_path) else open_wsi(wsi_path)
        if version not in ["default", "grandqc", "threshold"]:
            raise ValueError("version must be one of ['default', 'grandqc', 'threshold']")
        self.version = f"tissue_{version}"
        self.TILE_KEY = "tiles_px512_mpp1.5_overlap0.1"
        self.elapsed_time = None
        self.process_slide()

    def process_slide(self):
        self.seg_tissue()
        tissue = self.wsi.get(self.TISSUE_KEY)
        if is_empty_array(tissue):
            raise RuntimeError("No tissue detected")
        self.tile_tissue()
        tiles = self.wsi.get(self.TILE_KEY)
        if is_empty_array(tiles):
            raise RuntimeError("No tiles generated")
        print(f"Tissue processing complete for {self.wsi.path}")

    def seg_tissue(self):
        if self.version == "tissue_default":
            self.TISSUE_KEY = 'tissue_default'
            self._seg_default()
        elif self.version == "tissue_grandqc":
            self.TISSUE_KEY = 'tissue_grandqc'
            self._seg_grandqc()
        elif self.version == "tissue_threshold":
            self.TISSUE_KEY = 'tissue_threshold'
            self._seg_threshold()

    def _seg_default(self):
        try: 
            if self.TISSUE_KEY in self.wsi.shapes:
                print(f"Tissue already segmented: {self.TISSUE_KEY}")
                return
            start = datetime.now()
            zs.pp.find_tissues(self.wsi, key_added=self.TISSUE_KEY)
            self.elapsed_time = datetime.now() - start
            self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during default tissue segmentation: {e}")

    def _seg_grandqc(self):
        try: 
            if self.TISSUE_KEY in self.wsi.shapes:
                print(f"Tissue already segmented: {self.TISSUE_KEY}")
                return
            start = datetime.now()
            zs.seg._tissue.tissue(self.wsi, model='grandqc', key_added=self.TISSUE_KEY, device="cpu")
            self.elapsed_time = datetime.now() - start
            self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during grandqc tissue segmentation: {e}")

    def _seg_threshold(self):
        try: 
            if self.TISSUE_KEY in self.wsi.shapes:
                print(f"Tissue already segmented: {self.TISSUE_KEY}")
                return
            start = datetime.now()
            zs.pp.find_tissues(self.wsi, threshold=5, to_hsv=True, filter_artifacts=False, key_added=self.TISSUE_KEY)
            self.elapsed_time = datetime.now() - start
            self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during threshold tissue segmentation: {e}")

    def tile_tissue(self, tile_px=512, mpp=1.5, overlap=0.1):
        if self.TILE_KEY not in self.wsi.shapes:
            zs.pp.tile_tissues(self.wsi, tile_px=tile_px, mpp=mpp, overlap=overlap, key_added=self.TILE_KEY, tissue_key=self.TISSUE_KEY)
            self.wsi.write(self.zarr_path)

    def get_full_tissue_area(self):
        return self.wsi.shapes[self.TISSUE_KEY].area.sum()


class SegmentArtifacts:
    def __init__(self, wsi_path:str, local_zarr_dir:str, version="default"):
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(local_zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        if os.path.exists(self.zarr_path):
            self.wsi = open_wsi(wsi_path, self.zarr_path)
        else:
            self.wsi = open_wsi(wsi_path)
            os.makedirs(local_zarr_dir, exist_ok=True)
        if version not in ["default", "10x"]:
            raise ValueError("version must be one of ['default', '10x']")
        self.version = f"artifact_{version}"
        self.TILE_KEY = "tiles_px512_mpp1.5_overlap0.1" if version=="default" else "tiles_px512_mpp1.0_overlap0.2"
        self.ARTIFACT_KEY = 'artifacts_grandqc' if version=="default" else 'artifacts_grandqc_10x'
        self.elapsed_time = None
        self.process_slide()

    def find_tissue_key(self):
        keys = [k for k in ['tissue_default','tissue_grandqc','tissue_threshold'] if k in self.wsi.shapes]
        if not keys:
            raise KeyError("No tissue key found")
        return keys

    def tile_tissue_10x(self):
        zs.pp.tile_tissues(self.wsi, 512, overlap=0.2, mpp=1.0, key_added=self.TILE_KEY)

    def process_slide(self):
        try:
            tissue_keys = self.find_tissue_key()
            for key in tissue_keys:
                tissue = self.wsi.get(key)
                if not is_empty_array(tissue):
                    self.TISSUE_KEY = key
                    break
            
            if self.TILE_KEY not in self.wsi.shapes and self.version == "artifact_10x":
                self.tile_tissue_10x()
            
            self._seg_artifacts()
            artifacts = self.wsi.get(self.ARTIFACT_KEY)
            if is_empty_array(artifacts):
                raise RuntimeError("No artifacts detected")
            print(f"Artifact processing complete for {self.wsi.path}")
        except Exception as e:
            raise RuntimeError(f"Error processing slide: {e}")

    def _seg_artifacts(self):
        try:
            if self.ARTIFACT_KEY not in self.wsi.shapes:
                start = datetime.now()
                if self.version == "artifact_default":
                    zs.seg._artifact.artifact(self.wsi, tissue_key=self.TISSUE_KEY, tile_key=self.TILE_KEY, key_added=self.ARTIFACT_KEY)
                else:
                    zs.seg._artifact.artifact(self.wsi, tissue_key=self.TISSUE_KEY, tile_key=self.TILE_KEY, variant="10x", key_added=self.ARTIFACT_KEY)
                self.elapsed_time = datetime.now() - start
                self.wsi.write(self.zarr_path)
        except Exception as e:
            raise RuntimeError(f"Error during artifact segmentation: {e}")

    def get_artifacts(self):
        return self.wsi.shapes[self.ARTIFACT_KEY]

    def get_full_tissue_area(self):
        return self.wsi.shapes[self.TISSUE_KEY].area.sum()

    def get_artifact_dataframe(self):
        artifacts = self.get_artifacts()
        artifact_df = artifacts.groupby(["class"], as_index=True).agg(
            area=("geometry", lambda x: x.area.sum()),
            count=("geometry", "size")
        )
        full_area = self.get_full_tissue_area()
        artifact_df["percentage"] = (artifact_df["area"] / full_area) * 100
        return artifact_df

    def get_artifact_percentage(self):
        artifact_df = self.get_artifact_dataframe()
        artifact_sum = artifact_df["area"].sum()
        total_area = self.get_full_tissue_area()
        percentage = (artifact_sum / total_area) * 100
        return percentage
    
    def view_artifacts(self):
        viewer = zs.pl.WSIViewer(self.wsi)
        viewer.add_image('thumbnail')
        viewer.add_polygons(self.ARTIFACT_KEY, color_by='class', alpha=0.5)
        viewer.add_contours(self.TISSUE_KEY)
        viewer.show()


class SegmentMany:
    def __init__(self, wsi_paths, cache_file, local_zarr_dir, segmentation_type, version="default"):
        self.wsi_paths = wsi_paths
        self.cache_file = cache_file
        self.local_zarr_dir = local_zarr_dir
        if segmentation_type not in ["tissue", "artifact"]:
            raise ValueError("segmentation_type must be one of ['tissue', 'artifact']")
        self.segmentation_type = segmentation_type
        self.version = version
        self.full_version = f"tissue_{version}" if segmentation_type=="tissue" else f"artifact_{version}"
        self.big_cache = load_big_cache(cache_file)
        self.processed = already_processed(self.big_cache)
        self.run_segmentation()

    def run_segmentation(self):
        for path in tqdm(self.wsi_paths, desc=f"{self.segmentation_type} segmentation"):
            slide_name = os.path.basename(path)
            category = self.segmentation_type
            version = self.version
            key = (slide_name, category, version)

            if key not in self.processed:
                print("MISS:", key)
                print("Closest processed keys:", list(self.processed.keys())[:5])
            
            if key in self.processed:
                status = self.processed[key]
                if status == "complete":
                    print(f"Skipping {slide_name} ({category} {version}) — already successfully processed")
                    continue
                elif status.startswith("error:"):
                    print(f"Skipping {slide_name} ({category} {version}) — previously failed: {status}")
                    continue

            try:
                if category == "tissue":
                    wsiobj = SegmentTissue(path, self.local_zarr_dir, version=self.version)
                    slide_data = {
                        "size": wsiobj.get_full_tissue_area(),
                        "elapsed_time": wsiobj.elapsed_time,
                        "status": "complete"
                    }
                elif category == "artifact":
                    wsiobj = SegmentArtifacts(path, self.local_zarr_dir, version=self.version)
                    slide_data = {
                        "pct": wsiobj.get_artifact_percentage(),
                        "df": wsiobj.get_artifact_dataframe(),
                        "elapsed_time": wsiobj.elapsed_time,
                        "status": "complete"
                    }

                self.big_cache.setdefault(slide_name, {}).setdefault(category, {})[version] = slide_data
                save_big_cache(self.big_cache, self.cache_file)
                del wsiobj

            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                slide_data = {"status": f"error: {str(e)}"}
                self.big_cache.setdefault(slide_name, {}).setdefault(category, {})[version] = slide_data
                save_big_cache(self.big_cache, self.cache_file)


if __name__ == "__main__":
    slide_paths = [
        "path/to/wsi1.mrxs",
        "path/to/wsi2.mrxs"
    ]
    local_zarr_dir = "path/to/local_zarr"
    cache_file = "all_slides_cache.pkl"

    # Tissue segmentation
    SegmentMany(slide_paths, cache_file, local_zarr_dir, segmentation_type="tissue", version="threshold")

    # Artifact segmentation
    SegmentMany(slide_paths, cache_file, local_zarr_dir, segmentation_type="artifact", version="default")