from wsidata import open_wsi
import lazyslide as zs
import geopandas as gpd
import os
import pandas as pd
from tqdm import tqdm
import gc
from datetime import datetime

class SegmentTissue:
    def __init__(self, wsi_path, local_zarr_dir, try_grandqc = False):
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(local_zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        if os.path.exists(self.zarr_path):
            self.wsi = open_wsi(wsi_path, self.zarr_path)
        else:
            self.wsi = open_wsi(wsi_path)
            os.makedirs(local_zarr_dir, exist_ok=True)
        self.try_grandqc = try_grandqc
        self.TILE_KEY = "tiles_px512_mpp1.5_overlap0.1"
        self.process_slide()

    def is_empty_array(self, arr):
        if arr is None:
            return True
        if hasattr(arr, "is_empty"):
            return arr.is_empty.all()
        try:
            return arr.sum() == 0
        except Exception:
            return len(arr) == 0
    
    def process_slide(self):
        try:
            slide_name = os.path.basename(self.wsi_path)
            self.seg_tissue()
            tissue = self.wsi.get(self.TISSUE_KEY)
            if self.is_empty_array(tissue):
                raise RuntimeError(f"No tissue detected")

            self.tile_tissue()
            tiles = self.wsi.get(self.TILE_KEY)
            if self.is_empty_array(tiles):
                raise RuntimeError(f"No tiles generated")

            print(f"Processing complete for {self.wsi.path}")
            return True
        
        except Exception as e:
            raise RuntimeError(str(e))
    
    def seg_tissue(self):      
        default_key = 'tissue_default'
        grandqc_key = 'tissue_grandqc'
        self.elapsed_time = None

        if default_key in self.wsi.shapes:
            self.TISSUE_KEY = default_key
        elif grandqc_key in self.wsi.shapes:
            self.TISSUE_KEY = grandqc_key
        else:
            start_time = datetime.now()
            zs.pp.find_tissues(self.wsi, key_added=default_key)
            end_time = datetime.now()
            self.elapsed_time = end_time - start_time
            self.TISSUE_KEY = default_key

            tissue = self.wsi.get(default_key)
            if self.is_empty_array(tissue):
                print("Default tissue detection produced empty tissue.")
                if self.try_grandqc:
                    print("Trying GrandQC...")
            
                    start_time = datetime.now()
                    zs.seg._tissue.tissue(self.wsi, model='grandqc', key_added=grandqc_key, device="cpu") # TODO START GPU FALL BACK TO CPU (ISSUES RUNNING ON GPU)
                    end_time = datetime.now()
                    self.elapsed_time = end_time - start_time
                    self.TISSUE_KEY = grandqc_key

                    tissue = self.wsi.get(grandqc_key)
                    if self.is_empty_array(tissue):
                        print("GrandQC tissue detection also produced empty tissue")
                        raise RuntimeError("No tissue detected")

        print(f"Tissue detection successful using: {self.TISSUE_KEY}")
        self.wsi.write(self.zarr_path)
        return self.wsi
   
    def tile_tissue(self, tile_px=512, mpp=1.5, overlap=0.1):
        """ GrandQC trained on px=512, and mpp = 1 / 1.5 / 2 """
        try:
            
            if self.TILE_KEY not in self.wsi.shapes:
                zs.pp.tile_tissues(self.wsi, tile_px=tile_px, mpp=mpp, overlap=overlap, key_added=self.TILE_KEY, tissue_key=self.TISSUE_KEY)
                self.wsi.write(self.zarr_path)
            return self.wsi
        except Exception as e:
            print(f"Error in tile_tissue(): {e}")
            raise

    def get_full_tissue_area(self):
        return self.wsi.shapes[self.TISSUE_KEY].area.sum()


class SegmentArtifacts:
    def __init__(self, wsi_path, local_zarr_dir):
        self.wsi_path = wsi_path
        self.zarr_path = os.path.join(local_zarr_dir, os.path.basename(wsi_path).replace(".mrxs", ".zarr"))
        if os.path.exists(self.zarr_path):
            self.wsi = open_wsi(wsi_path, self.zarr_path)
        else:
            self.wsi = open_wsi(wsi_path)
            os.makedirs(local_zarr_dir, exist_ok=True)
        self.TILE_KEY = "tiles_px512_mpp1.5_overlap0.1"
        self.ARTIFACT_KEY = 'artifacts_grandqc'
        self.process_slide()

    def is_empty_array(self, arr):
        if arr is None:
            return True
        if hasattr(arr, "is_empty"):
            return arr.is_empty.all()
        try:
            return arr.sum() == 0
        except Exception:
            return len(arr) == 0
    
    def process_slide(self):
        try:
            slide_name = os.path.basename(self.wsi_path)

            default_key = 'tissue_default'
            grandqc_key = 'tissue_grandqc'
            
            if default_key in self.wsi.shapes:
                self.TISSUE_KEY = default_key
            elif grandqc_key in self.wsi.shapes:
                self.TISSUE_KEY = grandqc_key
            else:
                raise RuntimeError(f"No tissue key found")
            
            tissue = self.wsi.get(self.TISSUE_KEY)
            if self.is_empty_array(tissue):
                tissue = self.wsi.get(grandqc_key)
                if self.is_empty_array(tissue):
                    raise RuntimeError(f"No tissue detected")

            if self.TILE_KEY not in self.wsi.shapes:
                raise RuntimeError(f"No tile key found")
            
            tiles = self.wsi.get(self.TILE_KEY)
            if self.is_empty_array(tiles):
                raise RuntimeError(f"No tiles generated")

            self.seg_artifacts()
            artifacts = self.wsi.get(self.ARTIFACT_KEY)
            if self.is_empty_array(artifacts):
                raise RuntimeError(f"No artifacts detected")

            print(f"Processing complete for {self.wsi.path}")
            return True
        
        except Exception as e:
            raise RuntimeError(str(e))

    def seg_artifacts(self):
        self.elapsed_time = None
        try:
            if self.ARTIFACT_KEY not in self.wsi.shapes:
                start_time = datetime.now()
                zs.seg._artifact.artifact(self.wsi, tile_key=self.TILE_KEY, key_added=self.ARTIFACT_KEY)
                end_time = datetime.now()
                self.elapsed_time = end_time - start_time
                self.wsi.write(self.zarr_path)
            return self.wsi
        except Exception as e:
            raise RuntimeError(f"Error during artifact segmentaion: {e}")
        
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
    def __init__(self, wsi_paths, output_dir, local_zarr_dir, segmentation_type, try_grandqc=False):
        self.wsi_paths = wsi_paths
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.local_zarr_dir = local_zarr_dir
        self.segmentation_type = segmentation_type # TODO MUST BE EITHER TISSUE OR ARTIFACT
        self.try_grandqc = try_grandqc
        self.csv_path = os.path.join(self.output_dir, f"{self.segmentation_type}_summary.csv")
        self.run_segmentation()

    def already_processed(self):
        if not os.path.exists(self.csv_path):
            return set()
        try:
            df_existing = pd.read_csv(self.csv_path, usecols=["wsi_path"])
            return set(os.path.abspath(p) for p in df_existing["wsi_path"].astype(str))
        except Exception:
            return set()

    def run_segmentation(self):
        processed_slides = self.already_processed()
        remaining_paths = [p for p in self.wsi_paths if os.path.abspath(p) not in processed_slides]

        print(f"Found {len(processed_slides)} slides already processed.")
        print(f"{len(remaining_paths)} remaining to process.")

        for path in tqdm(remaining_paths, desc=f"{self.segmentation_type} segmentation progress"):
            slide_name = os.path.basename(path)
            print(f"\nProcessing {slide_name}...")
            try:
                if self.segmentation_type == "artifact":
                    wsiobj = SegmentArtifacts(path, self.local_zarr_dir)
                    artifact_percentage = wsiobj.get_artifact_percentage()
                    artifact_df = wsiobj.get_artifact_dataframe()
                elif self.segmentation_type == "tissue":
                    wsiobj = SegmentTissue(path, self.local_zarr_dir, try_grandqc=self.try_grandqc)
                    artifact_percentage = None
                    artifact_df = None
                tissue_size = wsiobj.get_full_tissue_area()
                elapsed_time = wsiobj.elapsed_time
                status = f"{self.segmentation_type} segmentation complete"
                del wsiobj
            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                tissue_size = None
                elapsed_time = None
                artifact_percentage = None
                artifact_df = None
                status = f"error: {str(e)}"

            new_row = pd.DataFrame([{
                "slide": slide_name,
                "wsi_path": path,
                "tissue_size": tissue_size,
                "elapsed_time": elapsed_time,
                "artifact_percentage": artifact_percentage,
                "artifact_df": artifact_df,
                "status": status
            }])

            header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            new_row.to_csv(self.csv_path, mode='a', header=header, index=False)
            print(f"Appended result for {slide_name} → {self.csv_path}")
            
            gc.collect()


# TODO Implement
# SEGMENT_CLASSES = {
#    "tissue": SegmentTissue,
#    "artifact": SegmentArtifacts
# }
# segmenter_class = SEGMENT_CLASSES[self.segmentation_type]
# wsiobj = segmenter_class(path, self.local_zarr_dir, try_grandqc=self.try_grandqc)

# Example usage
if __name__ == "__main__":
    # Single slide QC
    slide_paths = [
        "path/to/wsi",
        "path/to/another/wsi"
    ]
    output_dir = "path/to/output"
    local_dir = "path/to/local_zarr_storage"
    wsi_tissue = SegmentMany(slide_paths, output_dir, local_dir, "tissue", try_grandqc=False)
    