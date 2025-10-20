from wsidata import open_wsi
import lazyslide as zs
import geopandas as gpd
from wsi_metadata import WSIMetadata
import os
import pandas as pd
from tqdm import tqdm
import gc

class SingleQC:
    TISSUE_KEY = 'tissue_qc'
    TILE_KEY = 'tiles_qc'
    ARTIFACT_KEY = 'artifacts'
    
    def __init__(self, wsi_path):
        self.wsi_path = wsi_path
        self.wsi = open_wsi(wsi_path)
        self.seg_tissue()
        self.tile_tissue()
        self.seg_artifacts()
        self.gdf = self.artifact_gdf()
        self.adf = self.artifact_dataframe()
        
    def seg_tissue(self):
        if self.TISSUE_KEY not in self.wsi.shapes:
            zs.seg._tissue.tissue(self.wsi, model='grandqc', key_added=self.TISSUE_KEY)
            self.wsi.write()
        return self.wsi
    
    def tile_tissue(self, tile_px=512, mpp=1.5):
        """ GrandQC trained on px=512, and mpp = 1 / 1.5 / 2 """
        if self.TILE_KEY not in self.wsi.shapes:
            zs.pp.tile_tissues(self.wsi, tile_px=tile_px, mpp=mpp, key_added=self.TILE_KEY, tissue_key=self.TISSUE_KEY)
            self.wsi.write()
        return self.wsi
    
    def seg_artifacts(self):
        if self.ARTIFACT_KEY not in self.wsi.shapes:
            zs.seg._artifact.artifact(self.wsi, tile_key=self.TILE_KEY, key_added=self.ARTIFACT_KEY)
            self.wsi.write()
        return self.wsi

    def artifact_gdf(self):
        return self.wsi.shapes[self.ARTIFACT_KEY]

    def get_full_tissue_area(self):
        return self.wsi.shapes[self.TISSUE_KEY].area.sum()

    def artifact_dataframe(self):
        adf = self.gdf.groupby(["class"], as_index=True).agg(
            area=("geometry", lambda x: x.area.sum()),
            count=("geometry", "size")
        )
        full_area = self.get_full_tissue_area()
        adf["percentage"] = (adf["area"] / full_area) * 100
        return adf

    def get_artifact_percentage(self):
        """ Sum the area of all artifacts. """
        sum = self.adf["area"].sum()
        total_area = self.get_full_tissue_area()
        percentage = (sum / total_area) * 100
        return percentage

    def view_artifacts(self):
        viewer = zs.pl.WSIViewer(self.wsi)
        viewer.add_image('thumbnail')
        viewer.add_polygons(self.ARTIFACT_KEY, color_by='class', alpha=0.5)
        viewer.add_contours(self.TISSUE_KEY)
        viewer.show()

class MultipleQC:
    def __init__(self, wsi_paths, output_dir, max_artifact_percentage=20.0):
        self.wsi_paths = wsi_paths
        self.output_dir = output_dir
        self.max_artifact_percentage = max_artifact_percentage
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "qc_summary.csv")
        self.qc_results = self.get_artifact_summary()
        self.save_summary()

    def get_artifact_summary(self):
        qc_results = []
        for path in tqdm(self.wsi_paths, desc="QC Progress"):
            slide_name = os.path.basename(path)
            print(f"Processing {slide_name}...")
            try:
                wsiqc = SingleQC(path)
                artifact_percentage = wsiqc.get_artifact_percentage()
                if artifact_percentage < self.max_artifact_percentage:
                    qc_results.append({
                        "slide": slide_name,
                        "wsi_path": path,
                        "qc_object": wsiqc,
                        "artifact_percentage": artifact_percentage,
                        "status": "accepted"
                    })
                else: 
                    print(f"Skipping {slide_name}: {artifact_percentage:.2f}% artifacts (>{self.max_artifact_percentage}%)")
                    qc_results.append({
                        "slide": slide_name,
                        "wsi_path": path,
                        "qc_object": wsiqc,
                        "artifact_percentage": artifact_percentage,
                        "status": "rejected"
                    })
                del wsiqc
                gc.collect()
            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                qc_results.append({
                    "slide": slide_name,
                    "wsi_path": path,
                    "qc_object": None,
                    "artifact_percentage": None,
                    "status": f"error: {str(e)}"
                })
                gc.collect()
        return qc_results

    def save_summary(self):
        try:
            df_qc = pd.DataFrame(self.qc_results)
            df_qc.to_csv(self.csv_path, index=False)
            print(f"QC summary saved to {self.csv_path}")
        except Exception as e:
            print(f"Error saving QC summary: {e}")

# Example usage
if __name__ == "__main__":
    # Single slide QC
    slide_paths = [
        "path/to/wsi",
        "path/to/another/wsi"
    ]
    wsiqc = SingleQC(slide_paths[0])
    wsiqc.view_artifacts()

    # Multiple slides QC
    output_dir = "path/to/output"
    multiqc = MultipleQC(slide_paths, output_dir)
    qc_results = multiqc.get_artifact_summary()
    print(qc_results)