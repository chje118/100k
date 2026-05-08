import os
import pandas as pd
from wsidata import open_wsi
import tqdm

class WSIMetadata:
    def __init__(self, wsi_paths, cache_path=None, force_refresh=False):
        self.wsi_paths = wsi_paths
        self.cache_path = cache_path
        self.force_refresh = force_refresh
        self.metadata = self._extract_all()

    def _extract_metadata(self, wsi_path):
        metadata = []
        filename = os.path.basename(wsi_path)
        print(f"Extracting metadata from {filename}...")
        try:
            wsi = open_wsi(wsi_path)
            meta = wsi.properties
            shape = getattr(meta, "shape", None)
            n_level = getattr(meta, "n_level", None)
            level_shape = getattr(meta, "level_shape", None)
            level_downsample = getattr(meta, "level_downsample", None)
            mpp = getattr(meta, "mpp", None)
            magnification = getattr(meta, "magnification", None)
            bounds = getattr(meta, "bounds", None)
            vendor = wsi.raw_properties.get("openslide.vendor")
            comment = wsi.raw_properties.get("openslide.comment")
            wsi.close()

            metadata.append({
                'filename': filename,
                'wsi_path': wsi_path,
                'shape': shape,
                'n_level': n_level,
                'level_shape': level_shape,
                'level_downsample': level_downsample,
                'mpp': mpp,
                'magnification': magnification,
                'bounds': bounds,
                'vendor': vendor,
                'comment': comment,
            })
        except Exception as e:
            print(f"Error reading {wsi_path}: {e}")
            metadata.append({
                'filename': filename,
                'wsi_path': wsi_path,
                'error': str(e)
            })
        return pd.DataFrame(metadata)

    def _extract_all(self):
        # Load cache if available and not forcing refresh
        if self.cache_path and os.path.exists(self.cache_path) and not self.force_refresh:
            print(f"Loading metadata from cache: {self.cache_path}")
            cached = pd.read_csv(self.cache_path)
            processed_paths = set(cached['wsi_path'])
        else:
            cached = pd.DataFrame()
            processed_paths = set()

        all_metadata = [cached] if not cached.empty else []
        for wsi_path in tqdm.tqdm(self.wsi_paths, desc="Extracting WSI metadata"):
            if wsi_path in processed_paths:
                continue  # Skip already processed
            df = self._extract_metadata(wsi_path)
            all_metadata.append(df)
            # Save to cache after each file
            if self.cache_path:
                pd.concat(all_metadata, ignore_index=True).to_csv(self.cache_path, index=False)
        return pd.concat(all_metadata, ignore_index=True)

if __name__ == "__main__":
    wsi_paths = [
        "C:\\Users\\chris\\OneDrive\\Dokumenter\\SDU\\Master's Thesis Project\\Sample MRXS\\CMU-1.mrxs",
        "C:\\Users\\chris\\OneDrive\\Dokumenter\\SDU\\Master's Thesis Project\\Sample MRXS\\Mirax2.2-3.mrxs"
    ]
    cache_file = "wsi_metadata_cache.csv"
    wsi_meta = WSIMetadata(wsi_paths, cache_path=cache_file)
    df_meta = wsi_meta.metadata
    print(df_meta)
    print(df_meta.describe())

