import pandas as pd
import os
import pickle
import re
from tqdm import tqdm 

class WSIStats:
    def __init__(self, filename):
        self.filename = filename
        self.rekvnr = os.path.basename(filename)[:8]
        self.file_size = os.path.getsize(filename)
        self.data_folder = os.path.splitext(filename)[0]
        self.data_folder_size = self.get_folder_size(self.data_folder) if os.path.isdir(self.data_folder) else 0

    @staticmethod
    def get_folder_size(folder):
        total = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total

    def to_dict(self):
        return {
            "filename": self.filename,
            "rekvnr": self.rekvnr,
            "file_size": self.file_size,
            "data_folder_size": self.data_folder_size,
        }

class WSIStatsCache:
    def __init__(self, root_dir, cache_file):
        self.root_dir = root_dir
        self.cache_file = cache_file
        self.stats = []

    def scan_files(self):
        cached_files = set(stat.filename for stat in self.stats)
        new_count = 0
        pbar = tqdm(desc="Scanning files")
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for fname in filenames:
                pbar.update(1)
                if fname.lower().endswith('.mrxs'):
                    fpath = os.path.join(dirpath, fname)
                    if fpath not in cached_files:
                        self.stats.append(WSIStats(fpath))
                        self.save_cache()
                        new_count += 1
                        pbar.set_description(f"Found {new_count} new files")
        pbar.close()
        print(f"Scan complete. Found {new_count} new files.")

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.stats, f)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.stats = pickle.load(f)
                print(f"Loaded {len(self.stats)} previously processed entries.")
            return True
        return False

    def get_stats_dicts(self):
        return [stat.to_dict() for stat in self.stats]
    
    def main(self, reload=True):
        self.load_cache()
        if reload:
            self.scan_files()
            self.save_cache()
        stats_dicts = self.get_stats_dicts()
        # Create DataFrame with filenames as index
        df = pd.DataFrame(stats_dicts)
        df.set_index("filename", inplace=True)
        return df

class FindWSIData:
    def __init__(self, base_dir, df, col):
        self.base_dir = base_dir
        self.df = df
        self.col = col
        self._index_folders()
        self.df_results = self.get_df_results()

    def _index_folders(self):
        folder_index = {}
        for root, dirs, files in tqdm(os.walk(self.base_dir), desc = "Indexing folders..."):
            for d in dirs:
                folder_index.setdefault(d, []).append(os.path.join(root, d))
        self.folder_index = folder_index

    def _get_filename(self, wsi_path):
        """ Extract the filename from a WSI path. """
        return os.path.basename(wsi_path)

    def _get_name_wo_extension(self, filename):
        """ Get folder name without extension. """
        return os.path.splitext(filename)[0]

    def _find_matching_folders(self, wsi_path):
        """
        Find all folders anywhere under base_dir that match the base name of the WSI file,
        including numbered variants like 'slide1', 'slide1 (2)', etc.
        """
        filename = self._get_filename(wsi_path)
        base_name = self._get_name_wo_extension(filename)
        search_base = re.sub(r" \(\d+\)$", "", base_name)
        pattern = re.compile(rf"^{re.escape(search_base)}( \(\d+\))?$")
        matches = []
        for folder_name, paths in self.folder_index.items():
            if pattern.match(folder_name):
                matches.extend(paths)
        return matches

    def _has_files(self, folder):
        """Return True if the folder contains any files."""
        try:
            return any(os.path.isfile(os.path.join(folder, f)) for f in os.listdir(folder))
        except Exception:
            return False

    def _filter_empty_folders(self, folders):
        """Return only folders that contain files."""
        return [f for f in folders if self._has_files(f)]

    def _find_folders(self, wsi_path):
        matching_folders = self._find_matching_folders(wsi_path)
        if not matching_folders:
            print(f"No matching folder found for {wsi_path}.")
            return None
        else:
            print(f"Number of matching folders found for {wsi_path}: {len(matching_folders)}")
            matching_folders = self._filter_empty_folders(matching_folders)
            if not matching_folders:
                print(f"No matching folder with files found for {wsi_path}.")
                return None
            else:
                print(f"Number of matching folders with files found for {wsi_path}: {len(matching_folders)}")
            return matching_folders

    def get_df_results(self):
        self.df['matching_folders'] = self.df[self.col].apply(self._find_folders)
        return self.df

    def remove_empty_rows(self):
        """Remove rows from the DataFrame where no matching folders with files were found."""
        self.df_results = self.df_results[self.df_results['matching_folders'].apply(lambda x: x is not None and len(x) > 0)]

    def delete_files_with_empty_size(self):
        """ Delete files from disk where file_size is NaN or 0, and actual file size is 0. """
        for idx, row in self.df.iterrows():
            file_path = row[self.col]
            file_size = row.get("file_size", None)
            if pd.isna(file_size) or file_size == 0:
                if os.path.isfile(file_path):
                    try:
                        actual_size = os.path.getsize(file_path)
                        if actual_size == 0:
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")
                            self.find_other_mrxs_files(file_path)
                        else:
                            print(f"Skipped (not empty): {file_path} (actual size: {actual_size})")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                    
    def find_other_mrxs_files(self, file_path):
        """Search for .mrxs files with the same base name in the same directory as file_path."""
        dir_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pattern = re.compile(rf"^{re.escape(base_name)}(.*)?\.mrxs$", re.IGNORECASE)
        found_files = []
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if pattern.match(f):
                    found_files.append(os.path.join(dir_path, f))
        print(f"Found {len(found_files)} .mrxs files with base name {base_name} in {dir_path}")
        print(found_files)
        return found_files

# Example usage
if __name__ == "__main__":
    # STEP (1) Identify all WSIs in a root directory 
    
    ROOT_DIR = "path/to/root/dir"
    CACHE_FILE = "path/to/cache/file.pkl"

    wsi_cache = WSIStatsCache(ROOT_DIR, CACHE_FILE)
    df_wsi = wsi_cache.main(reload=True)
    
    # STEP (2)
    # Identify WSIs with missing data 
    missing_data = df_wsi[
        (df_wsi["file_size"].isna() | (df_wsi["file_size"] == 0)) | 
        (df_wsi["data_folder_size"].isna() | (df_wsi["data_folder_size"] == 0))
    ]
    
    missing_data = missing_data.reset_index()  # filename becomes a column

    finder = FindWSIData(ROOT_DIR, df = missing_data, col = 'filename')
    df_result = finder.df_results
    print("Before removing empty rows:", len(df_result))
    
    finder.remove_empty_rows()
    df_result = finder.df_results
    print("After removing empty rows:", len(df_result))