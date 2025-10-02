import os
import shutil
from pathlib import Path
import re
import pandas as pd

class FindWSIFolder:
    def __init__(self, base_dir, df, col):
        self.base_dir = base_dir
        self.df = df
        self.col = col
        self.df_results = self.get_df_results()

    def _get_filename(self, wsi_path):
        """ Extract the filename from a WSI path. """
        return Path(wsi_path).name # e.g., slide1.mrxs or slide1 (2).mrxs

    def _get_base_name(self, filename):
        """ Get base name (folder) without extension. """
        return Path(filename).stem  # e.g., slide1 or slide1 (2)

    def _find_matching_folders(self, wsi_path):
        """ Find all folders matching the base name of the WSI file, including (2), (3), etc."""
        filename = self._get_filename(wsi_path)
        base_name = self._get_base_name(filename)
        search_base = re.sub(r" \(\d+\)$", "", base_name) # remove (number) suffix
        pattern = re.compile(rf"^{re.escape(search_base)}( \(\d+\))?$") # match folders with base name or base name + (number)
        return [folder for folder in self.base_dir.glob("**/*") if folder.is_dir() and pattern.match(folder.name)]

    def _has_files(self, folder):
        """Return True if the folder contains any files."""
        return any(f.is_file() for f in folder.rglob('*'))

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
        self.df['matching_folders'] = [self._find_folders(wsi_path) for wsi_path in self.df[self.col]]
        return self.df
    
    def remove_empty_rows(self):
        """Remove rows from the DataFrame where no matching folders with files were found."""
        self.df_results = self.df_results[self.df_results['matching_folders'].notna() & self.df_results['matching_folders'].map(len) > 0]

if __name__ == "__main__":
    # Example usage
    df_wsi = pd.read_csv("path/to/wsi_stats.csv")
    
    missing_data = df_wsi[
        (df_wsi["file_size"].isna() | (df_wsi["file_size"] == 0)) | 
        (df_wsi["data_folder_size"].isna() | (df_wsi["data_folder_size"] == 0))
    ]
    
    missing_data = missing_data.reset_index()  # filename becomes a column

    base_directory = Path(r"//regsj/.intern/appl/Deep_Visual_Proteomics")
    finder = FindWSIFolder(base_directory, df = missing_data, col = 'filename')
    df_result = finder.df_results
    print("Before removing empty rows:", len(df_result))
    finder.remove_empty_rows()
    df_result = finder.df_results
    print("After removing empty rows:", len(df_result))
