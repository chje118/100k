import os
import shutil
import re
import pandas as pd

class FindWSIData:
    def __init__(self, base_dir, df, col):
        self.base_dir = base_dir
        self.df = df
        self.col = col
        self._index_folders()
        self.df_results = self.get_df_results()

    def _index_folders(self):
        folder_index = {}
        for root, dirs, files in os.walk(self.base_dir):
            for d in dirs:
                folder_index.setdefault(d, []).append(os.path.join(root, d))
        self.folder_index = folder_index

    def _get_filename(self, wsi_path):
        """ Extract the filename from a WSI path. """
        return os.path.basename(wsi_path)

    def _get_base_name(self, filename):
        """ Get base name (folder) without extension. """
        return os.path.splitext(filename)[0]

    def _find_matching_folders(self, wsi_path):
        """
        Find all folders anywhere under base_dir that match the base name of the WSI file,
        including numbered variants like 'slide1', 'slide1 (2)', etc.
        """
        filename = self._get_filename(wsi_path)
        base_name = self._get_base_name(filename)
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

if __name__ == "__main__":
    # Example usage
    df_wsi = pd.read_csv("path/to/wsi_stats.csv")
    
    missing_data = df_wsi[
        (df_wsi["file_size"].isna() | (df_wsi["file_size"] == 0)) | 
        (df_wsi["data_folder_size"].isna() | (df_wsi["data_folder_size"] == 0))
    ]
    
    missing_data = missing_data.reset_index()  # filename becomes a column

    base_directory = "//regsj/.intern/appl/Deep_Visual_Proteomics"
    finder = FindWSIData(base_directory, df = missing_data, col = 'filename')
    df_result = finder.df_results
    print("Before removing empty rows:", len(df_result))
    finder.remove_empty_rows()
    df_result = finder.df_results
    print("After removing empty rows:", len(df_result))
