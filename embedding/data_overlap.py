import os
import pickle
from pathlib import Path
import pandas as pd

# Constants
FILE_PATH_2013 = "D:/DATA/glasdata 2013.xlsx"
FILE_PATH_2012 = "D:/DATA/glasdata 2012.xlsx"
CACHE_FILE = "D:/DATA/mrxs_cache.pkl"

def load_cache() -> pd.DataFrame:
    """Load the cache file if it exists, otherwise return an empty DataFrame."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return pd.DataFrame(columns=[
        "File Path", "Creation Date", "Modification Date", 
        "MRXS Size", "Associated Data Size", "Last Checked"
    ])

def load_excel_data_2012(file_path: str) -> pd.DataFrame:
    """Load and combine data from all Excel sheets."""
    print(f"Loading {file_path}")
    excel_file = pd.ExcelFile(file_path)
    dfs = [excel_file.parse(sheet, skiprows=4) for sheet in excel_file.sheet_names]
    combined_df = pd.concat(dfs)
    combined_df['modtdato'] = pd.to_datetime(combined_df['modtdato'], format='%Y-%m-%d')
    return combined_df.reset_index(drop=True)

def load_excel_data_2013(file_path: str) -> pd.DataFrame:
    """Load and combine data from all Excel sheets."""
    print(f"Loading {file_path}")
    excel_file = pd.ExcelFile(file_path)
    dfs = excel_file.parse('datafile')
    dfs['modtdato'] = pd.to_datetime(dfs['modtdato'], format='%Y-%m-%d')
    return dfs.reset_index(drop=True)

def create_file_dict(mrxs_paths: list) -> dict:
    """Create dictionary for file path lookups."""
    return {Path(filepath).stem[:8]: filepath for filepath in mrxs_paths}

def match_records(df: pd.DataFrame, file_dict: dict) -> pd.DataFrame:
    """Match records with corresponding files."""
    df = df.copy()
    df['rekvnr_short'] = df['rekvnr'].astype(str).str[:8]
    df['match'] = df['rekvnr_short'].map(file_dict)
    return df

def main():
    # Load data

    load_fctn = {FILE_PATH_2013:load_excel_data_2013, FILE_PATH_2012:load_excel_data_2012}

    print('Loading individual files')

    all_df = []

    for FILE_PATH in [FILE_PATH_2012, FILE_PATH_2013]:
        print('\n===')
        combined_df = load_fctn[FILE_PATH](FILE_PATH)

        all_df.append(combined_df)
        mrxs_df = load_cache()
        all_mrxs = mrxs_df['File Path'].tolist()
        
        # Process and match records
        file_dict = create_file_dict(all_mrxs)
        result_df = match_records(combined_df, file_dict)
        
        # Print statistics
        matched_count = result_df['match'].notna().sum()
        print(f"Matched {matched_count} out of {len(result_df)} records")
        print(f"Used {len(file_dict)} out of {len(all_mrxs)} files")
        pct = matched_count/len(result_df)*100
        print(f"{pct:.2f} %")

    print('Combined matches')
    combined_df = pd.concat(all_df)
    mrxs_df = load_cache()
    all_mrxs = mrxs_df['File Path'].tolist()
    
    # Process and match records
    file_dict = create_file_dict(all_mrxs)
    result_df = match_records(combined_df, file_dict)
    
    # Print statistics
    matched_count = result_df['match'].notna().sum()
    print(f"Matched {matched_count} out of {len(result_df)} records")
    print(f"Used {len(file_dict)} out of {len(all_mrxs)} files")
    pct = matched_count/len(result_df)*100
    print(f"{pct:.2f} %")

    return result_df

if __name__ == "__main__":

    result_df = main()