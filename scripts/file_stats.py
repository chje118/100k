import os
import datetime
import statistics
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import time

FOLDER_PATH = r'\\regsj.intern\appl\Deep_Visual_Proteomics'
CACHE_FILE = 'mrxs_cache.parquet'

def get_file_info(file_path: str) -> Dict:
    """
    Get file information including creation time, modification time, and size.

    Args:
        file_path (str): Path to the file.

    Returns:
        Dict: Dictionary containing file information.
    """
    creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
    modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    size = os.path.getsize(file_path)
    return {
        "File Path": file_path,
        "Creation Date": creation_time,
        "Modification Date": modification_time,
        "MRXS Size": size,
    }

def get_associated_data_size(folder_path: str) -> int:
    """
    Calculate the total size of files in a given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Total size of files in bytes.
    """
    total_size = 0
    if os.path.exists(folder_path):
        for dirpath, _, filenames in os.walk(folder_path):
            total_size += sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
    return total_size

def load_cache() -> pd.DataFrame:
    """
    Load the cache file if it exists, otherwise return an empty DataFrame.

    Returns:
        pd.DataFrame: Cached file information
    """
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)
    return pd.DataFrame(columns=["File Path", "Creation Date", "Modification Date", "MRXS Size", "Associated Data Size", "Last Checked"])

def collect_mrxs_info(folder_path: str, log_print) -> pd.DataFrame:
    """
    Collect information about .mrxs files and their associated data folders, using cache when possible.
    Updates cache every 1000 files to prevent data loss in case of crashes.

    Args:
        folder_path (str): Path to the root folder to search for .mrxs files.
        log_print (callable): Function to handle logging and printing.

    Returns:
        pd.DataFrame: DataFrame containing information about each .mrxs file.
    """
    # Load cached data
    cache_df = load_cache()
    current_time = datetime.datetime.now()
    new_records = []
    updated_count = 0
    
    log_print(f"Found {len(cache_df)} cached records")
    
    # Collect all .mrxs files
    log_print("Searching for mrxs files.")
    start_time = time.time() 
    all_mrxs_files = []
    for root, _, files in tqdm(os.walk(folder_path)):
        all_mrxs_files.extend(
            os.path.join(root, f) for f in files if f.endswith(".mrxs")
        )
    end_time = time.time()

    delta_min = (end_time-start_time)/60 

    log_print(f"Found {len(all_mrxs_files)} mrxs files. Took {delta_min:.2f} minutes.")
    
    # Process files with progress bar
    for mrxs_path in tqdm(all_mrxs_files):
        # Check if file exists in cache
        cached_info = cache_df[cache_df["File Path"] == mrxs_path]
        
        if len(cached_info) > 0:
            # Check if file has been modified since last cache
            modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(mrxs_path))
            if modification_time <= cached_info["Modification Date"].iloc[0]:
                continue
        
        # Get new file information
        info = get_file_info(mrxs_path)
        data_folder = os.path.splitext(os.path.basename(mrxs_path))[0]
        data_folder_path = os.path.join(os.path.dirname(mrxs_path), data_folder)
        info["Associated Data Size"] = get_associated_data_size(data_folder_path)
        info["Last Checked"] = current_time
        
        new_records.append(info)
        updated_count += 1
        
        # Update cache every 1000 files
        if len(new_records) >= 1000:
            new_df = pd.DataFrame(new_records)
            # Remove old entries for updated files
            cache_df = cache_df[~cache_df["File Path"].isin(new_df["File Path"])]
            # Combine old and new data
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            # Save updated cache
            cache_df.to_parquet(CACHE_FILE)
            log_print(f"Updated {updated_count} files in cache")
            # Clear new records after saving
            new_records = []
    
    # Save any remaining new records
    if new_records:
        new_df = pd.DataFrame(new_records)
        # Remove old entries for updated files
        cache_df = cache_df[~cache_df["File Path"].isin(new_df["File Path"])]
        # Combine old and new data
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        # Save updated cache
        cache_df.to_parquet(CACHE_FILE)
        log_print(f"Updated final {len(new_records)} files in cache")
    
    return cache_df

def calculate_statistics(mrxs_df: pd.DataFrame) -> Dict:
    """
    Calculate various statistics from the collected .mrxs file information.

    Args:
        mrxs_df (pd.DataFrame): DataFrame containing .mrxs file information.

    Returns:
        Dict: Dictionary containing calculated statistics.
    """
    total_size = mrxs_df["Associated Data Size"].sum() / (1024**3)
    count = len(mrxs_df)
    avg_file_size = total_size / count if count > 0 else 0
    
    dates = sorted(mrxs_df["Modification Date"])
    deltas = [(dates[i+1] - dates[i]).seconds for i in range(len(dates)-1)]
    
    return {
        "count": count,
        "total_size_gb": total_size,
        "avg_file_size_gb": avg_file_size,
        "median_delta": statistics.median(deltas) if deltas else 0,
        "mean_delta": statistics.mean(deltas) if deltas else 0
    }

def print_monthly_statistics(mrxs_df: pd.DataFrame, log_print):
    """
    Print monthly statistics for .mrxs file creation.

    Args:
        mrxs_df (pd.DataFrame): DataFrame containing .mrxs file information.
        log_print (callable): Function to handle logging and printing.
    """
    # Convert to datetime if not already
    mrxs_df["Modification Date"] = pd.to_datetime(mrxs_df["Modification Date"])
    
    # Group by month and calculate statistics
    monthly_stats = mrxs_df.groupby(mrxs_df["Modification Date"].dt.strftime("%b")).agg({
        "File Path": "count",
        "Modification Date": lambda x: x.sort_values().diff().dt.total_seconds()
    }).rename(columns={"File Path": "count"})
    
    log_print('Month - Number of Images created : Average time to acquire one image')
    
    for month, row in monthly_stats.iterrows():
        if len(row["Modification Date"]) > 2:
            deltas = row["Modification Date"].dropna()
            log_print(f"{month} - {row['count']:<10,} : Median {deltas.median():.0f} s, Mean {deltas.mean():.2f} s")

def main():
    """
    Main function to run the MRXS analysis script.
    """
    start_time = time.time()
    current_date = datetime.datetime.now()
    log_filename = f"mrxs_analysis_{current_date.strftime('%Y-%m-%d_%H-%M-%S')}.log"

    with open(log_filename, 'w') as log_file:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log_file)
            log_file.flush() 

        log_print(f"Script started at: {current_date}")
        log_print(f"Analyzing folder: {FOLDER_PATH}")

        mrxs_df = collect_mrxs_info(FOLDER_PATH, log_print)
        stats = calculate_statistics(mrxs_df)
        
        log_print(f'Number of files {stats["count"]:,}, total size: {stats["total_size_gb"]:.2f} GB, average file size: {stats["avg_file_size_gb"]:.2f} GB')
        log_print(f"Average time to acquire one image: Median {stats['median_delta']} s, Mean {stats['mean_delta']:.2f} s")
        
        print_monthly_statistics(mrxs_df, log_print)
        
        log_print(f'Number of files {stats["count"]:,}, total size: {stats["total_size_gb"]/1024:.2f} TB, average file size: {stats["avg_file_size_gb"]:.2f} GB')
        
        if stats["mean_delta"] > 0:
            estimate_completion_dates(stats["count"], stats["mean_delta"], log_print)

        end_time = time.time()
        execution_time = (end_time - start_time)/60
        log_print(f"\nScript execution time: {execution_time:.2f} minutes")
        log_print(f"Script completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()