import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Set
from collections import Counter, defaultdict
from datetime import datetime
import os
from tqdm import tqdm
import logging

def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_tracker(tracker_path: str) -> Dict:
    """Load the processing tracker file."""
    try:
        with open(tracker_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Tracker file not found at: {tracker_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading tracker: {str(e)}")
        return None

def get_file_stats(filepath: Path) -> Dict:
    """Get file statistics."""
    try:
        stats = filepath.stat()
        return {
            'size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_ctime),
            'modified': datetime.fromtimestamp(stats.st_mtime)
        }
    except Exception as e:
        logging.error(f"Error getting stats for {filepath}: {str(e)}")
        return None

def calculate_processing_times(embedding_dir: Path) -> Dict:
    """Calculate processing times based on sequential coordinate file creation timestamps."""
    # Get all coordinate files and their creation times
    coordinate_files = list(embedding_dir.glob('*_coordinates.pkl'))
    
    # Get creation times and sort them
    creation_times = []
    for file in coordinate_files:
        stats = get_file_stats(file)
        if stats:
            creation_times.append((file, stats['created']))
    
    # Sort by creation time
    creation_times.sort(key=lambda x: x[1])
    
    # Calculate time differences between consecutive files
    processing_times = []
    for i in range(1, len(creation_times)):
        time_diff = (creation_times[i][1] - creation_times[i-1][1]).total_seconds()
        if time_diff < 3600:  # Ignore gaps longer than 1 hour
            processing_times.append(time_diff)
    
    if not processing_times:
        return {
            'median_time': 0,
            'mean_time': 0,
            'min_time': 0,
            'max_time': 0,
            'processing_rate': 0,
            'sample_size': 0
        }
    
    median_time = np.median(processing_times)
    return {
        'median_time': median_time,
        'mean_time': np.mean(processing_times),
        'min_time': np.min(processing_times),
        'max_time': np.max(processing_times),
        'processing_rate': 3600 / median_time if median_time > 0 else 0,
        'sample_size': len(processing_times)
    }

def analyze_embedding_file(file_path: Path) -> Dict:
    """Analyze a single embedding file."""
    try:
        if file_path.suffix == '.npy':
            data = np.load(file_path)
            return {
                'datapoints': data.shape[0],
                'dimensions': data.shape[1],
                'file_size': file_path.stat().st_size,
                'memory_size': data.nbytes,
                'dtype': str(data.dtype)
            }
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return {
                'datapoints': len(data),
                'file_size': file_path.stat().st_size,
                'type': 'coordinates' if isinstance(data, pd.DataFrame) else 'other'
            }
    except Exception as e:
        logging.error(f"Error analyzing {file_path}: {str(e)}")
        return None

def analyze_embeddings(embedding_dir: Path) -> Dict:
    """Analyze all embedding files in the directory."""
    embedding_files = list(embedding_dir.glob('*_embeddings.npy'))
    coordinate_files = list(embedding_dir.glob('*_coordinates.pkl'))
    
    stats = defaultdict(dict)
    total_size = 0
    datapoints_per_slide = []
    
    # Analyze embedding files
    for file in tqdm(embedding_files, desc="Analyzing embedding files"):
        result = analyze_embedding_file(file)
        if result:
            model_name = file.stem.split('_')[-2]
            if model_name not in stats['model_stats']:
                stats['model_stats'][model_name] = {
                    'total_datapoints': 0,
                    'total_size': 0,
                    'file_count': 0
                }
            
            stats['model_stats'][model_name]['total_datapoints'] += result['datapoints']
            stats['model_stats'][model_name]['total_size'] += result['file_size']
            stats['model_stats'][model_name]['file_count'] += 1
            total_size += result['file_size']
            datapoints_per_slide.append(result['datapoints'])
    
    # Analyze coordinate files
    coord_sizes = []
    for file in coordinate_files:
        result = analyze_embedding_file(file)
        if result:
            coord_sizes.append(result['file_size'])
            total_size += result['file_size']
    
    # Calculate aggregate statistics
    stats['embedding_stats'] = {
        'total_slides_processed': len(coordinate_files),
        'total_size': total_size,
        'average_datapoints_per_slide': np.mean(datapoints_per_slide) if datapoints_per_slide else 0,
        'median_datapoints_per_slide': np.median(datapoints_per_slide) if datapoints_per_slide else 0,
        'min_datapoints_per_slide': np.min(datapoints_per_slide) if datapoints_per_slide else 0,
        'max_datapoints_per_slide': np.max(datapoints_per_slide) if datapoints_per_slide else 0,
        'average_coordinates_file_size': np.mean(coord_sizes) if coord_sizes else 0
    }
    
    # Add processing time calculations
    stats['time_stats'] = calculate_processing_times(embedding_dir)
    
    return dict(stats)

def analyze_tracker(tracker_data: Dict, embedding_dir: Path) -> Dict:
    """Analyze tracker data and embeddings."""
    if not tracker_data:
        return {}
    
    completed: Set = tracker_data.get('completed', set())
    failed: Dict = tracker_data.get('failed', {})
    in_progress: Set = tracker_data.get('in_progress', set())
    
    stats = {
        'tracker_stats': {
            'total_files': len(completed) + len(failed) + len(in_progress),
            'completed_count': len(completed),
            'failed_count': len(failed),
            'in_progress_count': len(in_progress),
            'success_rate': len(completed) / (len(completed) + len(failed)) * 100 if (len(completed) + len(failed)) > 0 else 0,
            'error_types': dict(Counter(failed.values())),
            'failed_files': {str(Path(filepath).name): error for filepath, error in failed.items()}
        }
    }
    
    embedding_stats = analyze_embeddings(embedding_dir)
    stats.update(embedding_stats)
    
    return stats

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def print_analysis(stats: Dict):
    """Print the analysis results."""
    print("\n=== Comprehensive Embedding Analysis ===")
    
    # Tracker Statistics
    print("\n=== Processing Status ===")
    t_stats = stats['tracker_stats']
    print(f"Total Files: {t_stats['total_files']}")
    print(f"Completed: {t_stats['completed_count']} ({t_stats['success_rate']:.1f}%)")
    print(f"Failed: {t_stats['failed_count']}")
    print(f"In Progress: {t_stats['in_progress_count']}")
    
    # Processing Time Statistics
    print("\n=== Processing Time Statistics ===")
    t_stats = stats['time_stats']
    print(f"Sample Size: {t_stats['sample_size']} time differences")
    print(f"Median Processing Time per Slide: {format_time(t_stats['median_time'])}")
    print(f"Mean Processing Time per Slide: {format_time(t_stats['mean_time'])}")
    print(f"Fastest Processing Time: {format_time(t_stats['min_time'])}")
    print(f"Slowest Processing Time: {format_time(t_stats['max_time'])}")
    print(f"Processing Rate: {t_stats['processing_rate']:.1f} slides/hour")
    
    # Embedding Statistics
    print("\n=== Embedding Statistics ===")
    e_stats = stats['embedding_stats']
    print(f"Total Slides Processed: {e_stats['total_slides_processed']}")
    print(f"Total Data Size: {format_bytes(e_stats['total_size'])}")
    print(f"\nTiles per Slide:")
    print(f"  Average: {e_stats['average_datapoints_per_slide']:.0f}")
    print(f"  Median:  {e_stats['median_datapoints_per_slide']:.0f}")
    print(f"  Min:     {e_stats['min_datapoints_per_slide']:.0f}")
    print(f"  Max:     {e_stats['max_datapoints_per_slide']:.0f}")
    
    # Model Statistics
    print("\n=== Model Statistics ===")
    for model, m_stats in stats['model_stats'].items():
        print(f"\n{model.upper()}:")
        print(f"  Total Datapoints: {m_stats['total_datapoints']:,}")
        print(f"  Total Size: {format_bytes(m_stats['total_size'])}")
        print(f"  Average Size per File: {format_bytes(m_stats['total_size']/m_stats['file_count'])}")
    
    # Error Summary
    if stats['tracker_stats']['failed_count'] > 0:
        print("\n=== Error Summary ===")
        for error, count in stats['tracker_stats']['error_types'].items():
            print(f"\n{count} files failed with error:")
            print(f"{error}")

def main():
    logger = setup_logging()
    
    # Configuration
    TRACKER_FILE = Path("D:/DATA/processing_tracker.pkl")
    EMBEDDING_DIR = Path("D:/DATA/embeddings")
    
    # Load and analyze data
    logger.info("Loading tracker data...")
    tracker_data = load_tracker(TRACKER_FILE)
    
    if tracker_data:
        logger.info("Analyzing data...")
        stats = analyze_tracker(tracker_data, EMBEDDING_DIR)
        
        logger.info("Generating report...")
        print_analysis(stats)

if __name__ == "__main__":
    main()