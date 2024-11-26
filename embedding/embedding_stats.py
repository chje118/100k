import pickle
from pathlib import Path
from typing import Dict, Set
from collections import Counter
from pprint import pprint

def load_tracker(tracker_path: str) -> Dict:
    """Load the processing tracker file."""
    try:
        with open(tracker_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Tracker file not found at: {tracker_path}")
        return None
    except Exception as e:
        print(f"Error loading tracker: {str(e)}")
        return None

def analyze_tracker(tracker_data: Dict) -> Dict:
    """Analyze the tracker data and return statistics."""
    if not tracker_data:
        return {}
    
    completed: Set = tracker_data.get('completed', set())
    failed: Dict = tracker_data.get('failed', {})
    in_progress: Set = tracker_data.get('in_progress', set())
    
    # Group failed files by error message
    error_types = Counter(failed.values())
    
    stats = {
        'total_files': len(completed) + len(failed) + len(in_progress),
        'completed_count': len(completed),
        'failed_count': len(failed),
        'in_progress_count': len(in_progress),
        'success_rate': len(completed) / (len(completed) + len(failed)) * 100 if (len(completed) + len(failed)) > 0 else 0,
        'error_types': dict(error_types),
        'failed_files': {
            str(Path(filepath).name): error 
            for filepath, error in failed.items()
        }
    }
    
    return stats

def print_tracker_analysis(stats: Dict):
    """Print the analysis results in a formatted way."""
    print("\n=== Processing Tracker Analysis ===")
    print(f"\nOverall Statistics:")
    print(f"Total Files Processed: {stats['total_files']}")
    print(f"Successfully Completed: {stats['completed_count']}")
    print(f"Failed: {stats['failed_count']}")
    print(f"In Progress: {stats['in_progress_count']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    
    print("\nError Type Distribution:")
    for error, count in stats['error_types'].items():
        print(f"- {error}: {count} occurrences")
    
    print("\nFailed Files and Their Errors:")
    print("-" * 50)
    for filename, error in stats['failed_files'].items():
        print(f"\nFile: {filename}")
        print(f"Error: {error}")

def main():
    TRACKER_FILE = "D:/DATA/processing_tracker.pkl"  # Update this path as needed
    
    # Load tracker data
    tracker_data = load_tracker(TRACKER_FILE)
    if not tracker_data:
        return
    
    # Analyze tracker data
    stats = analyze_tracker(tracker_data)
    
    # Print analysis
    print_tracker_analysis(stats)

if __name__ == "__main__":
    main()