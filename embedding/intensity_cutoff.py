import sys
import logging
from datetime import datetime
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm
import seaborn as sns
from PIL import Image
import pandas as pd

OPENSLIDE_PATH = r'D:\openslide-bin-4.0.0.2-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide.deepzoom import DeepZoomGenerator

# Constants
CACHE_FILE = r'D:\DATA\mrxs_cache.pkl'
SAMPLE_SIZE = 1000  # Number of random slides to analyze
PIXELS_PER_SLIDE = 100000  # Number of random pixels to sample per slide
OUTPUT_DIR = Path(r"D:\DATA\intensity_analysis")

def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_filename = log_dir / f"{current_date}-INTENSITY_ANALYSIS.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_cache():
    """Load the slide cache file"""
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)

def analyze_slide_intensity(slide_path):
    """Analyze intensity distribution of a single slide"""
    try:
        # Open slide and create thumbnail
        slide = openslide.open_slide(slide_path)
        tiles = DeepZoomGenerator(slide, 224, overlap=0, limit_bounds=True)
        
        # Get dimensions for the lowest level
        cols, rows = tiles.level_tiles[-1]
        thumb = slide.get_thumbnail((cols, rows))
        thumb_rgb = thumb.convert('RGB')
        
        # Convert to numpy array and calculate mean intensity
        tile_array = np.array(thumb_rgb)
        mean_intensity = tile_array.mean(axis=2)
        
        # Randomly sample pixels
        flat_intensity = mean_intensity.flatten()
        sampled_intensities = random.sample(list(flat_intensity), 
                                          min(PIXELS_PER_SLIDE, len(flat_intensity)))
        
        slide.close()
        return sampled_intensities
        
    except Exception as e:
        logging.error(f"Error processing {slide_path}: {str(e)}")
        return None

def main():
    logger = setup_logging()
    logger.info("Starting intensity analysis")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load and sample slides
    df = load_cache()
    all_slides = list(df['File Path'].values)
    sampled_slides = random.sample(all_slides, min(SAMPLE_SIZE, len(all_slides)))
    
    # Collect intensity data
    all_intensities = []
    for slide_path in tqdm(sampled_slides, desc="Analyzing slides"):
        intensities = analyze_slide_intensity(slide_path)
        if intensities:
            all_intensities.extend(intensities)
    
    # Convert to numpy array
    intensity_array = np.array(all_intensities)
    
    # Calculate statistics
    mean_intensity = np.mean(intensity_array)
    std_intensity = np.std(intensity_array)
    
    # Use Otsu's method to find optimal threshold
    hist, bins = np.histogram(intensity_array, bins=256)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Weight arrays
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    
    # Mean arrays
    mean1 = np.cumsum(hist * bin_centers) / w1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]
    
    # Calculate variance
    variance = w1[:-1] * w2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    optimal_threshold = bin_centers[np.argmax(variance)]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with log scale
    sns.histplot(data=intensity_array, bins=100, color='blue', alpha=0.6)
    plt.yscale('log')  # Set y-axis to log scale
    
    # Add vertical line for optimal threshold
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.1f}')
    
    # Add statistical annotations
    plt.text(0.02, 0.98, 
            f'Mean: {mean_intensity:.1f}\nStd: {std_intensity:.1f}\n'
            f'Suggested Cutoff: {optimal_threshold:.1f}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.title('Slide Intensity Distribution Analysis (Log Scale)')
    plt.xlabel('Mean Pixel Intensity')
    plt.ylabel('Count (Log Scale)')
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = OUTPUT_DIR / f'intensity_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Save numerical results
    results = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'optimal_threshold': optimal_threshold,
        'analyzed_slides': len(sampled_slides),
        'total_pixels': len(intensity_array),
        'timestamp': timestamp
    }
    
    results_path = OUTPUT_DIR / f'intensity_analysis_{timestamp}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    logger.info(f"Suggested intensity cutoff: {optimal_threshold:.1f}")

if __name__ == "__main__":
    main()