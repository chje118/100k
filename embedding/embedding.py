import sys
import logging
from datetime import datetime
sys.path.append(r'D:\TransPath-main')
OPENSLIDE_PATH = r'D:\openslide-bin-4.0.0.2-windows-x64\bin'
CACHE_FILE = r'D:\DATA\mrxs_cache.pkl'  # Changed from .parquet to .pkl

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # this is the unpatched timm, note that there is a patched timm for Ctranspath
from skimage.filters import threshold_otsu


import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

sys.path.append(r'D:\TransPath-main')

from ctran import ctranspath


def load_cache() -> pd.DataFrame:
    """
    Load the cache file if it exists, otherwise return an empty DataFrame.

    Returns:
        pd.DataFrame: Cached file information
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return pd.DataFrame(columns=["File Path", "Creation Date", "Modification Date", "MRXS Size", "Associated Data Size", "Last Checked"])



def setup_logging():
    """Set up logging to both file and console"""
    # Create the logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_filename = log_dir / f"{current_date}-EMBEDDING.log"
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def get_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file"""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def print_versions(logger):
    """Print version information using logger"""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PIL version: {Image.__version__}")
    logger.info(f"OpenSlide version: {openslide.__version__}")
    logger.info(f"Timm version: {timm.__version__}")
    
    # Print model file hashes
    model_files = [
        "D:/DATA/models/ctranspath.pth",
        "D:/DATA/models/uni_mass100k.bin",
        "D:/DATA/models/hoptimus0.bin"
    ]
    
    logger.info("\nModel File Hashes:")
    for model_path in model_files:
        try:
            file_hash = get_file_hash(model_path)
            logger.info(f"{Path(model_path).name}: {file_hash}")
        except Exception as e:
            logger.error(f"Error computing hash for {Path(model_path).name}: {str(e)}")


@dataclass
class ModelConfig:
    name: str
    path: str
    model_type: str  # 'ctranspath' or 'timm'
    output_dim: int
    model_arch: Optional[str] = None  # timm model architecture name
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

class ProcessingTracker:
    def __init__(self, tracker_path: str, output_dir: str = "D:/DATA/embeddings"):
        self.tracker_path = tracker_path
        self.output_dir = output_dir
        self.processed_files = self._load_tracker()
    
    def _scan_output_directory(self) -> Dict[str, Set]:
        """
        Scan the output directory to identify completed files based on existing embeddings.
        
        Returns:
            Dict containing sets of completed and failed files
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Scanning output directory: {self.output_dir}")
        
        completed = set()
        embedding_files = {}
        
        # Scan output directory for existing files
        for file in os.listdir(self.output_dir):
            if file.endswith('_embeddings.npy'):
                # Extract original MRXS filename from embedding filename
                # Format: slidename_modelname_embeddings.npy
                slide_name = file.split('_')[0]
                if slide_name not in embedding_files:
                    embedding_files[slide_name] = set()
                embedding_files[slide_name].add(file)
        
        # Check which slides have embeddings from all models
        expected_models = {'ctranspath', 'uni', 'hoptimus'}
        for slide_name, files in embedding_files.items():
            model_names = {f.split('_')[1] for f in files}
            if model_names >= expected_models:  # Using >= to handle case where there might be extra files
                # Reconstruct original MRXS path
                # Note: This assumes MRXS files are in the cache DataFrame
                # You might need to adjust this based on your file naming convention
                mrxs_path = None
                try:
                    cache_df = load_cache()
                    matching_files = cache_df[cache_df['File Path'].str.contains(slide_name, case=False)]
                    if not matching_files.empty:
                        mrxs_path = matching_files.iloc[0]['File Path']
                        completed.add(mrxs_path)
                        logger.info(f"Found completed slide: {mrxs_path}")
                except Exception as e:
                    logger.warning(f"Error matching slide {slide_name} to MRXS path: {str(e)}")
        
        logger.info(f"Found {len(completed)} completed files from output directory")
        
        return {
            'completed': completed,
            'failed': {},  # We can't determine failed files from output directory
            'in_progress': set()
        }
    
    def _load_tracker(self) -> Dict:
        """
        Load the tracker file if it exists, otherwise reconstruct from output directory.
        """
        logger = logging.getLogger(__name__)
        
        if os.path.exists(self.tracker_path):
            logger.info(f"Loading existing tracker from {self.tracker_path}")
            try:
                with open(self.tracker_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading tracker file: {str(e)}")
                logger.info("Falling back to output directory scan")
                return self._scan_output_directory()
        else:
            logger.info("No tracker file found, scanning output directory")
            return self._scan_output_directory()
    
    def save_tracker(self):
        """Save the current state to the tracker file"""
        with open(self.tracker_path, 'wb') as f:
            pickle.dump(self.processed_files, f)
    
    def mark_completed(self, filepath: str):
        """Mark a file as completed and save the tracker"""
        self.processed_files['completed'].add(filepath)
        if filepath in self.processed_files['in_progress']:
            self.processed_files['in_progress'].remove(filepath)
        self.save_tracker()
    
    def mark_failed(self, filepath: str, error: str):
        """Mark a file as failed with error message and save the tracker"""
        self.processed_files['failed'][filepath] = error
        if filepath in self.processed_files['in_progress']:
            self.processed_files['in_progress'].remove(filepath)
        self.save_tracker()
    
    def mark_in_progress(self, filepath: str):
        """Mark a file as in progress and save the tracker"""
        self.processed_files['in_progress'].add(filepath)
        self.save_tracker()
    
    def is_processed(self, filepath: str) -> bool:
        """Check if a file has been processed (either completed or failed)"""
        return (filepath in self.processed_files['completed'] or 
                filepath in self.processed_files['failed'])

class TileDataset(Dataset):
    def __init__(self, images: List[Image.Image], transform=None):
        self.images = images
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                              std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


class ModelManager:
    def __init__(self, model_configs: List[ModelConfig], device: str = 'cuda'):
        self.device = device
        self.models: Dict[str, tuple] = {}  # (model, transform)
        self._load_models(model_configs)
    
    def _load_models(self, configs: List[ModelConfig]):
        for config in configs:
            logger = logging.getLogger(__name__)
            logger.info(f"Loading {config.name}")

            # Create transform for this model
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std)
            ])
            
            # Load model based on type
            if config.model_type == 'ctranspath':
                model = ctranspath()
                model.head = torch.nn.Identity()
                state_dict = torch.load(config.path)
                model.load_state_dict(state_dict['model'], strict=True)
                
            elif config.model_type == 'timm':
                if not config.model_arch:
                    raise ValueError(f"Model architecture must be specified for timm models")
                
                if config.name == "uni":
                    # Special initialization for UNI model
                    model = timm.create_model(
                        config.model_arch,
                        img_size=224,
                        patch_size=16,
                        init_values=1e-5,
                        num_classes=0,
                        dynamic_img_size=True
                    )

                elif config.name == 'hoptimus':
                    model = timm.create_model(
                        config.model_arch,
                        img_size=224,
                        patch_size=14,  # Explicitly set patch size
                        num_classes=0,
                        global_pool='token',
                        pretrained=False
                    )

                else:
                    # Default initialization for other timm models
                    model = timm.create_model(
                        config.model_arch,
                        img_size=224,
                        num_classes=0,
                        pretrained=False
                    )
                
                state_dict = torch.load(config.path)
                model.load_state_dict(state_dict, strict=True)
            
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            model.eval()
            model.to(self.device)

            logger.info(f"Model {config.name} loaded to device: {self.device}")
            
            self.models[config.name] = (model, transform, config.output_dim)

    def get_embeddings(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        embeddings = {}
        logger = logging.getLogger(__name__)
        
        for model_name, (model, _, output_dim) in self.models.items():
            model_embeddings = np.zeros(
                (len(dataloader.dataset), output_dim), 
                dtype=np.float32
            )
            
            batch_idx = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'Embedding with {model_name}'):
                    batch = batch.to(self.device)
                    features = model(batch)
                    features = features.cpu().numpy()
                        
                    model_embeddings[batch_idx:batch_idx+len(features)] = features
                    batch_idx += len(features)
            
            embeddings[model_name] = model_embeddings
            logger.info(f"Completed embeddings for model {model_name}")
        
        return embeddings

def extract_tiles(tiles: DeepZoomGenerator,
                 level: int = -1,
                 limit: int = 0,
                 int_filter: int = 231,
                 tile_size: int = 224) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """
    Extract tiles from a whole slide image using DeepZoomGenerator.
    """
    logger = logging.getLogger(__name__)
    
    # Get dimensions for the specified level
    cols, rows = tiles.level_tiles[level]
    logger.info(f"Processing slide with dimensions: {cols}x{rows}")
    
    # Get thumbnail for tissue detection
    thumb = tiles._osr.get_thumbnail((cols, rows))
    temp_tile_RGB = thumb.convert('RGB')
    
    # Find tissue regions using intensity filtering
    #tile_array = np.array(temp_tile_RGB)

    thumb_gray = np.array(temp_tile_RGB.convert("L"))
    threshold_val = threshold_otsu(thumb_gray)

    print(f"t-val is {threshold_val}")

    r, c = np.where(thumb_gray < threshold_val)
    tile_coordinates = list(zip(r, c))
    
    # Shuffle and limit if specified
    if limit > 0:
        import random
        random.shuffle(tile_coordinates)
        tile_coordinates = tile_coordinates[:limit]
    
    extracted_tiles = []
    final_coordinates = []
    
    # Extract tiles
    for row, col in tqdm(tile_coordinates, desc='Extracting tiles'):
        try:
            # Get tile at the specified coordinates
            tile = tiles.get_tile(tiles.level_count + level, (col, row))
            temp_tile_RGB = tile.convert('RGB')
            
            # Only keep tiles of the correct size
            if temp_tile_RGB.size == (tile_size, tile_size):
                final_coordinates.append((row, col))
                extracted_tiles.append(temp_tile_RGB)
                
        except Exception as e:
            logger.error(f"Error extracting tile at ({row}, {col}): {str(e)}")
            continue
        
        # Check if we've reached the limit
        if limit > 0 and len(extracted_tiles) >= limit:
            break
    
    logger.info(f"Extracted {len(extracted_tiles)} tiles")
    return extracted_tiles, final_coordinates

def create_model_dataloaders(
    images: List[Image.Image],
    model_manager: ModelManager,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    """
    Create separate dataloaders for each model with its specific transform.
    """
    logger = logging.getLogger(__name__)
    dataloaders = {}
    
    for model_name, (_, transform, _) in model_manager.models.items():
        logger.info(f"Creating dataloader for model {model_name}")
        # Create dataset with model-specific transform
        dataset = TileDataset(
            images=images,
            transform=transform
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for coordinate matching
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # Process all tiles
            persistent_workers=True if num_workers > 0 else False
        )
        
        dataloaders[model_name] = dataloader
        logger.info(f"Created dataloader for {model_name} with {len(dataset)} samples")
        
    return dataloaders


def main():
    # Set up logging
    logger = setup_logging()
    
    # Print versions
    print_versions(logger)
    
    # Define model configurations for offline use
    MODEL_CONFIGS = [
        ModelConfig(
            name="ctranspath",
            path="D:/DATA/models/ctranspath.pth",
            model_type="ctranspath",
            output_dim=768,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ModelConfig(
            name="uni",
            path="D:/DATA/models/uni_mass100k.bin",
            model_type="timm",
            model_arch="vit_large_patch16_224",
            output_dim=1024,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ModelConfig(
            name="hoptimus",
            path="D:/DATA/models/hoptimus0.bin",
            model_type="timm",
            model_arch="vit_giant_patch14_reg4_dinov2",
            output_dim=1536,
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        )
    ]

    MODEL_CONFIGS = [MODEL_CONFIGS[0], MODEL_CONFIGS[1]]

    
    
    OUTPUT_DIR = "D:/DATA/embeddings"
    TRACKER_FILE = "D:/DATA/processing_tracker.pkl"
    BATCH_SIZE = 256

    mrxs_df = load_cache()
    logger.info(f"Loaded cache with {len(mrxs_df)} entries")
    
    all_mrxs = list(mrxs_df['File Path'].values)
    logger.info(f"Found {len(all_mrxs)} MRXS files to process")
    
    # Initialize components
    tracker = ProcessingTracker(TRACKER_FILE)
    logger.info("Initializing model manager...")
    model_manager = ModelManager(MODEL_CONFIGS)
    
    # Process slides
    # Replace the main processing loop with this corrected version:

    # In main():
    for filepath in tqdm(all_mrxs, desc='Processing slides'):
        if tracker.is_processed(filepath):
            logger.info(f"Skipping already processed file: {filepath}")
            continue
            
        try:
            tracker.mark_in_progress(filepath)
            logger.info(f"Processing slide: {filepath}")
            
            # Extract tiles
            slide = open_slide(filepath)
            tiles_generator = DeepZoomGenerator(slide, 224, overlap=0, limit_bounds=True)
            tiles_data, coordinates = extract_tiles(tiles_generator, level=-1, limit=0)
            logger.info(f"Extracted {len(tiles_data)} tiles from slide")
            
            # Create single dataloader for all models
            dataloader = DataLoader(
                TileDataset(tiles_data),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )
            
            # Get embeddings once
            logger.info("Processing embeddings for all models")
            all_embeddings = model_manager.get_embeddings(dataloader)
            
            # Save results
            slide_name = Path(filepath).stem
            coords_df = pd.DataFrame(coordinates, columns=['row', 'col'])
            coords_path = f"{OUTPUT_DIR}/{slide_name}_coordinates.pkl"
            coords_df.to_pickle(coords_path)
            logger.info(f"Saved coordinates to {coords_path}")
            
            for model_name, embedding in all_embeddings.items():
                output_path = f"{OUTPUT_DIR}/{slide_name}_{model_name}_embeddings.npy"
                np.save(output_path, embedding)
                logger.info(f"Saved embeddings for {model_name} to {output_path}")
            
            tracker.mark_completed(filepath)
            logger.info(f"Successfully processed slide: {filepath}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}", exc_info=True)
            tracker.mark_failed(filepath, str(e))
            
        finally:
            # Clean up slide object
            try:
                slide.close()
            except:
                pass

if __name__ == "__main__":
    try:
        logger = setup_logging()
        logger.info("Starting embedding process...")
        main()
        logger.info("Embedding process completed successfully")
    except Exception as e:
        logger.error("Fatal error in main process", exc_info=True)
        raise