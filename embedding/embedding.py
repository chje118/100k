import sys
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
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm #this is the ctranspath timm...


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
    def __init__(self, tracker_path: str):
        self.tracker_path = tracker_path
        self.processed_files = self._load_tracker()
    
    def _load_tracker(self) -> Dict:
        if os.path.exists(self.tracker_path):
            with open(self.tracker_path, 'rb') as f:
                return pickle.load(f)
        return {
            'completed': set(),
            'failed': {},
            'in_progress': set()
        }
    
    def save_tracker(self):
        with open(self.tracker_path, 'wb') as f:
            pickle.dump(self.processed_files, f)
    
    def mark_completed(self, filepath: str):
        self.processed_files['completed'].add(filepath)
        if filepath in self.processed_files['in_progress']:
            self.processed_files['in_progress'].remove(filepath)
        self.save_tracker()
    
    def mark_failed(self, filepath: str, error: str):
        self.processed_files['failed'][filepath] = error
        if filepath in self.processed_files['in_progress']:
            self.processed_files['in_progress'].remove(filepath)
        self.save_tracker()
    
    def mark_in_progress(self, filepath: str):
        self.processed_files['in_progress'].add(filepath)
        self.save_tracker()
    
    def is_processed(self, filepath: str) -> bool:
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

            print(f"Model at device: {self.device}")
            
            self.models[config.name] = (model, transform, config.output_dim)
    def get_embeddings(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        embeddings = {}
        
        for model_name, (model, _, output_dim) in self.models.items():
            model_embeddings = np.zeros(
                (len(dataloader.dataset), output_dim), 
                dtype=np.float32
            )
            
            batch_idx = 0
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'Embedding with {model_name}'):
                    batch = batch.to(self.device)
                    features = model(batch)
                    features = features.cpu().numpy()
                        
                    model_embeddings[batch_idx:batch_idx+len(features)] = features
                    batch_idx += len(features)
            
            embeddings[model_name] = model_embeddings
        
        return embeddings

def extract_tiles(tiles: DeepZoomGenerator,
                 level: int = -1,
                 limit: int = 0,
                 int_filter: int = 250,
                 tile_size: int = 224) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """
    Extract tiles from a whole slide image using DeepZoomGenerator.
    
    Args:
        tiles: DeepZoomGenerator instance
        level: Level at which to extract tiles (default: -1)
        limit: Maximum number of tiles to extract (0 for no limit)
        int_filter: Intensity filter threshold for tissue detection
        tile_size: Size of tiles to extract
    
    Returns:
        Tuple containing:
        - List of extracted tile images
        - List of (row, col) coordinates for each tile
    """
    # Get dimensions for the specified level
    cols, rows = tiles.level_tiles[level]
    
    # Get thumbnail for tissue detection
    thumb = tiles._osr.get_thumbnail((cols, rows))
    temp_tile_RGB = thumb.convert('RGB')
    
    # Find tissue regions using intensity filtering
    tile_array = np.array(temp_tile_RGB)
    r, c = np.where(tile_array.mean(axis=2) < int_filter)
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
            print(f"Error extracting tile at ({row}, {col}): {str(e)}")
            continue
        
        # Check if we've reached the limit
        if limit > 0 and len(extracted_tiles) >= limit:
            break
    
    print(f"Extracted {len(extracted_tiles)} tiles")
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
    
    Args:
        images: List of PIL Images to process
        model_manager: ModelManager instance containing models and their transforms
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory in dataloader (recommended for GPU)
    
    Returns:
        Dictionary mapping model names to their respective DataLoaders
    """
    dataloaders = {}
    
    for model_name, (_, transform, _) in model_manager.models.items():
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
        
    return dataloaders


def main():
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
            name="hoptimus",
            path="D:/DATA/models/hoptimus0.bin",
            model_type="timm",
            model_arch="vit_base_patch16_224",  # or whatever the correct architecture is
            output_dim=1536,
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
        ModelConfig(
            name="uni",
            path="D:/DATA/models/uni_mass100k.bin",
            model_type="timm",
            model_arch="vit_large_patch16_224",
            output_dim=1024,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ]


    MODEL_CONFIGS  = [MODEL_CONFIGS[0]]
    
    OUTPUT_DIR = "D:/DATA/embeddings"
    TRACKER_FILE = "D:/DATA/processing_tracker.pkl"
    BATCH_SIZE = 256  # Reduced batch size to handle larger models

    mrxs_df = load_cache()
    
    all_mrxs = list(mrxs_df['File Path'].values)
    
    
    # Initialize components
    tracker = ProcessingTracker(TRACKER_FILE)
    model_manager = ModelManager(MODEL_CONFIGS)
    
    # Process slides
    for filepath in tqdm(all_mrxs, desc='Processing slides'):
        if tracker.is_processed(filepath):
            continue
            
        try:
            tracker.mark_in_progress(filepath)
            
            # Extract tiles
            slide = open_slide(filepath)
            tiles_generator = DeepZoomGenerator(slide, 224, overlap=0, limit_bounds=True)
            tiles_data, coordinates = extract_tiles(tiles_generator, level=-1, limit=0)
            
            # Create dataloaders for each model
            dataloaders = create_model_dataloaders(tiles_data, model_manager, BATCH_SIZE)
            
            # Process with each model
            all_embeddings = {}
            for model_name, dataloader in dataloaders.items():
                embeddings = model_manager.get_embeddings(dataloader)
                all_embeddings[model_name] = embeddings[model_name]
            
            # Save results
            slide_name = Path(filepath).stem
            coords_df = pd.DataFrame(coordinates, columns=['row', 'col'])
            coords_df.to_pickle(f"{OUTPUT_DIR}/{slide_name}_coordinates.pkl")
            
            for model_name, embedding in all_embeddings.items():
                np.save(
                    f"{OUTPUT_DIR}/{slide_name}_{model_name}_embeddings.npy",
                    embedding
                )
            
            tracker.mark_completed(filepath)
            
        except Exception as e:
            raise
            print(f"Error processing {filepath}: {str(e)}")
            tracker.mark_failed(filepath, str(e))

if __name__ == "__main__":
    main()