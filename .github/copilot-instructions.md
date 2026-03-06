# 100k Codebase AI Agent Instructions

## Project Overview
This is a **development sandbox** for iterating on medical imaging ML pipelines for whole slide image (WSI) analysis. The goal is to develop and test feature extraction, tissue segmentation, and multiple instance learning (ABMIL) workflows that will be integrated into the production codebase offline.

**Key Directories:**
- `notebooks/` - Exploratory ML development: feature extraction workflows, ABMIL training, analysis scripts (primary AI development focus)
- `embedding/` - In-progress feature extraction and embedding generation code (some hardcoded paths; intended for offline integration)
- `scripts/` - Analysis utilities and data inspection tools

## Architecture & Data Flow

### WSI Processing Pipeline
1. **Raw Input**: `.mrxs` (Mirax) whole slide image files
2. **Zarr Caching**: Slides converted to Zarr format (`.zarr`) for efficient tiling and annotation storage
3. **Tissue Segmentation**: `SegmentTissue` class identifies tissue regions, creates tiles (224px default)
4. **Feature Extraction**: `ExtractFeatures` class uses deep learning models (h-optimus-0, UniMASS) to extract tile embeddings
5. **MIL Classification**: `ABMIL` (attention-based multiple instance learning) on tile embeddings for slide-level predictions

**Critical Pattern**: MRXS → Zarr conversion happens via `wsidata.open_wsi(slide_path, zarr_path)`. Zarr paths follow: `basename(mrxs).replace(".mrxs", ".zarr")`

### Core Libraries & APIs
- **wsidata** - Opens WSI files, manages Zarr caching; key: `open_wsi(slide_path, zarr_path)` returns WSI object with `.tables`, `.shapes`, `.write()`
- **lazyslide (zs)** - Preprocessing: `zs.pp.tile_tissues()`, `zs.pp.find_tissues()`, `zs.seg._tissue.tissue()`
- **PyTorch** - ABMIL model training with custom `ZarrSlideDataset` (batch_size=1 for variable tile counts)

## Key Patterns & Conventions

### Dataset & Feature Keys
- **Tissue keys**: `tissue_default`, `tissue_grandqc`, `tissue_threshold` (segmentation methods)
- **Tile keys**: `tiles_224` (224px tiles), `clean_tiles_224` (artifacts removed)
- **Feature keys**: `features_{model_name}` or `clean_features_{model_name}` (e.g., `features_h-optimus-0`)
- Example: `ExtractFeatures(wsi_path, zarr_dir, model="h-optimus-0", remove_artifacts=True)` → generates `clean_features_h-optimus-0`

### Feature Normalization
Per-slide normalization required in ABMIL training: `feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)`

### Caching & Lookups
- `tissue_artifact_segmentation.py` uses pickle-based caching with versioning: `cache[slide_name][category][version] = {"status": status}`
- `ExtractFeatures.is_empty_array()` checks for None, empty geometry, or zero arrays

### Error Handling
- Catch `RuntimeError` for missing tissue/tiles: "No tissue detected", "No tiles generated"
- DataFrame conversions: `lists2tuples()` for hashability, `tuples2lists()` to reverse
- Missing value handling: `strings2lists()` parses string representations of lists

## Development Workflows

### Running Feature Extraction
```python
from notebooks.feature_extraction import ExtractMany
extractor = ExtractMany(
    wsi_paths=list_of_paths,
    output_path="embeddings_dir",
    local_zarr_dir="zarr_cache_dir",
    model="h-optimus-0",
    remove_artifacts=False
)
# Auto-processes slides, caches Zarr files, writes embeddings
```

### Training ABMIL Model
```python
from notebooks.abmil_multiclass import ZarrSlideDataset, train_ABMIL, validate_ABMIL
dataset = ZarrSlideDataset(df, filename_col="path", label_col="diagnosis", 
                           feature_key="features_h-optimus-0", 
                           tile_key="tiles_224", zarr_dir="zarr_cache")
model = train_ABMIL(train_df, train_dataset, label_col="diagnosis", n_epochs=10)
labels, preds = validate_ABMIL(model, val_dataset)
```

### Environment Setup
- Windows paths hardcoded in `embedding/embedding.py` (D:\ drive) - update for cross-platform use
- OpenSlide library requires platform-specific binaries (handled via `os.add_dll_directory` on Windows)
- Offline mode for HuggingFace models: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` env vars set in extraction code

## Important Gotchas
1. **Batch Size = 1**: ABMIL uses batch_size=1 in DataLoader because slides have variable tile counts
2. **Device Handling**: Explicitly move features/labels to device in training loops; use `next(model.parameters()).device`
3. **Zarr Path Conversion**: Always replace `.mrxs` → `.zarr` consistently; mismatches break feature lookup
4. **Empty Slides**: Check `is_empty_array()` after tile generation; some slides may have no tissue
5. **DataFrame Index Reset**: `ZarrSlideDataset.__init__` calls `df.reset_index(drop=True)` for safe indexing

## File References for Common Tasks
- Tile generation & tissue segmentation: [tissue_artifact_segmentation.py](notebooks/tissue_artifact_segmentation.py)
- Feature extraction orchestration: [feature_extraction.py](notebooks/feature_extraction.py)
- ABMIL model & training: [abmil_multiclass.py](notebooks/abmil_multiclass.py)
- Helper utilities (data conversions): [helper_functions.py](notebooks/helper_functions.py)
- Large-scale embedding generation: [embedding/embedding.py](embedding/embedding.py)
