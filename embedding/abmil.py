import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def calculate_baseline_metrics(labels: pd.Series) -> Tuple[float, float]:
    """
    Calculate baseline accuracy and precision based on majority class prediction
    """
    # Count class distribution
    class_counts = labels.value_counts()
    total_samples = len(labels)
    
    # Majority class prediction
    majority_class = class_counts.index[0]
    majority_count = class_counts[0]
    
    # Calculate baseline metrics
    baseline_accuracy = majority_count / total_samples
    baseline_precision = majority_count / total_samples  # For binary classification
    
    return baseline_accuracy, baseline_precision

def collate_batch(batch):
    """
    Custom collate function to handle variable-sized embeddings
    Args:
        batch: List of tuples (embeddings, label)
    Returns:
        embeddings_batch: Padded embeddings tensor
        labels_batch: Labels tensor
    """
    # Sort batch by sequence length in descending order
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get sequence lengths and max length
    lengths = [x[0].shape[0] for x in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    hidden_dim = batch[0][0].shape[1]
    
    # Create padded batch tensor
    embeddings_batch = torch.zeros(batch_size, max_len, hidden_dim)
    labels_batch = torch.zeros(batch_size, 1)
    
    # Fill the tensors
    for i, (embeddings, label) in enumerate(batch):
        seq_len = embeddings.shape[0]
        embeddings_batch[i, :seq_len] = embeddings
        labels_batch[i] = label
    
    return embeddings_batch, labels_batch

def load_cache(CACHE_FILE) -> pd.DataFrame:
    """Load the cache file if it exists, otherwise return an empty DataFrame."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return pd.DataFrame(columns=[
        "File Path", "Creation Date", "Modification Date", 
        "MRXS Size", "Associated Data Size", "Last Checked"
    ])

def load_excel_data_2012(file_path: str) -> pd.DataFrame:
    """Load and combine data from 2012 Excel file."""
    print(f"Loading {file_path}")
    excel_file = pd.ExcelFile(file_path)
    dfs = [excel_file.parse(sheet, skiprows=4) for sheet in excel_file.sheet_names]
    combined_df = pd.concat(dfs)
    combined_df['modtdato'] = pd.to_datetime(combined_df['modtdato'], format='%Y-%m-%d')
    return combined_df.reset_index(drop=True)

def load_excel_data_2013(file_path: str) -> pd.DataFrame:
    """Load and combine data from 2013 Excel file."""
    print(f"Loading {file_path}")
    excel_file = pd.ExcelFile(file_path)
    dfs = excel_file.parse('datafile')
    dfs['modtdato'] = pd.to_datetime(dfs['modtdato'], format='%Y-%m-%d')
    return dfs.reset_index(drop=True)

def load_and_combine_data(file_paths: Dict[str, str]) -> pd.DataFrame:
    """Load and combine data from multiple Excel files."""
    print('Loading individual files')
    all_df = []
    load_fctn = {
        '2013': load_excel_data_2013,
        '2012': load_excel_data_2012
    }
    
    for year, file_path in file_paths.items():
        print(f'\n=== Processing {year} data ===')
        combined_df = load_fctn[year](file_path)
        all_df.append(combined_df)
        print(f"Loaded {len(combined_df)} records from {year}")
    
    print('\nCombining all data...')
    final_df = pd.concat(all_df, ignore_index=True)
    print(f"Total records after combination: {len(final_df)}")
    return final_df

def match_and_process_data(combined_df: pd.DataFrame, cache_file: str) -> pd.DataFrame:
    """Match and process the combined data with MRXS files."""
    # Load cache and create file dictionary
    mrxs_df = load_cache(cache_file)
    all_mrxs = mrxs_df['File Path'].tolist()
    file_dict = create_file_dict(all_mrxs)
    
    # Match records
    result_df = match_records(combined_df, file_dict)
    
    # Print statistics
    matched_count = result_df['match'].notna().sum()
    print(f"\nMatching Statistics:")
    print(f"Matched {matched_count} out of {len(result_df)} records")
    print(f"Used {len(file_dict)} out of {len(all_mrxs)} files")
    pct = matched_count/len(result_df)*100
    print(f"Matching rate: {pct:.2f}%")
    
    return result_df


def create_file_dict(mrxs_paths: list) -> dict:
    """Create dictionary for file path lookups."""
    return {Path(filepath).stem[:8]: filepath for filepath in mrxs_paths}

def match_records(df: pd.DataFrame, file_dict: dict) -> pd.DataFrame:
    """Match records with corresponding files."""
    df = df.copy()
    df['rekvnr_short'] = df['rekvnr'].astype(str).str[:8]
    df['match'] = df['rekvnr_short'].map(file_dict)
    return df

class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(AttentionMIL, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, num_instances, input_dim)
        
        # Calculate attention weights
        attn_weights = self.attention(x)  # (batch_size, num_instances, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention to get bag representation
        weighted_x = torch.sum(x * attn_weights, dim=1)  # (batch_size, input_dim)
        
        # Get predictions
        pred = self.classifier(weighted_x)  # (batch_size, 1)
        
        return pred, attn_weights

class SlideDataset(Dataset):
    def __init__(self, 
                 embeddings_dict: Dict[str, np.ndarray],
                 labels: pd.Series,
                 label_encoder: Optional[LabelEncoder] = None):
        """
        Initialize dataset with automatic label type handling
        
        Args:
            embeddings_dict: Dictionary of slide embeddings
            labels: Series of labels (can be string, int, or float)
            label_encoder: Optional pre-fitted LabelEncoder for string labels
        """
        self.embeddings = embeddings_dict
        self.slide_ids = list(embeddings_dict.keys())
        
        # Handle label encoding based on dtype
        label_values = [labels[k] for k in self.slide_ids]
        if isinstance(label_values[0], str):
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                numeric_labels = self.label_encoder.fit_transform(label_values)
            else:
                self.label_encoder = label_encoder
                numeric_labels = self.label_encoder.transform(label_values)
        else:
            self.label_encoder = None
            numeric_labels = label_values
            
        self.labels = {k: float(v) for k, v in zip(self.slide_ids, numeric_labels)}
        
    def __len__(self) -> int:
        return len(self.slide_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slide_id = self.slide_ids[idx]
        embeddings = torch.FloatTensor(self.embeddings[slide_id])
        label = torch.FloatTensor([self.labels[slide_id]])
        return embeddings, label

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to both file and console"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = log_dir / f"{current_date}-ABMIL_training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_embeddings(embeddings_dir: str, 
                   model_name: str,
                   matched_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Load embeddings for matched slides"""
    embeddings_dict = {}
    
    for _, row in matched_df.iterrows():
        if pd.isna(row['match']):
            continue
            
        slide_id = Path(row['match']).stem
        embedding_path = os.path.join(
            embeddings_dir, 
            f"{slide_id}_{model_name}_embeddings.npy"
        )
        
        if os.path.exists(embedding_path):
            embeddings_dict[slide_id] = np.load(embedding_path)
    
    return embeddings_dict

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> float:
    model.train()
    epoch_loss = 0
    
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions, _ = model(embeddings)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            predictions, _ = model(embeddings)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    pred_classes = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_classes)
    precision = precision_score(all_labels, pred_classes)
    recall = recall_score(all_labels, pred_classes)
    f1 = f1_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, precision, recall, f1, auc, avg_loss

def print_dataset_statistics(matched_df: pd.DataFrame, 
                           target_column: str, 
                           embeddings_dict: Dict[str, np.ndarray],
                           logger: logging.Logger):
    """Print detailed statistics about the dataset"""
    # Overall dataset statistics
    total_samples = len(matched_df)
    matched_samples = matched_df['match'].notna().sum()
    
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples in dataset: {total_samples}")
    logger.info(f"Samples with matched slides: {matched_samples}")
    logger.info(f"Matching rate: {(matched_samples/total_samples)*100:.2f}%")
    
    # Target variable statistics
    matched_data = matched_df[matched_df['match'].notna()]
    class_counts = matched_data[target_column].value_counts()
    class_proportions = class_counts / len(matched_data) * 100
    
    logger.info(f"\nClass Distribution for {target_column}:")
    for class_label, count in class_counts.items():
        logger.info(f"Class {class_label}: {count} samples ({class_proportions[class_label]:.2f}%)")
    
    # Embedding statistics
    total_embeddings = len(embeddings_dict)
    embedding_dim = next(iter(embeddings_dict.values())).shape[1]
    avg_instances = np.mean([emb.shape[0] for emb in embeddings_dict.values()])
    min_instances = min([emb.shape[0] for emb in embeddings_dict.values()])
    max_instances = max([emb.shape[0] for emb in embeddings_dict.values()])
    
    logger.info("\nEmbedding Statistics:")
    logger.info(f"Total slides with embeddings: {total_embeddings}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Average instances per slide: {avg_instances:.2f}")
    logger.info(f"Min instances per slide: {min_instances}")
    logger.info(f"Max instances per slide: {max_instances}")

def plot_class_distribution(matched_df: pd.DataFrame, 
                          target_column: str,
                          output_dir: str):
    """Plot class distribution and save to file"""
    plt.figure(figsize=(10, 6))
    
    # Only use matched samples
    matched_data = matched_df[matched_df['match'].notna()]
    class_counts = matched_data[target_column].value_counts()
    total_samples = len(matched_data)
    
    # Create bar plot
    ax = class_counts.plot(kind='bar')
    plt.title(f'Class Distribution for {target_column} (Matched Samples Only)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage labels on top of each bar
    for i, count in enumerate(class_counts):
        percentage = count / total_samples * 100
        ax.text(i, count, f'{percentage:.1f}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{target_column}_class_distribution.png')
    plt.savefig(plot_path)
    plt.close()

def plot_training_curves(train_losses: List[float], 
                        val_metrics: List[List[float]], 
                        val_losses: List[float],
                        fold: int,
                        target_column: str,
                        output_dir: str,
                        baseline_acc: float,
                        baseline_prec: float):
    """Plot training curves for each fold"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation metrics with baselines
    metrics = list(zip(*val_metrics))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    for metric, name in zip(metrics, metric_names):
        ax2.plot(metric, label=name)
    
    # Add baseline lines
    ax2.axhline(y=baseline_acc, color='r', linestyle='--', 
                label=f'Baseline Acc: {baseline_acc:.3f}')
    ax2.axhline(y=baseline_prec, color='g', linestyle='--', 
                label=f'Baseline Prec: {baseline_prec:.3f}')
    
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{target_column}_fold{fold}_training_curves.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def train_model(embeddings_dir: str,
                matched_df: pd.DataFrame,
                target_column: str,
                model_name: str = "ctranspath",
                n_splits: int = 5,
                epochs: int = 10,  # Changed to 10 epochs
                batch_size: int = 32,
                learning_rate: float = 0.001,
                hidden_dim: int = 256,
                output_dir: str = "D:/DATA/abmil_results",
                device: str = "cuda"):
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting ABMIL training for target: {target_column}")
    
    # Load embeddings
    embeddings_dict = load_embeddings(embeddings_dir, model_name, matched_df)
    logger.info(f"Loaded embeddings for {len(embeddings_dict)} slides")
    
    # Print dataset statistics
    print_dataset_statistics(matched_df, target_column, embeddings_dict, logger)
    
    # Plot class distribution
    plot_class_distribution(matched_df, target_column, output_dir)
    
    # Prepare labels
    labels_dict = {
        Path(row['match']).stem: row[target_column] 
        for _, row in matched_df.iterrows() 
        if pd.notna(row['match'])
    }
    
    # Keep only slides with both embeddings and labels
    common_slides = set(embeddings_dict.keys()) & set(labels_dict.keys())
    embeddings_dict = {k: embeddings_dict[k] for k in common_slides}
    labels_dict = {k: labels_dict[k] for k in common_slides}
    
    # Convert to arrays for stratification
    slide_ids = np.array(list(common_slides))
    labels = pd.Series(labels_dict)
    
    # Calculate baseline metrics
    matched_labels = matched_df[matched_df['match'].notna()][target_column]
    baseline_acc, baseline_prec = calculate_baseline_metrics(matched_labels)
    logger.info(f"\nBaseline Metrics:")
    logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")
    logger.info(f"Baseline Precision: {baseline_prec:.4f}")
    
    # Create label encoder if needed
    if isinstance(labels.iloc[0], str):
        label_encoder = LabelEncoder()
        stratification_labels = label_encoder.fit_transform(labels)
    else:
        label_encoder = None
        stratification_labels = labels.values
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = []
    
    # Initialize results dictionary
    results = {
        'fold_metrics': [],
        'training_curves': [],
        'model_paths': [],
        'predictions': [],
        'label_encoder': label_encoder  # Save for later use
    }
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(slide_ids, stratification_labels)):
        logger.info(f"\nStarting fold {fold + 1}/{n_splits}")
        
        # Prepare datasets
        train_slides = slide_ids[train_idx]
        val_slides = slide_ids[val_idx]
        
        train_embeddings = {k: embeddings_dict[k] for k in train_slides}
        val_embeddings = {k: embeddings_dict[k] for k in val_slides}
        
        train_labels = pd.Series({k: labels_dict[k] for k in train_slides})
        val_labels = pd.Series({k: labels_dict[k] for k in val_slides})
        
        # Print fold-specific class distribution
        logger.info("\nFold class distribution:")
        logger.info("Training set:")
        for label, count in train_labels.value_counts().items():
            logger.info(f"Class {label}: {count} samples")
        logger.info("\nValidation set:")
        for label, count in val_labels.value_counts().items():
            logger.info(f"Class {label}: {count} samples")
        
        # Create datasets with shared label encoder
        train_dataset = SlideDataset(train_embeddings, train_labels, label_encoder)
        val_dataset = SlideDataset(val_embeddings, val_labels, label_encoder)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_batch
        )
        
        # Initialize model
        input_dim = next(iter(embeddings_dict.values())).shape[1]
        model = AttentionMIL(input_dim, hidden_dim).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        best_auc = 0
        best_metrics = None
        train_losses = []
        val_losses = []
        val_metrics_history = []
        
        for epoch in range(epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            train_losses.append(train_loss)
            
            # Evaluation
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_metrics_history.append(val_metrics[:-1])  # Exclude loss from metrics history
            val_losses.append(val_metrics[-1])  # Add validation loss
            current_auc = val_metrics[4]
            
            if current_auc > best_auc:
                best_auc = current_auc
                best_metrics = val_metrics[:-1]  # Exclude loss from best metrics
                
                model_path = os.path.join(
                    output_dir, 
                    f"{target_column}_fold{fold}_best_model.pth"
                )
                torch.save(model.state_dict(), model_path)
                results['model_paths'].append(model_path)
            
            if (epoch + 1) % 2 == 0:  # Log every 2 epochs
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_metrics[-1]:.4f} - "
                    f"Val AUC: {current_auc:.4f}"
                )
        
        # Plot training curves with validation loss and baselines
        plot_training_curves(
            train_losses, 
            val_metrics_history, 
            val_losses,
            fold, 
            target_column, 
            output_dir,
            baseline_acc,
            baseline_prec
        )
        
        cv_metrics.append(best_metrics)
        logger.info(
            f"Fold {fold + 1} best metrics - "
            f"Accuracy: {best_metrics[0]:.4f} - "
            f"Precision: {best_metrics[1]:.4f} - "
            f"Recall: {best_metrics[2]:.4f} - "
            f"F1: {best_metrics[3]:.4f} - "
            f"AUC: {best_metrics[4]:.4f}"
        )
        
        results['fold_metrics'].append(best_metrics)
        results['training_curves'].append((train_losses, val_metrics_history))
    
    # Calculate and log average metrics
    avg_metrics = np.mean(cv_metrics, axis=0)
    std_metrics = np.std(cv_metrics, axis=0)
    
    logger.info("\nFinal cross-validation metrics:")
    logger.info(f"Accuracy: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
    logger.info(f"Precision: {avg_metrics[1]:.4f} (±{std_metrics[1]:.4f})")
    logger.info(f"Recall: {avg_metrics[2]:.4f} (±{std_metrics[2]:.4f})")
    logger.info(f"F1 Score: {avg_metrics[3]:.4f} (±{std_metrics[3]:.4f})")
    logger.info(f"AUC-ROC: {avg_metrics[4]:.4f} (±{std_metrics[4]:.4f})")
    
    # Save results
    results_path = os.path.join(output_dir, f"{target_column}_training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    return avg_metrics, std_metrics, results

if __name__ == "__main__":
    # Constants
    FILE_PATHS = {
        '2012': "D:/DATA/glasdata 2012.xlsx",
        '2013': "D:/DATA/glasdata 2013.xlsx"
    }
    CACHE_FILE = "D:/DATA/mrxs_cache.pkl"
    EMBEDDINGS_DIR = "D:/DATA/embeddings"
    OUTPUT_DIR = "D:/DATA/abmil_results"
    MODEL_NAME = "hoptimus"
    TARGET_COLUMN = "sex"  # Replace with actual column name
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and combine data from multiple files
    combined_df = load_and_combine_data(FILE_PATHS)
    
    # Match and process the combined data
    result_df = match_and_process_data(combined_df, CACHE_FILE)
    
    
    # Train model
    avg_metrics, std_metrics, results = train_model(
        embeddings_dir=EMBEDDINGS_DIR,
        matched_df=result_df,
        target_column=TARGET_COLUMN,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )