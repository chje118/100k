import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                 labels: pd.Series):
        self.embeddings = embeddings_dict
        self.labels = labels
        self.slide_ids = list(embeddings_dict.keys())
        
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
            device: str) -> Tuple[float, float, float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            predictions, _ = model(embeddings)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    pred_classes = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_classes)
    precision = precision_score(all_labels, pred_classes)
    recall = recall_score(all_labels, pred_classes)
    f1 = f1_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, auc

def train_model(embeddings_dir: str,
                matched_df: pd.DataFrame,
                target_column: str,
                model_name: str = "ctranspath",
                n_splits: int = 5,
                epochs: int = 50,
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
    labels = np.array([labels_dict[k] for k in slide_ids])
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = []
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(slide_ids, labels)):
        logger.info(f"\nStarting fold {fold + 1}/{n_splits}")
        
        # Prepare datasets
        train_slides = slide_ids[train_idx]
        val_slides = slide_ids[val_idx]
        
        train_embeddings = {k: embeddings_dict[k] for k in train_slides}
        val_embeddings = {k: embeddings_dict[k] for k in val_slides}
        
        train_labels = pd.Series({k: labels_dict[k] for k in train_slides})
        val_labels = pd.Series({k: labels_dict[k] for k in val_slides})
        
        train_dataset = SlideDataset(train_embeddings, train_labels)
        val_dataset = SlideDataset(val_embeddings, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Initialize model
        input_dim = next(iter(embeddings_dict.values())).shape[1]
        model = AttentionMIL(input_dim, hidden_dim).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        best_auc = 0
        best_metrics = None
        
        for epoch in range(epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Evaluation
            val_metrics = evaluate(model, val_loader, device)
            current_auc = val_metrics[4]  # AUC is the last metric
            
            if current_auc > best_auc:
                best_auc = current_auc
                best_metrics = val_metrics
                
                # Save best model
                model_path = os.path.join(
                    output_dir, 
                    f"{target_column}_fold{fold}_best_model.pth"
                )
                torch.save(model.state_dict(), model_path)
            
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {train_loss:.4f} - "
                    f"Val AUC: {current_auc:.4f}"
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
    
    # Calculate and log average metrics
    avg_metrics = np.mean(cv_metrics, axis=0)
    std_metrics = np.std(cv_metrics, axis=0)
    
    logger.info("\nFinal cross-validation metrics:")
    logger.info(f"Accuracy: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
    logger.info(f"Precision: {avg_metrics[1]:.4f} (±{std_metrics[1]:.4f})")
    logger.info(f"Recall: {avg_metrics[2]:.4f} (±{std_metrics[2]:.4f})")
    logger.info(f"F1 Score: {avg_metrics[3]:.4f} (±{std_metrics[3]:.4f})")
    logger.info(f"AUC-ROC: {avg_metrics[4]:.4f} (±{std_metrics[4]:.4f})")
    
    return avg_metrics, std_metrics

if __name__ == "__main__":
    # Load matched data
    matched_df = main()  # From your provided code
    
    # Set up parameters
    EMBEDDINGS_DIR = "D:/DATA/embeddings"
    OUTPUT_DIR = "D:/DATA/abmil_results"
    MODEL_NAME = "ctranspath"  # or "uni" or "hoptimus"
    TARGET_COLUMN = "target_column_name"  # Replace with actual column name
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train model
    avg_metrics, std_metrics = train_model(
        embeddings_dir=EMBEDDINGS_DIR,
        matched_df=matched_df,
        target_column=TARGET_COLUMN,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )