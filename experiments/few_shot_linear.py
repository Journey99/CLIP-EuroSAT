import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import clip
from tqdm import tqdm
from torch.utils.data import Subset

from utils import set_seed, get_dataloaders, compute_metrics, log_metrics, setup_logger
from utils.visualization import plot_confusion_matrix, plot_few_shot_curve
from models.clip_wrapper import CLIPWithLinearProbe

def few_shot_linear_experiment(shots_per_class=5):
    """
    Few-shot classification (Linear Probe)
    """
    logger = setup_logger(f'few_shot_linear_{shots_per_class}shot')
    logger.info(f"Starting Few-shot Linear Probe ({shots_per_class} shots)")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    
    # 데이터 로드
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir='data/eurosat',
        clip_preprocess=preprocess,
        batch_size=32
    )
    
    # Few-shot subset 생성
    logger.info(f"Creating {shots_per_class}-shot subset...")
    from utils.dataset import EuroSATDataset
    
    full_dataset = train_loader.dataset.dataset  # unwrap from Subset
    few_shot_indices = full_dataset.get_few_shot_subset(shots_per_class)
    few_shot_dataset = Subset(full_dataset, few_shot_indices)
    
    few_shot_loader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    logger.info(f"Few-shot dataset: {len(few_shot_dataset)} samples")
    
    # 모델 생성 (backbone 고정)
    model = CLIPWithLinearProbe(clip_model, len(class_names), freeze_backbone=True)
    model = model.to(device)
    
    # Optimizer (classifier만 학습)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    logger.info("Training linear probe...")
    num_epochs = 100
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in few_shot_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            val_acc = evaluate(model, val_loader, device)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'results/checkpoints/linear_{shots_per_class}shot_best.pth')
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(f'results/checkpoints/linear_{shots_per_class}shot_best.pth'))
    
    all_preds, all_labels = [], []
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    log_metrics(metrics, f'results/tables/linear_{shots_per_class}shot_metrics.json', logger)
    
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=f'results/figures/linear_{shots_per_class}shot_confusion.png'
    )
    
    logger.info("Few-shot linear experiment completed!")
    
    return metrics

def evaluate(model, dataloader, device):
    """Validation accuracy 계산"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

if __name__ == '__main__':
    # 여러 shot 설정 실험
    shots_list = [1, 5, 10, 20]
    results = {}
    
    for shots in shots_list:
        metrics = few_shot_linear_experiment(shots)
        results[f'{shots}-shot'] = metrics['accuracy']
    
    # Few-shot curve 플롯
    plot_few_shot_curve(
        shots_list,
        [results[f'{s}-shot']*100 for s in shots_list],
        save_path='results/figures/few_shot_curve.png'
    )