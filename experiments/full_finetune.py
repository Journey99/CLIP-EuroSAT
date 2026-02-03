import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import clip
from tqdm import tqdm

from utils import set_seed, get_dataloaders, compute_metrics, log_metrics, setup_logger
from utils.visualization import plot_confusion_matrix

def full_finetune_experiment():
    """
    Full fine-tuning (전체 모델 학습)
    """
    logger = setup_logger('full_finetune')
    logger.info("Starting Full Fine-tuning")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    
    # Classifier 추가
    feature_dim = clip_model.visual.output_dim
    num_classes = 10
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 데이터 로드
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir='data/eurosat',
        clip_preprocess=preprocess,
        batch_size=64
    )
    
    # Optimizer (전체 모델 학습)
    optimizer = optim.AdamW(
        list(clip_model.visual.parameters()) + list(classifier.parameters()),
        lr=1e-5,  # 낮은 learning rate
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # 학습
    logger.info("Training full model...")
    num_epochs = 30
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        clip_model.train()
        classifier.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = classifier(image_features.float())
            
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss/len(train_loader)})
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_full(clip_model, classifier, val_loader, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'clip_visual': clip_model.visual.state_dict(),
                'classifier': classifier.state_dict()
            }, 'results/checkpoints/full_finetune_best.pth')
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load('results/checkpoints/full_finetune_best.pth')
    clip_model.visual.load_state_dict(checkpoint['clip_visual'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    all_preds, all_labels = [], []
    clip_model.eval()
    classifier.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = classifier(image_features.float())
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    log_metrics(metrics, 'results/tables/full_finetune_metrics.json', logger)
    
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path='results/figures/full_finetune_confusion.png'
    )
    
    logger.info("Full fine-tuning experiment completed!")
    
    return metrics

def evaluate_full(clip_model, classifier, dataloader, device):
    """Validation accuracy"""
    clip_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = classifier(image_features.float())
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

if __name__ == '__main__':
    metrics = full_finetune_experiment()