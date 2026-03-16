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
    
    # 폴더 생성
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    clip_model.float()

    # ✅ Visual encoder만 가져오기 (더 안정적)
    visual_encoder = clip_model.visual
    
    # Classifier 추가
    feature_dim = visual_encoder.output_dim
    num_classes = 10
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 데이터 로드
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir='data/eurosat/',
        clip_preprocess=preprocess,
        batch_size=64
    )
    
    # ✅ Optimizer - visual encoder만
    optimizer = optim.AdamW(
        list(visual_encoder.parameters()) + list(classifier.parameters()),
        lr=1e-5,  # 아주 낮은 learning rate
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    # ✅ Warmup + Cosine Scheduler
    from torch.optim.lr_scheduler import LambdaLR
    
    num_epochs = 30
    warmup_epochs = 2
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: 0 → 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            import math
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 학습
    logger.info("Training full model...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # ✅ Train mode (이제 안전함)
        visual_encoder.train()
        classifier.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            image_features = visual_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = classifier(image_features.float())
            
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # ✅ Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 통계
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{train_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Scheduler step
        scheduler.step()
        
        # Epoch 통계
        train_acc = train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        val_acc = evaluate_full(visual_encoder, classifier, val_loader, device)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Val Acc: {val_acc*100:.2f}%, LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'visual_encoder': visual_encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }, 'results/checkpoints/full_finetune_best.pth')
            logger.info(f"  ✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load('results/checkpoints/full_finetune_best.pth')
    visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    all_preds, all_labels = [], []
    visual_encoder.eval()
    classifier.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            image_features = visual_encoder(images)
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
    logger.info(f"Best Val Acc: {best_val_acc*100:.2f}%")
    
    return metrics

def evaluate_full(visual_encoder, classifier, dataloader, device):
    """Validation accuracy"""
    visual_encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            image_features = visual_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = classifier(image_features.float())
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

if __name__ == '__main__':
    metrics = full_finetune_experiment()