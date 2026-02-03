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
from utils.visualization import plot_confusion_matrix
from lora.lora_config import LoRAConfig
from lora.lora_layers import apply_lora_to_model

def few_shot_lora_experiment(shots_per_class=5):
    """
    Few-shot classification with LoRA
    """
    logger = setup_logger(f'few_shot_lora_{shots_per_class}shot')
    logger.info(f"Starting Few-shot LoRA ({shots_per_class} shots)")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'results/logs/tensorboard/lora_{shots_per_class}shot')
    
    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    
    # LoRA 적용
    logger.info("Applying LoRA...")
    lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
    clip_model.visual = apply_lora_to_model(
        clip_model.visual,
        lora_config,
        target_modules=['q_proj', 'v_proj', 'out_proj']
    )
    
    # Classifier 추가
    feature_dim = clip_model.visual.output_dim
    num_classes = 10  
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 데이터 로드
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir='data/eurosat',
        clip_preprocess=preprocess,
        batch_size=32
    )
    
    # Few-shot subset
    logger.info(f"Creating {shots_per_class}-shot subset...")
    from utils.dataset import EuroSATDataset
    
    full_dataset = train_loader.dataset.dataset
    few_shot_indices = full_dataset.get_few_shot_subset(shots_per_class)
    few_shot_dataset = Subset(full_dataset, few_shot_indices)
    
    few_shot_loader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    # Optimizer (LoRA params + classifier)
    lora_params = [p for n, p in clip_model.named_parameters() if 'lora' in n and p.requires_grad]
    classifier_params = list(classifier.parameters())
    
    optimizer = optim.AdamW(
        lora_params + classifier_params,
        lr=5e-4,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 학습
    logger.info("Training with LoRA...")
    num_epochs = 50
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in range(num_epochs):
        clip_model.train()
        classifier.train()
        train_loss = 0.0
        
        for images, labels in few_shot_loader:
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
            
            # TensorBoard 로깅
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
        
        scheduler.step()
        
        # Epoch 평균 loss
        avg_train_loss = train_loss / len(few_shot_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_acc = evaluate_lora(clip_model, classifier, val_loader, device)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc*100:.2f}%")
            
            # TensorBoard 로깅
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                writer.add_scalar('Accuracy/best_val', best_val_acc, epoch)
                torch.save({
                    'clip_visual': clip_model.visual.state_dict(),
                    'classifier': classifier.state_dict()
                }, f'results/checkpoints/lora_{shots_per_class}shot_best.pth')
    
    writer.close()
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(f'results/checkpoints/lora_{shots_per_class}shot_best.pth')
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
    log_metrics(metrics, f'results/tables/lora_{shots_per_class}shot_metrics.json', logger)
    
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=f'results/figures/lora_{shots_per_class}shot_confusion.png'
    )
    
    logger.info("Few-shot LoRA experiment completed!")
    
    return metrics

def evaluate_lora(clip_model, classifier, dataloader, device):
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