import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import clip
import yaml
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset

from utils import set_seed, get_dataloaders, compute_metrics, log_metrics, setup_logger
from utils.visualization import plot_confusion_matrix, plot_few_shot_curve
from models.clip_wrapper import CLIPWithLinearProbe

PROJECT_ROOT = Path(__file__).parent.parent

def few_shot_linear_experiment(shots_per_class=5, config_path=None):
    """
    Few-shot classification (Linear Probe). Few-shot은 train split 내에서만 샘플링.
    """
    # Config 로드
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logger(f'few_shot_linear_{shots_per_class}shot')
    logger.info(f"Starting Few-shot Linear Probe ({shots_per_class} shots)")

    ckpt_dir = PROJECT_ROOT / 'results' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config['experiment']['seed'])

    # Device (cuda -> mps -> cpu)
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif config['experiment']['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Device: {device}")

    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(config['model']['clip_model'], device=device)

    # 데이터 로드
    logger.info("Loading dataset...")
    data_dir = str(PROJECT_ROOT / config['data']['data_dir'])
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        clip_preprocess=preprocess,
        batch_size=config['data']['batch_size'],
        num_workers=config['experiment']['num_workers'],
    )

    # Few-shot subset: train split 내에서만 클래스당 k개 샘플링 (데이터 누수 방지)
    full_dataset = train_loader.dataset.dataset
    train_indices = train_loader.dataset.indices  # train만 허용
    few_shot_indices = full_dataset.get_few_shot_subset(
        shots_per_class, allowed_indices=train_indices
    )
    few_shot_dataset = Subset(full_dataset, few_shot_indices)

    few_shot_loader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['experiment']['num_workers'],
    )
    
    logger.info(f"Few-shot dataset: {len(few_shot_dataset)} samples")
    
    # 모델 생성 (backbone 고정)
    model = CLIPWithLinearProbe(clip_model, len(class_names), freeze_backbone=True)
    model = model.to(device)
    
    cfg_linear = config.get('few_shot_linear', {})
    num_epochs = int(cfg_linear.get('epochs', 100))
    lr = float(cfg_linear.get('learning_rate', 1e-3))

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 학습
    logger.info("Training linear probe...")
    best_val_acc = 0.0
    ckpt_path = ckpt_dir / f'linear_{shots_per_class}shot_best.pth'         

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
                torch.save(model.state_dict(), ckpt_path)

    # Test evaluation
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
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
    log_metrics(
        metrics,
        str(PROJECT_ROOT / 'results' / 'tables' / f'linear_{shots_per_class}shot_metrics.json'),
        logger,
    )
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=str(PROJECT_ROOT / 'results' / 'figures' / f'linear_{shots_per_class}shot_confusion.png'),
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
    config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    shots_list = config['few_shot']['shots_per_class']

    results = {}
    for shots in shots_list:
        metrics = few_shot_linear_experiment(shots_per_class=shots, config_path=config_path)
        results[f'{shots}-shot'] = metrics['accuracy']

    plot_few_shot_curve(
        shots_list,
        [results[f'{s}-shot'] * 100 for s in shots_list],
        save_path=str(PROJECT_ROOT / 'results' / 'figures' / 'few_shot_curve.png'),
    )