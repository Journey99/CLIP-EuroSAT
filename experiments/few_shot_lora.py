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
from lora.lora_config import LoRAConfig
from lora.lora_layers import apply_lora_to_model

PROJECT_ROOT = Path(__file__).parent.parent


def few_shot_lora_experiment(shots_per_class=5, config_path=None):
    """
    Few-shot classification with LoRA (train split에서만 few-shot 샘플링).
    """
    # Config 로드
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logger(f'few_shot_lora_{shots_per_class}shot')
    logger.info(f"Starting Few-shot LoRA ({shots_per_class} shots)")

    set_seed(config['experiment']['seed'])

    # Device 선택 (cuda -> mps -> cpu)
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif config['experiment']['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Device: {device}")

    # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
    log_dir = PROJECT_ROOT / 'results' / 'logs' / 'tensorboard' / f'lora_{shots_per_class}shot'
    writer = SummaryWriter(str(log_dir))

    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(config['model']['clip_model'], device=device)

    # LoRA 적용
    logger.info("Applying LoRA...")
    lora_cfg = config['lora']
    lora_config = LoRAConfig(
        r=int(lora_cfg['r']),
        alpha=int(lora_cfg['alpha']),
        dropout=float(lora_cfg['dropout']),
    )
    clip_model.visual = apply_lora_to_model(
        clip_model.visual,
        lora_config,
        target_modules=lora_cfg['target_modules'],
    )

    # Classifier 추가
    feature_dim = clip_model.visual.output_dim
    # class_names에서 자동으로 개수 계산
    logger.info("Loading dataset...")
    data_dir = str(PROJECT_ROOT / config['data']['data_dir'])
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        clip_preprocess=preprocess,
        batch_size=config['data']['batch_size'],
        num_workers=config['experiment']['num_workers'],
    )
    num_classes = len(class_names)
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    # Few-shot subset (train split 내에서만 샘플링)
    logger.info(f"Creating {shots_per_class}-shot subset from train split...")
    full_dataset = train_loader.dataset.dataset
    train_indices = train_loader.dataset.indices
    few_shot_indices = full_dataset.get_few_shot_subset(
        shots_per_class, allowed_indices=train_indices
    )
    few_shot_dataset = Subset(full_dataset, few_shot_indices)

    few_shot_loader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=config['experiment']['num_workers'],
    )

    # Optimizer (LoRA params + classifier)
    lora_params = [p for n, p in clip_model.named_parameters() if 'lora' in n and p.requires_grad]
    classifier_params = list(classifier.parameters())

    lr = 5e-4
    weight_decay = 0.01
    optimizer = optim.AdamW(
        lora_params + classifier_params,
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 학습
    logger.info("Training with LoRA...")
    best_val_acc = 0.0
    global_step = 0
    ckpt_path = PROJECT_ROOT / 'results' / 'checkpoints' / f'lora_{shots_per_class}shot_best.pth'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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
        avg_train_loss = train_loss / max(len(few_shot_loader), 1)
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
                torch.save(
                    {
                        'clip_visual': clip_model.visual.state_dict(),
                        'classifier': classifier.state_dict(),
                    },
                    ckpt_path,
                )

    writer.close()

    # Test evaluation
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(ckpt_path, map_location=device)
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
    log_metrics(
        metrics,
        str(PROJECT_ROOT / 'results' / 'tables' / f'lora_{shots_per_class}shot_metrics.json'),
        logger,
    )

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        save_path=str(
            PROJECT_ROOT / 'results' / 'figures' / f'lora_{shots_per_class}shot_confusion.png'
        ),
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


if __name__ == '__main__':
    config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    shots_list = config['few_shot']['shots_per_class']
    results = {}

    for shots in shots_list:
        metrics = few_shot_lora_experiment(shots_per_class=shots, config_path=config_path)
        results[f'{shots}-shot'] = metrics['accuracy']

    # LoRA few-shot curve (Accuracy vs shots)
    plot_few_shot_curve(
        shots_list,
        [results[f'{s}-shot'] * 100 for s in shots_list],
        save_path=str(PROJECT_ROOT / 'results' / 'figures' / 'few_shot_lora_curve.png'),
    )