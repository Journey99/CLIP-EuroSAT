import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import clip
import json
import yaml
from pathlib import Path
from tqdm import tqdm

from utils import set_seed, get_dataloaders, compute_metrics, log_metrics, setup_logger
from utils.visualization import plot_confusion_matrix
from models.clip_wrapper import CLIPClassifier

PROJECT_ROOT = Path(__file__).parent.parent

def zero_shot_experiment(config_path=None):
    """
    Zero-shot classification 실험
    """
    # Config 로드
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Logger 설정
    logger = setup_logger('zero_shot')
    logger.info("Starting Zero-shot Experiment")

    # Seed 고정
    set_seed(config['experiment']['seed'])

    # Device 설정
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif config['experiment']['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Device: {device}")

    # Prompts 로드
    with open(PROJECT_ROOT / 'prompts' / 'eurosat_prompts.json', 'r') as f:
        prompt_config = json.load(f)

    class_names = prompt_config['class_names']
    templates = prompt_config['templates']

    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(config['model']['clip_model'], device=device)

    # Classifier 생성
    model = CLIPClassifier(clip_model, class_names, templates)
    model = model.to(device)
    model.eval()

    # 데이터 로드
    logger.info("Loading dataset...")
    _, _, test_loader, _ = get_dataloaders(
        data_dir=str(PROJECT_ROOT / config['data']['data_dir']),
        clip_preprocess=preprocess,
        batch_size=config['data']['batch_size'],
        num_workers=config['experiment']['num_workers'],
        class_names=class_names
    )
    
    # Evaluation
    logger.info("Running zero-shot classification...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Zero-shot"):
            images = images.to(device)
            
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Metrics 계산
    metrics = compute_metrics(all_labels, all_preds, class_names)
    log_metrics(metrics, str(PROJECT_ROOT / 'results' / 'tables' / 'zero_shot_metrics.json'), logger)

    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=str(PROJECT_ROOT / 'results' / 'figures' / 'zero_shot_confusion.png')
    )
    
    logger.info("Zero-shot experiment completed!")
    
    return metrics

if __name__ == '__main__':
    metrics = zero_shot_experiment()