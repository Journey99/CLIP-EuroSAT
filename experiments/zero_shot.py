import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import clip
import json
from tqdm import tqdm

from utils import set_seed, get_dataloaders, compute_metrics, log_metrics, setup_logger
from utils.visualization import plot_confusion_matrix
from models.clip_wrapper import CLIPClassifier

def zero_shot_experiment(config_path='config/base_config.yaml'):
    """
    Zero-shot classification 실험
    """
    # Logger 설정
    logger = setup_logger('zero_shot')
    logger.info("Starting Zero-shot Experiment")
    
    # Seed 고정
    set_seed(42)
    
    # Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Prompts 로드
    with open('prompts/eurosat_prompts.json', 'r') as f:
        prompt_config = json.load(f)
    
    class_names = prompt_config['class_names']
    templates = prompt_config['templates']
    
    # CLIP 모델 로드
    logger.info("Loading CLIP model...")
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    
    # Classifier 생성
    model = CLIPClassifier(clip_model, class_names, templates)
    model.eval()
    
    # 데이터 로드
    logger.info("Loading dataset...")
    _, _, test_loader, _ = get_dataloaders(
        data_dir='data/eurosat',
        clip_preprocess=preprocess,
        batch_size=128,
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
    log_metrics(metrics, 'results/tables/zero_shot_metrics.json', logger)
    
    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path='results/figures/zero_shot_confusion.png'
    )
    
    logger.info("Zero-shot experiment completed!")
    
    return metrics

if __name__ == '__main__':
    metrics = zero_shot_experiment()