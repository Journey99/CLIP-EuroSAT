# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os

def compute_metrics(y_true, y_pred, class_names):
    """
    분류 성능 지표 계산
    
    Returns:
        metrics: dict with accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': {
            class_names[i]: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int(support[i])
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics

def log_metrics(metrics, save_path=None, logger=None):
    """
    메트릭 출력 및 저장
    """
    msg = f"\nMetrics:\n"
    msg += f"  Accuracy:  {metrics['accuracy']*100:.2f}%\n"
    msg += f"  Precision: {metrics['precision']*100:.2f}%\n"
    msg += f"  Recall:    {metrics['recall']*100:.2f}%\n"
    msg += f"  F1 Score:  {metrics['f1']*100:.2f}%\n"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # JSON 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {save_path}")
    
    return metrics