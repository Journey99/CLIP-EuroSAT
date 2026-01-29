# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(12, 10)):
    """
    Confusion matrix 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 정규화
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def plot_results(results_dict, save_path=None, figsize=(12, 6)):
    """
    여러 실험 결과 비교 플롯
    
    Args:
        results_dict: {'method_name': {'accuracy': 0.85, ...}, ...}
    """
    methods = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] * 100 for m in methods]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CLIP-EuroSAT: Method Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results plot saved to {save_path}")
    
    plt.show()

def plot_few_shot_curve(shots_list, accuracies, save_path=None):
    """
    Few-shot 성능 곡선
    """
    plt.figure(figsize=(10, 6))
    plt.plot(shots_list, accuracies, marker='o', linewidth=2, markersize=8)
    
    for x, y in zip(shots_list, accuracies):
        plt.text(x, y + 1, f'{y:.1f}%', ha='center', fontsize=10)
    
    plt.xlabel('Shots per Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Few-shot Learning Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Few-shot curve saved to {save_path}")
    
    plt.show()