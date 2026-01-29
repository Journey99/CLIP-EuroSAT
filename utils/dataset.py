# utils/dataset.py

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

class EuroSATDataset(Dataset):
    """
    EuroSAT 위성 이미지 데이터셋
    """
    def __init__(self, root_dir, transform=None, class_names=None):
        """
        Args:
            root_dir: data/eurosat 경로
            transform: 이미지 전처리
            class_names: 클래스 이름 리스트 (순서 고정용)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 클래스 이름 로드
        if class_names is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        else:
            self.classes = class_names
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 모든 이미지 경로 수집
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, class_idx))
        
        print(f"EuroSAT Dataset:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Class names: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_few_shot_subset(self, shots_per_class):
        """
        Few-shot 학습용 서브셋 생성
        
        Args:
            shots_per_class: 클래스당 샘플 수
        
        Returns:
            few_shot_indices: 선택된 샘플 인덱스
        """
        np.random.seed(42)  # 재현성
        
        few_shot_indices = []
        
        for class_idx in range(len(self.classes)):
            # 해당 클래스의 모든 인덱스
            class_indices = [i for i, (_, label) in enumerate(self.samples) if label == class_idx]
            
            # 랜덤 샘플링
            selected = np.random.choice(class_indices, size=shots_per_class, replace=False)
            few_shot_indices.extend(selected)
        
        return few_shot_indices

def get_dataloaders(
    data_dir='data/eurosat',
    clip_preprocess=None,
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    num_workers=4,
    class_names=None
):
    """
    Train/Val/Test DataLoader 생성
    
    Args:
        data_dir: EuroSAT 데이터 경로
        clip_preprocess: CLIP 전처리 함수
        batch_size: 배치 크기
        val_split: Validation 비율
        test_split: Test 비율
        num_workers: 데이터 로더 워커 수
        class_names: 클래스 이름 (고정 순서용)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # 데이터셋 생성
    dataset = EuroSATDataset(data_dir, transform=clip_preprocess, class_names=class_names)
    
    # Train/Val/Test 분할
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader Info:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, dataset.classes
