import torch
import torch.nn as nn
import clip

class CLIPClassifier(nn.Module):
    """
    CLIP 기반 분류 모델 래퍼
    """
    def __init__(self, clip_model, class_names, templates):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.templates = templates
        
        # Text features 미리 계산 (캐싱)
        self.register_buffer('text_features', self._encode_text())
    
    def _encode_text(self):
        """
        모든 클래스의 text features 계산
        """
        device = next(self.clip_model.parameters()).device
        
        all_text_features = []
        
        for class_name in self.class_names:
            # 여러 template으로 ensemble
            texts = [template.format(class_name) for template in self.templates]
            text_tokens = clip.tokenize(texts).to(device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Template 평균
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
                
                all_text_features.append(text_features)
        
        return torch.stack(all_text_features)
    
    def forward(self, images):
        """
        이미지를 인코딩하고 텍스트 features와 비교
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Image encoding
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        logits = 100.0 * image_features @ self.text_features.T
        
        return logits

class CLIPWithLinearProbe(nn.Module):
    """
    CLIP + Linear Classifier (Few-shot)
    """
    def __init__(self, clip_model, num_classes, freeze_backbone=True):
        super().__init__()
        self.clip_model = clip_model
        
        # Backbone 고정
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Linear classifier
        feature_dim = clip_model.visual.output_dim
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, images):
        # Image features 추출
        with torch.set_grad_enabled(self.training):
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Classification
        logits = self.classifier(image_features.float())
        
        return logits