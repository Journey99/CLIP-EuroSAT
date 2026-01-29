import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    
    W = W0 + BA, where B: (d, r), A: (r, k)
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        x @ W = x @ W0 + x @ (B @ A) * scaling
        """
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

def inject_lora_to_linear(linear_layer, config):
    """
    기존 Linear layer에 LoRA 추가
    """
    class LinearWithLoRA(nn.Module):
        def __init__(self, original_linear, lora_config):
            super().__init__()
            self.original_linear = original_linear
            self.lora = LoRALayer(
                original_linear.in_features,
                original_linear.out_features,
                r=lora_config.r,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout
            )
            
            # Original weight 고정
            for param in self.original_linear.parameters():
                param.requires_grad = False
        
        def forward(self, x):
            return self.original_linear(x) + self.lora(x)
    
    return LinearWithLoRA(linear_layer, config)

def apply_lora_to_model(model, config, target_modules=['q_proj', 'v_proj']):
    """
    모델의 특정 모듈에 LoRA 적용
    
    Args:
        model: CLIP 모델
        config: LoRAConfig
        target_modules: LoRA를 적용할 모듈 이름
    """
    for name, module in model.named_modules():
        # Attention의 q_proj, v_proj에 LoRA 적용
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 부모 모듈 찾기
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                
                # LoRA layer로 교체
                setattr(parent, child_name, inject_lora_to_linear(module, config))
    
    # LoRA 파라미터만 학습 가능하도록
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if 'lora' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    
    print(f"\nLoRA Injection:")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
    print(f"  All params: {all_params:,}")
    
    return model