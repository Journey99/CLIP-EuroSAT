from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """LoRA 설정"""
    r: int = 8  # Rank
    alpha: int = 16  # Scaling factor
    dropout: float = 0.1
    target_modules: list = None  # ['q_proj', 'v_proj']
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'v_proj']
