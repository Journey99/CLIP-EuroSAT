# utils/tensorboard_utils.py

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as T

class ExperimentLogger:
    """
    TensorBoard 로깅을 위한 wrapper class
    """
    def __init__(self, log_dir, experiment_name):
        """
        Args:
            log_dir: TensorBoard 로그 디렉토리
            experiment_name: 실험 이름 (zero_shot, lora_5shot 등)
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logging to: {self.log_dir}")
        print(f"View with: tensorboard --logdir={log_dir}")
    
    def log_scalar(self, tag, value, step):
        """스칼라 값 로깅 (loss, accuracy 등)"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """여러 스칼라를 한 그래프에 표시"""
        self.writer.add_scalars(tag, value_dict, step)
    
    def log_image(self, tag, image, step):
        """이미지 로깅"""
        self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag, figure, step):
        """Matplotlib figure를 이미지로 변환해서 로깅"""
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = T.ToTensor()(image)
        self.writer.add_image(tag, image_tensor, step)
        plt.close(figure)
    
    def log_histogram(self, tag, values, step):
        """히스토그램 로깅 (파라미터 분포 등)"""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """텍스트 로깅"""
        self.writer.add_text(tag, text, step)
    
    def log_hparams(self, hparam_dict, metric_dict):
        """하이퍼파라미터와 최종 메트릭 로깅"""
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Writer 종료"""
        self.writer.close()