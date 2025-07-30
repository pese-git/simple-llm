# /Users/sergey/Projects/ML/simple-llm/simple_llm/transformer/callback/resume_training_callback.py
import os
import torch
from typing import Optional
from .callback import Callback

class ResumeTrainingCallback(Callback):
    """Callback для восстановления обучения с последнего чекпоинта"""
    
    def __init__(self, checkpoint_dir: str, resume: bool = True):
        """
        Args:
            checkpoint_dir: Путь к директории с чекпоинтами
            resume: Флаг восстановления обучения (default=True)
        """
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.last_epoch = -1
        
    def on_train_begin(self, model):
        if not self.resume:
            return
            
        checkpoint_path = self._find_latest_checkpoint()
        if checkpoint_path:
            print(f"\n⚡ Восстанавливаем обучение из {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=model._device)
            
            # Убедимся, что загружаем на правильное устройство
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                if hasattr(model, 'scheduler'):
                    model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.last_epoch = checkpoint.get('epoch', -1)
            
            print(f"➔ Продолжаем с эпохи {self.last_epoch + 1}")
            print(f"➔ Последний loss: {checkpoint.get('train_loss', 'N/A'):.4f}\n")
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        if not os.path.exists(self.checkpoint_dir):
            return None
            
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            return None
            
        # Сортируем по времени создания
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])