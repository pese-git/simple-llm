from .callback import Callback
import torch
import os
from typing import Optional

class ModelCheckpointCallback(Callback):
    """Сохраняет чекпоинты модели во время обучения.
    
    Пример:
        >>> checkpoint = ModelCheckpointCallback('checkpoints/')
        >>> model.fit(callbacks=[checkpoint])
    
    Args:
        save_dir (str): Директория для сохранения
        save_best_only (bool): Если True, сохраняет только при улучшении loss
        save_freq (int): Сохранять каждые N эпох (default=1)
        monitor (str): Какой loss мониторить ('val' или 'train')
    """
    def __init__(self, 
                 save_dir: str, 
                 save_best_only: bool = True, 
                 save_freq: int = 1,
                 monitor: str = 'val'):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.best_loss = float('inf')
        
        # Создаем директорию если её нет
        os.makedirs(save_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, model, train_loss, val_loss):
        # Решаем какой loss использовать для сравнения
        current_loss = val_loss if (self.monitor == 'val' and val_loss is not None) else train_loss
        
        # Сохраняем по расписанию или при улучшении
        should_save = (
            (epoch + 1) % self.save_freq == 0 or  # по расписанию
            (self.save_best_only and current_loss < self.best_loss)  # или если это лучшая модель
        )
        
        if should_save:
            self.best_loss = current_loss
            checkpoint_path = os.path.join(
                self.save_dir, 
                f"checkpoint_epoch_{epoch}.pt"
            )
            
            # Собираем состояния всех callback'ов
            callback_states = {}
            if hasattr(model, '_callbacks'):
                for cb in model._callbacks:
                    if hasattr(cb, 'get_state'):
                        callback_states[cb.__class__.__name__] = cb.get_state()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': self.best_loss,
                'callback_states': callback_states,
                'config': {
                    'vocab_size': model._vocab_size,
                    'max_seq_len': model._max_seq_len,
                    'emb_size': model._emb_size,
                    'num_heads': model._num_heads,
                    'head_size': model._head_size,
                    'num_layers': model._num_layers
                }
            }, checkpoint_path)
            
            print(f"Модель сохранена в {checkpoint_path} (loss: {current_loss:.4f})")
