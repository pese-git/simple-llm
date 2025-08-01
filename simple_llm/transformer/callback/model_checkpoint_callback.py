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
        keep_last_n (int): Сколько последних чекпоинтов хранить на диске (по умолчанию 3)
    """
    def __init__(self, 
                 save_dir: str, 
                 save_best_only: bool = True, 
                 save_freq: int = 1,
                 monitor: str = 'val',
                 keep_last_n: int = 3):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.keep_last_n = keep_last_n
        self.best_loss = float('inf')
        
        # Создаем директорию если её нет
        os.makedirs(save_dir, exist_ok=True)
        
    def on_epoch_end(self, global_epoch, model, train_loss, val_loss):
        # Решаем какой loss использовать для сравнения
        current_loss = val_loss if (self.monitor == 'val' and val_loss is not None) else train_loss
        
        # Сохраняем по расписанию или при улучшении
        should_save = (
            (global_epoch + 1) % self.save_freq == 0 or  # по расписанию
            (self.save_best_only and current_loss < self.best_loss)  # или если это лучшая модель
        )
        
        if should_save:
            self.best_loss = current_loss
            checkpoint_path = os.path.join(
                self.save_dir, 
                f"checkpoint_epoch_{global_epoch}.pt"
            )
            
            # Собираем состояния всех callback'ов
            callback_states = {}
            if hasattr(model, '_callbacks'):
                for cb in model._callbacks:
                    if hasattr(cb, 'get_state'):
                        callback_states[cb.__class__.__name__] = cb.get_state()

            torch.save({
                'epoch': global_epoch,
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
            self._clean_old_checkpoints()

    def _clean_old_checkpoints(self):
        import glob
        files = sorted(
            glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pt')),
            key=os.path.getmtime
        )
        if len(files) > self.keep_last_n:
            for file in files[:-self.keep_last_n]:
                try:
                    os.remove(file)
                    print(f"Удалён старый чекпоинт: {file}")
                except Exception as e:
                    print(f"Ошибка при удалении чекпоинта {file}: {e}")