from .callback import Callback
import torch
import os

class ModelCheckpointCallback(Callback):
    """Сохраняет чекпоинты модели во время обучения.
    
    Пример:
        >>> checkpoint = ModelCheckpointCallback('checkpoints/')
        >>> model.fit(callbacks=[checkpoint])
    
    Args:
        save_dir (str): Директория для сохранения
        save_best_only (bool): Сохранять только лучшие модели
    """
    def __init__(self, save_dir, save_best_only=True):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        
    def on_epoch_end(self, epoch, model, train_loss, val_loss):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        current_loss = val_loss if val_loss else train_loss
        
        if not self.save_best_only or current_loss < self.best_loss:
            self.best_loss = current_loss
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': current_loss
            }, path)