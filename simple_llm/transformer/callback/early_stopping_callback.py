from .callback import Callback
import torch

class EarlyStoppingCallback(Callback):
    """Останавливает обучение, если loss не улучшается.
    
    Пример:
        >>> early_stopping = EarlyStoppingCallback(patience=3)
        >>> model.fit(callbacks=[early_stopping])
    
    Args:
        patience (int): Количество эпох без улучшения
        min_delta (float): Минимальное значимое улучшение loss
    """
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch, model, train_loss, val_loss):
        current_loss = val_loss if val_loss else train_loss
        
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                return True  # Остановить обучение
        return False