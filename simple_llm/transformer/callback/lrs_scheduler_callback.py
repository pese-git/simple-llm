from .callback import Callback

class LRSchedulerCallback(Callback):
    """Динамически регулирует learning rate.
    
    Пример:
        >>> lr_scheduler = LRSchedulerCallback(lr=0.001)
        >>> model.fit(callbacks=[lr_scheduler])
    
    Args:
        lr (float): Начальный learning rate
        decay (float): Коэффициент уменьшения LR
    """
    def __init__(self, lr, decay=0.95):
        self.base_lr = lr
        self.decay = decay
        self.last_epoch = -1  # Добавляем отслеживание эпохи

    def get_state(self):
        return {
            'base_lr': self.base_lr,
            'decay': self.decay,
            'last_epoch': self.last_epoch
        }
        
    def set_state(self, state):
        self.base_lr = state['base_lr']
        self.decay = state['decay']
        self.last_epoch = state['last_epoch']
    
    def on_epoch_begin(self, epoch, model):
        self.last_epoch = epoch  # Сохраняем текущую эпоху
        new_lr = self.base_lr * (self.decay ** epoch)
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = new_lr