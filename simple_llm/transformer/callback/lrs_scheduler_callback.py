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
    
    def on_epoch_begin(self, epoch, model):
        new_lr = self.base_lr * (self.decay ** epoch)
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = new_lr