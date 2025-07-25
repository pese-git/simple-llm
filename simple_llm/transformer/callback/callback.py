"""
Callback-система для управления процессом обучения GPT.

Реализует паттерн Observer для мониторинга и управления обучением.
"""

class Callback:
    """Абстрактный базовый класс для всех callback-ов.
    
    Методы вызываются автоматически во время обучения:
    - on_epoch_begin - перед началом эпохи
    - on_batch_end - после обработки батча 
    - on_epoch_end - в конце эпохи
    """
    
    def on_epoch_begin(self, epoch, model):
        """Вызывается перед началом эпохи.
        
        Args:
            epoch (int): Номер текущей эпохи (0-based)
            model (GPT): Обучаемая модель GPT
        """
        pass

    def on_batch_end(self, batch, model, loss):
        """Вызывается после обработки каждого батча.
        
        Args:
            batch (int): Номер батча в текущей эпохе
            model (GPT): Обучаемая модель GPT 
            loss (float): Значение функции потерь на батче
        """
        pass

    def on_epoch_end(self, epoch, model, train_loss, val_loss):
        """Вызывается в конце эпохи.
        
        Args:
            epoch (int): Номер завершенной эпохи
            model (GPT): Обучаемая модель GPT
            train_loss (float): Средний loss на обучении
            val_loss (float|None): Средний loss на валидации или None
            
        Returns:
            bool: Если True, обучение будет прервано
        """
        pass