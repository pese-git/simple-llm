"""
Callback-система для управления обучением GPT.

Доступные callback-и:
- EarlyStoppingCallback - ранняя остановка
- ModelCheckpointCallback - сохранение чекпоинтов  
- LRSchedulerCallback - регулировка learning rate
"""

from .callback import Callback
from .early_stopping_callback import EarlyStoppingCallback
from .lrs_scheduler_callback import LRSchedulerCallback
from .model_checkpoint_callback import ModelCheckpointCallback

__all__ = [
    'Callback',
    'EarlyStoppingCallback',
    'LRSchedulerCallback',
    'ModelCheckpointCallback'
]