"""
Callback-система для управления обучением GPT.

Доступные callback-и:
- EarlyStoppingCallback - ранняя остановка
- ModelCheckpointCallback - сохранение чекпоинтов  
- LRSchedulerCallback - регулировка learning rate
"""

# /Users/sergey/Projects/ML/simple-llm/simple_llm/transformer/callback/__init__.py
from .callback import Callback
from .early_stopping_callback import EarlyStoppingCallback
from .lrs_scheduler_callback import LRSchedulerCallback
from .model_checkpoint_callback import ModelCheckpointCallback
from .resume_training_callback import ResumeTrainingCallback

__all__ = [
    'Callback',
    'EarlyStoppingCallback',
    'LRSchedulerCallback', 
    'ModelCheckpointCallback',
    'ResumeTrainingCallback' 
]