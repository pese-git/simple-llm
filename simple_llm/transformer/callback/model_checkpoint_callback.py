from .callback import Callback
import torch
import os
import re
from typing import List, Optional, Tuple

_CHECKPOINT_RE = re.compile(r'^checkpoint_epoch_(\d+)\.pt$')


def checkpoint_epoch(path: str) -> Optional[int]:
    """Возвращает номер эпохи из имени файла чекпоинта или None."""
    match = _CHECKPOINT_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def list_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Список (эпоха, путь) чекпоинтов в директории, по возрастанию эпохи."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return []
    found = []
    for name in os.listdir(checkpoint_dir):
        epoch = checkpoint_epoch(name)
        if epoch is not None:
            found.append((epoch, os.path.join(checkpoint_dir, name)))
    return sorted(found)


class ModelCheckpointCallback(Callback):
    """Сохраняет чекпоинты модели во время обучения.

    Пример:
        >>> checkpoint = ModelCheckpointCallback('checkpoints/')
        >>> model.fit(callbacks=[checkpoint])

    Args:
        save_dir (str): Директория для сохранения
        save_best_only (bool): Если True, сохраняет только при улучшении loss.
            Если False, сохраняет каждые save_freq эпох (нужно для resume с последней эпохи)
        save_freq (int): Сохранять каждые N эпох при save_best_only=False (default=1)
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

    def get_state(self):
        return {'best_loss': self.best_loss}

    def set_state(self, state):
        self.best_loss = state['best_loss']

    def on_epoch_end(self, global_epoch, model, train_loss, val_loss):
        # Решаем какой loss использовать для сравнения
        current_loss = val_loss if (self.monitor == 'val' and val_loss is not None) else train_loss

        improved = current_loss < self.best_loss
        if improved:
            self.best_loss = current_loss

        if self.save_best_only:
            should_save = improved
        else:
            should_save = (global_epoch + 1) % self.save_freq == 0

        if should_save:
            checkpoint_path = os.path.join(
                self.save_dir,
                f"checkpoint_epoch_{global_epoch}.pt"
            )

            # Собираем состояния всех callback'ов
            callback_states = {}
            for cb in getattr(model, '_callbacks', []):
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
        checkpoints = list_checkpoints(self.save_dir)
        if len(checkpoints) > self.keep_last_n:
            for _, file in checkpoints[:-self.keep_last_n]:
                try:
                    os.remove(file)
                    print(f"Удалён старый чекпоинт: {file}")
                except Exception as e:
                    print(f"Ошибка при удалении чекпоинта {file}: {e}")
