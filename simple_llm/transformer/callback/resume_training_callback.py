import torch
from .callback import Callback
from .model_checkpoint_callback import list_checkpoints

class ResumeTrainingCallback(Callback):
    """Callback для восстановления обучения с последнего чекпоинта.

    В on_train_begin загружает веса модели, состояние оптимизатора и состояния
    callback-ов из самого свежего читаемого чекпоинта. Нечитаемые (битые) файлы
    пропускаются. Если чекпоинт не подходит к архитектуре модели, выбрасывается
    исключение — иначе обучение молча началось бы с нуля.
    """

    def __init__(self, checkpoint_dir: str, resume: bool = True):
        """
        Args:
            checkpoint_dir: Путь к директории с чекпоинтами
            resume: Флаг восстановления обучения (default=True)
        """
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.last_epoch = -1

    def on_train_begin(self, model):
        self.last_epoch = -1
        if not self.resume:
            return

        for epoch, path in reversed(list_checkpoints(self.checkpoint_dir)):
            try:
                checkpoint = torch.load(path, map_location=model._device)
            except Exception as e:
                print(f"⚠️ Чекпоинт поврежден или не читается, пропускаем: {path}\n{e}")
                continue

            print(f"\n⚡ Восстанавливаем обучение из {path}")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = getattr(model, 'optimizer', None)
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            callback_states = checkpoint.get('callback_states', {})
            for cb in getattr(model, '_callbacks', []):
                state = callback_states.get(cb.__class__.__name__)
                if state is not None and hasattr(cb, 'set_state'):
                    cb.set_state(state)

            self.last_epoch = checkpoint.get('epoch', epoch)
            print(f"➔ Продолжаем с эпохи {self.last_epoch + 1}")
            if checkpoint.get('train_loss') is not None:
                print(f"➔ Последний loss: {checkpoint['train_loss']:.4f}\n")
            return

        print("Чекпоинты для восстановления не найдены, обучение начинается с нуля")
