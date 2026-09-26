import os
import tempfile
import torch
import pytest
from simple_llm.transformer.gpt import GPT
from simple_llm.transformer.callback import ModelCheckpointCallback, ResumeTrainingCallback
from torch.utils.data import DataLoader, TensorDataset

from simple_llm.transformer.callback import (
    Callback,
    LRSchedulerCallback,
)

@pytest.fixture
def sample_data():
    # Создаем тестовые данные
    inputs = torch.randint(0, 100, (100, 10))  # 100 samples, seq_len=10
    targets = torch.randint(0, 100, (100, 10))
    return DataLoader(TensorDataset(inputs, targets), batch_size=10)

@pytest.fixture
def sample_model():
    return GPT(
        vocab_size=100,
        max_seq_len=10,
        emb_size=32,
        num_heads=4,
        head_size=8,
        num_layers=2
    )

def test_model_checkpoint_saving(sample_model, sample_data):
    """Тестирует корректность сохранения чекпоинтов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpointCallback(tmpdir, save_best_only=False)
        sample_model.fit(sample_data, num_epoch=1, callbacks=[checkpoint_cb])
        
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith('checkpoint_epoch_')
        
        checkpoint = torch.load(os.path.join(tmpdir, files[0]))
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'train_loss' in checkpoint

def test_resume_training(sample_model, sample_data):
    """Тестирует восстановление обучения из чекпоинта"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpointCallback(tmpdir, save_best_only=False)
        sample_model.fit(sample_data, num_epoch=1, callbacks=[checkpoint_cb])

        # Проверим, что чекпоинт создан
        files = os.listdir(tmpdir)
        assert any(f.startswith("checkpoint_epoch_") for f in files)

        new_model = GPT(
            vocab_size=100,
            max_seq_len=10,
            emb_size=32,
            num_heads=4,
            head_size=8,
            num_layers=2
        )

        resume_cb = ResumeTrainingCallback(tmpdir)

        # Дальнейшее обучение с fit/resume
        new_model.fit(
            sample_data,
            num_epoch=2,
            callbacks=[resume_cb, checkpoint_cb],
            resume_training=True
        )
        # Восстановились с эпохи 0 и досохранили эпохи 1 и 2
        assert resume_cb.last_epoch == 0
        files = sorted(f for f in os.listdir(tmpdir) if f.startswith('checkpoint_epoch_'))
        assert files == ['checkpoint_epoch_0.pt', 'checkpoint_epoch_1.pt', 'checkpoint_epoch_2.pt']

def test_resume_with_missing_checkpoint(sample_model, sample_data):
    """Тестирует поведение при отсутствии чекпоинтов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert len(os.listdir(tmpdir)) == 0
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        sample_model.fit(
            sample_data, 
            num_epoch=1, 
            callbacks=[resume_cb],
            resume_training=True
        )
        
        assert resume_cb.last_epoch == -1

def test_resume_with_corrupted_checkpoint(sample_model, sample_data):
    """Тестирует обработку битых чекпоинтов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_checkpoint = os.path.join(tmpdir, "checkpoint_epoch_0.pt")
        with open(bad_checkpoint, 'w') as f:
            f.write("corrupted data")
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        
        sample_model.fit(
            sample_data,
            num_epoch=1,
            callbacks=[resume_cb],
            resume_training=True
        )
        assert resume_cb.last_epoch == -1

def test_optimizer_state_restoration(sample_model, sample_data):
    """Тестирует восстановление состояния оптимизатора"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpointCallback(tmpdir)
        sample_model.fit(sample_data, num_epoch=1, callbacks=[checkpoint_cb])
        
        original_optimizer_state = sample_model.optimizer.state_dict()
        
        new_model = GPT(
            vocab_size=100,
            max_seq_len=10,
            emb_size=32,
            num_heads=4,
            head_size=8,
            num_layers=2
        )
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        new_model.fit(
            sample_data, 
            num_epoch=2, 
            callbacks=[resume_cb, checkpoint_cb],
            resume_training=True
        )
        
        assert 'state' in new_model.optimizer.state_dict()
        assert 'param_groups' in new_model.optimizer.state_dict()
        
        # Проверяем только параметры, кроме lr (так как он меняется scheduler'ом)
        for key in original_optimizer_state['param_groups'][0]:
            if key not in ['params', 'lr']:
                assert (
                    original_optimizer_state['param_groups'][0][key] == 
                    new_model.optimizer.state_dict()['param_groups'][0][key]
                )

def test_multiple_resumes(sample_model, sample_data):
    """Тестирует многократное восстановление обучения"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpointCallback(tmpdir, save_best_only=False)
        sample_model.fit(sample_data, num_epoch=1, callbacks=[checkpoint_cb])
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        sample_model.fit(
            sample_data, 
            num_epoch=2, 
            callbacks=[resume_cb, checkpoint_cb],
            resume_training=True
        )
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        sample_model.fit(
            sample_data, 
            num_epoch=3, 
            callbacks=[resume_cb, checkpoint_cb],
            resume_training=True
        )
        
        # 1 + 2 + 3 = 6 эпох (0..5), keep_last_n=3 оставляет последние три
        files = sorted(os.listdir(tmpdir))
        assert files == ['checkpoint_epoch_3.pt', 'checkpoint_epoch_4.pt', 'checkpoint_epoch_5.pt']

def test_scheduler_state_restoration(sample_model, sample_data):
    """Тестирует восстановление состояния LR"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpointCallback(tmpdir)
        lr_scheduler_cb = LRSchedulerCallback(lr=0.001)
        
        sample_model.fit(
            sample_data,
            num_epoch=1,
            callbacks=[checkpoint_cb, lr_scheduler_cb],
            learning_rate=0.001
        )
        
        # Resume с эпохи 0 + 2 эпохи => последняя эпоха имеет номер 2, lr = base * decay^2
        expected_lr = 0.001 * (0.95 ** 2)
        
        new_model = GPT(
            vocab_size=100,
            max_seq_len=10,
            emb_size=32,
            num_heads=4,
            head_size=8,
            num_layers=2
        )
        
        resume_cb = ResumeTrainingCallback(tmpdir)
        new_model.fit(
            sample_data,
            num_epoch=2,
            callbacks=[resume_cb, checkpoint_cb, lr_scheduler_cb],
            resume_training=True,
            learning_rate=0.001
        )
        
        # Проверяем что LR восстановлен с учетом decay
        assert new_model.optimizer.param_groups[0]['lr'] == pytest.approx(expected_lr)


def _new_model(emb_size=32):
    return GPT(
        vocab_size=100,
        max_seq_len=10,
        emb_size=emb_size,
        num_heads=4,
        head_size=8,
        num_layers=2
    )


class _CaptureStateCallback(Callback):
    """Запоминает веса модели сразу после on_train_begin предыдущих callback-ов"""
    def __init__(self):
        self.state = None

    def on_train_begin(self, model):
        self.state = {k: v.clone() for k, v in model.state_dict().items()}


def test_fit_restores_weights_from_checkpoint(sample_model, sample_data):
    """fit(resume_training=True) должен загрузить веса, а не только номер эпохи"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_model.fit(sample_data, num_epoch=1, checkpoint_dir=tmpdir)
        saved = torch.load(os.path.join(tmpdir, 'checkpoint_epoch_0.pt'))['model_state_dict']

        new_model = _new_model()
        capture = _CaptureStateCallback()
        new_model.fit(
            sample_data,
            num_epoch=1,
            callbacks=[capture],
            checkpoint_dir=tmpdir,
            resume_training=True
        )

        for key, value in saved.items():
            assert torch.equal(capture.state[key], value), key
        assert os.path.exists(os.path.join(tmpdir, 'checkpoint_epoch_1.pt'))


def test_resume_skips_corrupted_latest_checkpoint(sample_model, sample_data):
    """Битый последний чекпоинт пропускается, восстановление идёт с предыдущего"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_model.fit(sample_data, num_epoch=2, checkpoint_dir=tmpdir)
        saved_epoch0 = torch.load(os.path.join(tmpdir, 'checkpoint_epoch_0.pt'))['model_state_dict']
        with open(os.path.join(tmpdir, 'checkpoint_epoch_1.pt'), 'w') as f:
            f.write("corrupted data")

        new_model = _new_model()
        resume_cb = ResumeTrainingCallback(tmpdir)
        capture = _CaptureStateCallback()
        new_model.fit(
            sample_data,
            num_epoch=1,
            callbacks=[resume_cb, capture],
            checkpoint_dir=tmpdir,
            resume_training=True
        )

        assert resume_cb.last_epoch == 0
        for key, value in saved_epoch0.items():
            assert torch.equal(capture.state[key], value), key
        # Эпоха 1 обучена заново и битый файл перезаписан корректным чекпоинтом
        assert torch.load(os.path.join(tmpdir, 'checkpoint_epoch_1.pt'))['epoch'] == 1


def test_resume_with_incompatible_architecture_raises(sample_model, sample_data):
    """Чекпоинт от другой архитектуры — ошибка, а не молчаливое обучение с нуля"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_model.fit(sample_data, num_epoch=1, checkpoint_dir=tmpdir)

        other_model = _new_model(emb_size=64)
        with pytest.raises(RuntimeError):
            other_model.fit(
                sample_data,
                num_epoch=1,
                checkpoint_dir=tmpdir,
                resume_training=True
            )
        # Старые чекпоинты не тронуты
        assert os.listdir(tmpdir) == ['checkpoint_epoch_0.pt']


def test_fit_does_not_mutate_callbacks(sample_model, sample_data):
    """fit не должен дописывать стандартные callback-и в список пользователя"""
    with tempfile.TemporaryDirectory() as tmpdir:
        lr_cb = LRSchedulerCallback(lr=0.001)
        callbacks = [lr_cb]
        sample_model.fit(sample_data, num_epoch=1, callbacks=callbacks, checkpoint_dir=tmpdir)
        sample_model.fit(sample_data, num_epoch=1, callbacks=callbacks, checkpoint_dir=tmpdir)

        assert callbacks == [lr_cb]
        # Стандартный LRScheduler не добавлен поверх пользовательского
        schedulers = [cb for cb in sample_model._callbacks if isinstance(cb, LRSchedulerCallback)]
        assert schedulers == [lr_cb]
        checkpointers = [cb for cb in sample_model._callbacks if isinstance(cb, ModelCheckpointCallback)]
        assert len(checkpointers) == 1


def test_checkpoint_best_only_keeps_best_loss(sample_model, sample_data):
    """save_best_only: сохраняет только при улучшении, best_loss не затирается худшим"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_model.fit(sample_data, num_epoch=1)  # создаём optimizer
        checkpoint_cb = ModelCheckpointCallback(tmpdir, save_best_only=True)

        checkpoint_cb.on_epoch_end(0, sample_model, 1.0, None)
        checkpoint_cb.on_epoch_end(1, sample_model, 2.0, None)
        assert checkpoint_cb.best_loss == 1.0
        checkpoint_cb.on_epoch_end(2, sample_model, 0.5, None)

        assert sorted(os.listdir(tmpdir)) == ['checkpoint_epoch_0.pt', 'checkpoint_epoch_2.pt']
        assert checkpoint_cb.best_loss == 0.5
