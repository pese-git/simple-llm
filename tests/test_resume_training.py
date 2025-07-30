import os
import tempfile
import torch
import pytest
from simple_llm.transformer.gpt import GPT
from simple_llm.transformer.callback import ModelCheckpointCallback, ResumeTrainingCallback
from torch.utils.data import DataLoader, TensorDataset

from simple_llm.transformer.callback import (
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
        
        assert resume_cb.last_epoch == 0
        files = os.listdir(tmpdir)
        assert len(files) == 2

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
        
        with pytest.raises(Exception):
            sample_model.fit(
                sample_data, 
                num_epoch=1, 
                callbacks=[resume_cb],
                resume_training=True
            )

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
        checkpoint_cb = ModelCheckpointCallback(tmpdir)
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
        
        files = os.listdir(tmpdir)
        assert len(files) == 3

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
        
        # Сохраняем текущий lr с учетом decay
        expected_lr = 0.001 * (0.95 ** 1)  # decay^epoch
        
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
