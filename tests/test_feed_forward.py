import torch
import pytest
from simple_llm.transformer.feed_forward import FeedForward

class TestFeedForward:
    @pytest.fixture
    def ff_layer(self):
        return FeedForward(emb_size=512)

    def test_initialization(self, ff_layer):
        assert isinstance(ff_layer.net, torch.nn.Sequential)
        assert len(ff_layer.net) == 4
        assert isinstance(ff_layer.net[0], torch.nn.Linear)
        assert isinstance(ff_layer.net[1], torch.nn.ReLU)
        assert isinstance(ff_layer.net[2], torch.nn.Linear)
        assert isinstance(ff_layer.net[3], torch.nn.Dropout)
        
        assert ff_layer.net[0].in_features == 512
        assert ff_layer.net[0].out_features == 2048
        assert ff_layer.net[2].in_features == 2048
        assert ff_layer.net[2].out_features == 512

    def test_forward_pass_shape(self, ff_layer):
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 512)
        output = ff_layer(x)
        
        assert output.shape == (batch_size, seq_len, 512)

    def test_dropout_training(self):
        ff_layer = FeedForward(512, dropout=0.5)
        ff_layer.train()
        x = torch.randn(2, 5, 512)
        output = ff_layer(x)
        
        # Проверяем, что dropout действительно работает в режиме обучения
        layers = ff_layer.net
        no_dropout = layers[2](layers[1](layers[0](x)))
        assert not torch.allclose(output, no_dropout)

    def test_dropout_eval(self):
        ff_layer = FeedForward(512, dropout=0.5)
        ff_layer.eval()
        x = torch.randn(2, 5, 512)
        output = ff_layer(x)
        
        # В eval режиме dropout не должен работать
        layers = ff_layer.net
        expected = layers[2](layers[1](layers[0](x)))
        assert torch.allclose(output, expected)

    def test_dtype_preservation(self, ff_layer):
        x = torch.randn(2, 5, 512, dtype=torch.float64)
        output = ff_layer(x)
        assert output.dtype == torch.float64
