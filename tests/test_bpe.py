import pytest
from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

class TestBPE:
    @pytest.fixture(params=[SimpleBPE, OptimizeBPE])
    def bpe_class(self, request):
        return request.param
    
    def test_initialization(self, bpe_class):
        """Тест инициализации BPE-токенизатора"""
        bpe = bpe_class(vocab_size=100)
        assert bpe.vocab_size == 100
        assert bpe.vocab == []
        assert bpe.token2id == {}
        assert bpe.id2token == {}
    
    def test_fit_simple_text(self, bpe_class):
        """Тест обучения на простом тексте"""
        text = "мама мыла раму"
        bpe = bpe_class(vocab_size=20)
        bpe.fit(text)
        
        # Проверки словаря
        assert isinstance(bpe.vocab, list)
        assert len(bpe.vocab) > 0
        assert len(bpe.vocab) <= 20
        assert all(isinstance(token, str) for token in bpe.vocab)
        
        # Проверка словарей
        assert len(bpe.vocab) == len(bpe.token2id)
        assert len(bpe.vocab) == len(bpe.id2token)
        
        # Проверка соответствия токенов и ID
        for token in bpe.vocab:
            assert bpe.token2id[token] == bpe.vocab.index(token)
            assert bpe.id2token[bpe.token2id[token]] == token

    @pytest.mark.parametrize("text,expected_size", [
        ("", 0),
        ("а", 1),
        ("ааааа", 2)  # Должны быть 'а' и 'аа'
    ])
    def test_edge_cases(self, bpe_class, text, expected_size):
        """Тест граничных случаев"""
        bpe = bpe_class(vocab_size=10)
        bpe.fit(text)
        assert len(bpe.vocab) == expected_size

    def test_duplicate_protection(self, bpe_class):
        """Тест защиты от дубликатов токенов"""
        bpe = bpe_class(vocab_size=50)
        bpe.fit("аааааааааа" * 100)  # Много повторений
        assert len(bpe.vocab) == len(set(bpe.vocab))
