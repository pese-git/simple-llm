import pytest
from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

def test_large_text_processing(bpe_class, large_text):
    """Тест обработки большого текста"""
    bpe = bpe_class(vocab_size=100)
    bpe.fit(large_text)
    
    # Проверки
    assert 50 < len(bpe.vocab) <= 100
    assert all(len(token) <= 4 for token in bpe.vocab)  # Проверка на разумную длину токенов
    assert "мама" in bpe.vocab or "ма" in bpe.vocab  # Проверка на наличие ожидаемых токенов

def test_special_characters(bpe_class):
    """Тест обработки специальных символов"""
    text = "!@#$%^&*()_+1234567890"
    bpe = bpe_class(vocab_size=30)
    bpe.fit(text)
    
    # Проверки
    assert len(bpe.vocab) > 10
    for char in set(text):
        assert any(char in token for token in bpe.vocab)  # Каждый символ должен быть в каком-то токене

def test_unicode_characters(bpe_class):
    """Тест обработки unicode-символов"""
    text = "日本語 한국어 русский English"
    bpe = bpe_class(vocab_size=50)
    bpe.fit(text)
    
    # Проверки
    assert len(bpe.vocab) > 20
    assert any("日" in token for token in bpe.vocab)
    assert any("한" in token for token in bpe.vocab)
