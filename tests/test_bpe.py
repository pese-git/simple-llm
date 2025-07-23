import pytest
from simple_llm.tokenizer.bpe import BPE

def test_basic_bpe():
    """Базовый тест работы BPE"""
    tokenizer = BPE(vocab_size=10)
    text = "мама мыла раму"
    
    # Обучение
    tokenizer.fit(text)
    
    # Проверка размера словаря
    assert len(tokenizer.vocab) == 10
    
    # Кодирование/декодирование
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == text
    assert len(encoded) > 0
    
    # Проверка неизвестных символов
    unknown_encoded = tokenizer.encode("мама мыла окно")
    assert -1 in unknown_encoded  # Специальный токен для неизвестных символов
