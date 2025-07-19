# Simple LLM Tokenizer

Простой и эффективный токенизатор для языковых моделей на основе BPE (Byte Pair Encoding)

## Описание проекта

Проект предоставляет реализации алгоритма BPE (Byte Pair Encoding) для токенизации текста:
- `SimpleBPE` - базовая версия
- `OptimizeBPE` - оптимизированная версия с улучшенной производительностью

Основные возможности:
- Обучение на любом тексте (поддержка кириллицы и других алфавитов)
- Гибкая настройка размера словаря
- Простота интеграции в существующие проекты

## Установка

1. Склонируйте репозиторий:
```bash
git clone https://github.com/yourusername/simple-llm.git
cd simple-llm
```

2. Установите пакет:
```bash
pip install -e .
```

## Быстрый старт

```python
from simple_llm.tokenizer import SimpleBPE

# Инициализация и обучение
text = "мама мыла раму, папа пил какао"
bpe = SimpleBPE(vocab_size=50)
bpe.fit(text)

# Кодирование/декодирование
encoded = bpe.encode(text)
print(f"Закодировано: {encoded}")

decoded = bpe.decode(encoded)
print(f"Декодировано: '{decoded}'")
print(f"Совпадение с оригиналом: {text == decoded}")

# Обработка неизвестных слов
unknown = bpe.encode("неизвестное_слово")
print(f"Неизвестное слово: {unknown}")
```

Пример вывода:
```
Закодировано: [12, 12, 0, 15, 8, 0, 17, 9, 1, 0, 16, 16, 0, 14, 7, 0, 10, 10, 3]
Декодировано: 'мама мыла раму, папа пил какао'
Совпадение с оригиналом: True
Неизвестное слово: [-1, -1, 3, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, 5, 7, -1, 7]
```

### Работа с эмбеддингами
```python
from simple_llm.embedding import TokenEmbeddings, PositionalEmbeddings

# Инициализация
token_emb = TokenEmbeddings(vocab_size=1000, emb_size=256)
pos_emb = PositionalEmbeddings(max_seq_len=512, emb_size=256)

# Пример использования
tokens = [1, 2, 3]  # Индексы токенов
embeddings = token_emb(tokens) + pos_emb(len(tokens))
print(f"Объединенные эмбеддинги: {embeddings.shape}")
```

## Документация
- [Токенизация BPE](/doc/bpe_algorithm.md)
- [Токенные эмбеддинги](/doc/token_embeddings_ru.md) 
- [Позиционные эмбеддинги](/doc/positional_embeddings_ru.md)

## Интеграция в проект

Добавьте в ваш `requirements.txt`:
```
git+https://github.com/yourusername/simple-llm.git
```

Или установите напрямую:
```bash
pip install git+https://github.com/yourusername/simple-llm.git
```

## Примеры использования

Дополнительные примеры:
- [Базовый BPE](/example/example_bpe.py)
- [Токенные эмбеддинги](/example/example_token_embeddings.py)
- [Механизм внимания](/example/head_attention_example.py)

Документация:
- [Токенизация](/doc/bpe_algorithm.md)
- [Эмбеддинги](/doc/token_embeddings_ru.md)
- [Внимание](/doc/head_attention_ru.md)
- Сравнение SimpleBPE и OptimizeBPE
- Работа с разными языками
- Настройка параметров токенизации

## Разработка

Для запуска тестов:
```bash
pytest tests/
```

Для внесения изменений установите зависимости разработки:
```bash
pip install -e ".[dev]"
```

## Лицензия

Проект распространяется под лицензией MIT. Подробнее см. [LICENSE](LICENSE).
