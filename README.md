# Simple LLM Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

## Основные компоненты

### Токенизация
- `SimpleBPE` - алгоритм Byte Pair Encoding
- `OptimizeBPE` - оптимизированная версия

### Эмбеддинги
- `TokenEmbeddings` - векторные представления токенов
- `PositionalEmbeddings` - позиционное кодирование

### Transformer Layers
- `HeadAttention` - механизм внимания одной головы
- `MultiHeadAttention` - многоголовое внимание (4-16 голов)
- `FeedForward` - двухслойная FFN сеть (расширение → сжатие)
- `Decoder` - полный декодер Transformer (Self-Attention + FFN)

## Быстрый старт

```python
from simple_llm import SimpleBPE, MultiHeadAttention, FeedForward

# 1. Токенизация
bpe = SimpleBPE().fit(text_corpus)
tokens = bpe.encode("Пример текста")

# 2. Полный пайплайн
model = nn.Sequential(
    TokenEmbeddings(10000, 256),
    PositionalEmbeddings(256, 512),
    MultiHeadAttention(8, 256, 32),
    FeedForward(256)
)
```

## Документация
- [Токенизация](/doc/bpe_algorithm.md)
- [MultiHeadAttention](/doc/multi_head_attention_ru.md)
- [FeedForward](/doc/feed_forward_ru.md)
- [Decoder](/doc/decoder_ru.md)

## Примеры
```bash
# Запуск примеров
python -m example.multi_head_attention_example  # Визуализация внимания
python -m example.feed_forward_example         # Анализ FFN слоя
python -m example.decoder_example              # Демонстрация декодера
```

## Установка
```bash
git clone https://github.com/pese-git/simple-llm.git
cd simple-llm
pip install -e .
```
