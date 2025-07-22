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

## Примеры
```bash
# Запуск примеров
python -m example.multi_head_attention_example  # Визуализация внимания
python -m example.feed_forward_example         # Анализ FFN слоя
```

## Установка
```bash
git clone https://github.com/pese-git/simple-llm.git
cd simple-llm
pip install -e .
```

### Пример использования GPT
```python
from simple_llm.transformer.gpt import GPT

model = GPT(
    vocab_size=10000,
    max_seq_len=512,
    emb_size=768,
    num_heads=12,
    head_size=64,
    num_layers=6
)

# Генерация текста
output = model.generate(input_tokens, max_new_tokens=50)
```

## 🛠 How-To Guide

### 1. Работа с токенизатором
```python
from simple_llm.tokenizer import SimpleBPE

bpe = SimpleBPE().fit(text_corpus)
tokens = bpe.encode("Текст для токенизации")
```

### 2. Использование отдельных компонентов
```python
from simple_llm.transformer import MultiHeadAttention, FeedForward

attention = MultiHeadAttention(num_heads=8, emb_size=512, head_size=64)
ffn = FeedForward(emb_size=512)
```

### 3. Обучение GPT
```python
# Пример цикла обучения
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for batch in dataloader:
    logits = model(batch['input_ids'])
    loss = loss_fn(logits.view(-1, logits.size(-1)), batch['targets'].view(-1))
    loss.backward()
    optimizer.step()
```

## 📋 Системные требования

| Компонент       | Минимальные           | Рекомендуемые         |
|----------------|----------------------|----------------------|
| **Процессор**   | x86-64               | 8+ ядер              |
| **Память**      | 8GB RAM              | 16GB+ RAM            |
| **GPU**         | Не требуется         | NVIDIA (8GB+ VRAM)   |
| **ОС**          | Linux/MacOS/Windows  | Linux                |

## 📚 Документация

- [Архитектура GPT](/doc/gpt_documentation_ru.md)
- [Алгоритм BPE](/doc/bpe_algorithm.md)
- [MultiHeadAttention](/doc/multi_head_attention_ru.md)
- [Decoder](/doc/decoder_ru.md)

## 🧪 Примеры
```bash
# Запуск примеров
python -m example.example_gpt           # Генерация текста
python -m example.multi_head_attention  # Визуализация внимания
python -m example.decoder_example       # Демонстрация декодера
```

## 🤝 Участие в разработке
PR и issues приветствуются! Перед внесением изменений:
1. Создайте issue с описанием
2. Сделайте fork репозитория
3. Откройте Pull Request

## 📜 Лицензия
MIT License. Подробнее в [LICENSE](LICENSE).
