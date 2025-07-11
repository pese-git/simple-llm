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

# Токенизация
tokens = bpe.tokenize(text)
print(tokens)
```

## Интеграция в проект

Добавьте в ваш `requirements.txt`:
```
git+https://github.com/yourusername/simple-llm.git
```

Или установите напрямую:
```bash
pip install git+https://github.com/yourusername/simple-llm.git
```

## Примеры

Дополнительные примеры использования смотрите в папке [example](/example):
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
