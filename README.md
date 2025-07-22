# Simple-LLM Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange)]()

Простая и понятная реализация языковой модели GPT-стиля с нуля на PyTorch

## 🔍 Обзор

Simple-LLM предоставляет:
- Полную реализацию архитектуры GPT
- Эффективный токенизатор BPE
- Модули трансформера (внимание, FFN, эмбеддинги)
- Гибкую систему генерации текста
- Примеры использования и документацию

## 🚀 Быстрый старт

1. Установите зависимости:
```bash
pip install torch numpy tqdm
```

2. Запустите пример генерации:
```bash
python example/example_gpt.py
```

## 🧠 Основные компоненты

### Модель GPT
```python
from simple_llm.transformer.gpt import GPT

model = GPT(
    vocab_size=10000,
    max_seq_len=512,
    emb_size=768,
    num_heads=12,
    num_layers=6
)
```

### Генерация текста
```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.9,
    top_k=50,
    top_p=0.9
)
```

## 📚 Документация

Полная документация доступна в [doc/](./doc/):
- [Архитектура GPT](./doc/gpt_documentation_ru.md)
- [Алгоритм BPE](./doc/bpe_algorithm.md)
- [Примеры использования](./example/)

## 🛠 Тестирование
```bash
pytest tests/
```

## 🤝 Как внести вклад
1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/AmazingFeature`)
3. Сделайте коммит (`git commit -m 'Add some AmazingFeature'`)
4. Запушьте ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📜 Лицензия
Распространяется под лицензией MIT. См. [LICENSE](./LICENSE)
