# Simple-LLM Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange)]()

> **Актуально для Simple-LLM v1.0 (июль 2025)**

---

## Установка

Рекомендуется использовать виртуальное окружение (venv) для изоляции зависимостей и корректной работы импортов.

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

Также вы можете вручную установить необходимые зависимости:

```bash
pip install torch numpy dill
```

Если появится файл `requirements.txt`, используйте:

```bash
pip install -r requirements.txt
```

Если вы хотите использовать последнюю версию из PyPI:

```bash
pip install simple-llm
```

Если возникают ошибки с импортами, убедитесь, что пакет установлен через pip и вы находитесь в активированном виртуальном окружении.

### Основные зависимости
- torch
- numpy
- dill

**Краткая инструкция по обучению на своих данных:**
1. Обучите BPE-токенизатор на тексте (см. `simple_llm.tokenizer.bpe.BPE`).
2. Токенизируйте корпус и создайте датасет через `GetData`.
3. Инициализируйте модель `GPT` с нужными параметрами.
4. Обучите модель одной строкой: `model.fit(train_loader, num_epoch=10)`.
5. Для подробной инструкции и примеров см. [документацию](doc/train_on_custom_data_ru.md).

---

**Структура README:**
- Обзор
- Быстрый старт
- Основные компоненты
- Документация
- Тестирование
- Как внести вклад
- Лицензия
- [FAQ](#faq)

---

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

2. Запустите примеры:
```bash
# Пример генерации текста
python example/example_gpt.py

# Пример обучения модели
python example/train_gpt_example.py
```

## 🧠 Основные компоненты

### Обработка данных
```python
from simple_llm.data.get_data import GetData

dataset = GetData(
    data=[1, 2, 3, 4, 5],  # Входная последовательность
    seq_len=3,             # Длина окна
    device="cuda"          # Устройство (опционально)
)
```

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

### Обучение модели
```python
from torch.utils.data import DataLoader

# Данные должны быть в формате (input_ids, targets)
# targets - это input_ids, сдвинутые на 1 токен вперед
train_loader = DataLoader(...) 

model.fit(
    train_loader=train_loader,  # Обучающие данные (обязательно)
    valid_loader=None,          # Валидационные данные (опционально)
    num_epoch=10,               # Количество эпох
    learning_rate=0.001         # Скорость обучения
)

# Сохранение модели
model.save("model.pt")

# Загрузка модели
loaded_model = GPT.load("model.pt", device="cuda")
```

**Требования к данным:**
- Формат: `(input_ids, targets)` где `targets = roll(input_ids, -1)`
- `input_ids`: тензор формы `[batch_size, seq_len]`
- Поддерживаются как синтетические, так и реальные текстовые данные

## 📚 Документация

Полная документация доступна в [doc/](./doc/):
- [Архитектура GPT](./doc/gpt_documentation_ru.md)
- [Алгоритм BPE](./doc/bpe_algorithm.md)
- [Обработка последовательностей](./doc/get_data_documentation_ru.md)
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

---

## ❓ FAQ

**Q: Как установить Simple-LLM, чтобы работали все импорты?**
A: Рекомендуется установить через pip (локально: `pip install .` или с PyPI: `pip install simple-llm`). Тогда все примеры и импорты будут работать из любой директории.

**Q: Как запустить Simple-LLM на CPU?**
A: Передайте параметр `device="cpu"` при инициализации модели или обработке данных.

**Q: Как использовать свой датасет?**
A: Используйте класс `GetData` из `simple_llm.data.get_data` для подготовки своих последовательностей. Следуйте формату `(input_ids, targets)`.

**Q: Где посмотреть примеры?**
A: В папке [`example/`](./example/) есть скрипты генерации и обучения.

**Q: Ошибка CUDA out of memory!**
A: Уменьшите размер batch_size или размерность модели, либо используйте CPU.

**Q: Как добавить новый модуль или улучшение?**
A: Ознакомьтесь с документацией, следуйте рекомендациям по вкладу и открывайте Pull Request.

---

## 📜 Лицензия
Распространяется под лицензией MIT. См. [LICENSE](./LICENSE)
