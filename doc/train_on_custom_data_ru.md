# Обучение токенизатора и модели Simple-LLM на своих данных

> **Инструкция актуальна для Simple-LLM v1.0 (июль 2025)**

---

## Оглавление
- [1. Подготовка корпуса](#1-подготовка-корпуса)
- [2. Обучение BPE-токенизатора](#2-обучение-bpe-токенизатора)
- [3. Токенизация корпуса](#3-токенизация-корпуса)
- [4. Создание датасета](#4-создание-датасета)
- [5. Обучение модели с помощью fit()](#5-обучение-модели-с-помощью-fit)
- [6. Сохранение и генерация](#6-сохранение-и-генерация)
- [7. Советы и FAQ](#7-советы-и-faq)

---

## 1. Подготовка корпуса
- Соберите тексты в один или несколько `.txt` файлов.
- Очистите данные при необходимости.

## 2. Обучение BPE-токенизатора
```python
import torch
# Автоматический выбор устройства
if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon
else:
    device = 'cpu'
print(f"Используется устройство: {device}")

from simple_llm.tokenizer.bpe import BPE

with open('corpus.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()

tokenizer = BPE(vocab_size=5000)
tokenizer.train(texts, vocab_size=5000, min_freq=2)
tokenizer.save('bpe_tokenizer.json')
```

## 3. Токенизация корпуса
```python
from simple_llm.tokenizer.bpe import BPE
import pickle

tokenizer = BPE.load('bpe_tokenizer.json')
with open('corpus.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
tokenized = [tokenizer.encode(line) for line in lines]

with open('corpus_tokens.pkl', 'wb') as f:
    pickle.dump(tokenized, f)
```

## 4. Создание датасета
```python
from simple_llm.data.get_data import GetData
import pickle

with open('corpus_tokens.pkl', 'rb') as f:
    tokenized = pickle.load(f)
all_tokens = [token for line in tokenized for token in line]
seq_len = 64
dataset = GetData(data=all_tokens, seq_len=seq_len, device='cuda')
```

## 5. Обучение модели с помощью fit()
```python
from torch.utils.data import DataLoader
from simple_llm.transformer.gpt import GPT

loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = GPT(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=seq_len,
    emb_size=256,
    num_heads=4,
    head_size=64,
    num_layers=4,
    device='cuda'
)

# Обучение одной строкой!
model.fit(
    train_loader=loader,
    valid_loader=None,    # можно передать DataLoader для валидации
    num_epoch=10,
    learning_rate=1e-4
)

print('Train loss:', model.train_loss)
```

## 6. Сохранение и генерация
```python
import torch
# Сохранить веса
torch.save(model.state_dict(), 'simple_llm_gpt.pth')

# Генерация текста после обучения
from simple_llm.tokenizer.bpe import BPE

# Загрузим токенизатор и модель (если нужно)
tokenizer = BPE.load('bpe_tokenizer.json')
# model.load_state_dict(torch.load('simple_llm_gpt.pth'))  # если требуется загрузка
model.eval()

# Пример: сгенерировать продолжение для строки prompt
prompt = "Привет, мир! "
input_ids = torch.tensor([tokenizer.encode(prompt)], device=model._device)
output = model.generate(
    x=input_ids,
    max_new_tokens=30,
    do_sample=True,
    temperature=1.0
)
# Декодируем результат
result = tokenizer.decode(output[0].tolist())
print("Сгенерированный текст:", result)
```

## 7. Советы и FAQ
- Используйте GPU для ускорения обучения.
- Размер словаря токенизатора должен совпадать с vocab_size модели.
- Для генерации текста используйте метод `generate` и декодируйте результат.
- Для валидации можно передать valid_loader в fit().
- Ошибки по размерностям чаще всего связаны с некорректными параметрами seq_len, batch_size или vocab_size.

---

**Полезные ссылки:**
- [Документация по классу GetData](./get_data_documentation_ru.md)
- [Документация по GPT](./gpt_documentation_ru.md)
- [README.md](../README.md)
