# HeadAttention - Механизм самовнимания одной головы

## Назначение
Модуль реализует механизм внимания одной головы из архитектуры Transformer. Основные применения:
- Моделирование зависимостей в последовательностях
- Обработка естественного языка (NLP)
- Генерация текста с учетом контекста
- Анализ временных рядов

## Алгоритм работы

```mermaid
flowchart TD
    A[Входной тензор x] --> B[Вычисление Q, K, V]
    B --> C["Scores = Q·Kᵀ / √d_k"]
    C --> D[Применение нижнетреугольной маски]
    D --> E[Softmax]
    E --> F[Взвешенная сумма значений V]
    F --> G[Выходной тензор]
```

1. **Линейные преобразования**:
   ```python
   Q = W_q·x, K = W_k·x, V = W_v·x
   ```

2. **Вычисление attention scores**:
   ```python
   scores = matmul(Q, K.transpose(-2, -1)) / sqrt(head_size)
   ```

3. **Маскирование**:
   ```python
   scores.masked_fill_(mask == 0, -inf)  # Causal masking
   ```

4. **Взвешивание**:
   ```python
   weights = softmax(scores, dim=-1)
   output = matmul(weights, V)
   ```

## Пример использования
```python
import torch
from simple_llm.transformer.head_attention import HeadAttention

# Параметры
emb_size = 512
head_size = 64
max_seq_len = 1024

# Инициализация
attn_head = HeadAttention(emb_size, head_size, max_seq_len)

# Пример входа (batch_size=2, seq_len=10)
x = torch.randn(2, 10, emb_size)
output = attn_head(x)  # [2, 10, head_size]
```

## Особенности реализации

### Ключевые компоненты
| Компонент       | Назначение                          |
|-----------------|-------------------------------------|
| `self._q`       | Линейный слой для Query             |
| `self._k`       | Линейный слой для Key               |
| `self._v`       | Линейный слой для Value             |
| `self._tril_mask`| Нижнетреугольная маска             |

### Ограничения
- Требует O(n²) памяти для матрицы внимания
- Поддерживает только causal-режим
- Фиксированный максимальный размер последовательности

## Рекомендации по использованию
1. Размер головы (`head_size`) обычно выбирают 64-128
2. Для длинных последовательностей (>512) используйте оптимизации:
   - Локальное внимание
   - Разреженные паттерны
3. Сочетайте с MultiHeadAttention для лучшего качества

[Дополнительные примеры](/example/attention_examples.py)
