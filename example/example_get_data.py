"""
Пример использования класса GetData для работы с последовательными данными.

Этот пример показывает:
1. Как создать датасет из последовательности чисел
2. Как получить пары (вход, цель) для обучения
3. Как работать с разными длинами последовательностей
4. Как использовать GPU (если доступен)
"""

from simple_llm.data.get_data import GetData
import torch

def main():
    # 1. Простейший пример с последовательностью чисел
    print("\n=== Пример 1: Базовая последовательность ===")
    data = list(range(10))  # [0, 1, 2, ..., 9]
    seq_len = 3
    dataset = GetData(data=data, seq_len=seq_len)
    
    print(f"Длина датасета: {len(dataset)}")
    for i in range(min(3, len(dataset))):  # Покажем первые 3 примера
        x, y = dataset[i]
        print(f"Пример {i}:")
        print(f"  Вход: {x.tolist()} → Цель: {y.tolist()}")

    # 2. Пример с текстовыми данными (последовательность токенов)
    print("\n=== Пример 2: Токенизированный текст ===")
    text_tokens = [10, 20, 30, 40, 50, 60, 70]  # Пример токенов
    text_seq_len = 2
    text_dataset = GetData(data=text_tokens, seq_len=text_seq_len)
    
    print(f"Длина датасета: {len(text_dataset)}")
    for i in range(len(text_dataset)):
        x, y = text_dataset[i]
        print(f"Пример {i}: {x.tolist()} → {y.tolist()}")

    # 3. Пример с использованием GPU (если доступен)
    print("\n=== Пример 3: Работа с GPU ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    
    gpu_dataset = GetData(data=data, seq_len=seq_len, device=device)
    x, y = gpu_dataset[0]
    print(f"Пример на {device}:")
    print(f"  Вход: {x.tolist()} (устройство: {x.device})")
    print(f"  Цель: {y.tolist()} (устройство: {y.device})")

    # 4. Пример обработки ошибок
    print("\n=== Пример 4: Обработка ошибок ===")
    try:
        GetData(data=[1, 2, 3], seq_len=4)  # Слишком длинная последовательность
    except ValueError as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
