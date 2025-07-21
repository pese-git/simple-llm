"""Пример использования декодера из архитектуры Transformer.

Этот пример демонстрирует:
1. Создание экземпляра декодера
2. Прямой проход через декодер
3. Работу с маской внимания
4. Инкрементальное декодирование
"""
import torch
from simple_llm.transformer.decoder import Decoder

def main():
    # Конфигурация
    num_heads = 4
    emb_size = 64
    head_size = 32
    max_seq_len = 128
    batch_size = 2
    seq_len = 10

    # 1. Создаем декодер
    decoder = Decoder(
        num_heads=num_heads,
        emb_size=emb_size,
        head_size=head_size,
        max_seq_len=max_seq_len
    )
    print("Декодер успешно создан:")
    print(decoder)
    print(f"Всего параметров: {sum(p.numel() for p in decoder.parameters()):,}")

    # 2. Создаем тестовые данные
    x = torch.randn(batch_size, seq_len, emb_size)
    print(f"\nВходные данные: {x.shape}")

    # 3. Прямой проход без маски
    output = decoder(x)
    print(f"\nВыход без маски: {output.shape}")

    # 4. Прямой проход с маской
    mask = torch.tril(torch.ones(seq_len, seq_len))  # Нижнетреугольная маска
    masked_output = decoder(x, mask)
    print(f"Выход с маской: {masked_output.shape}")

    # 5. Инкрементальное декодирование (имитация генерации)
    print("\nИнкрементальное декодирование:")
    for i in range(1, 5):
        step_input = x[:, :i, :]
        step_mask = torch.tril(torch.ones(i, i))
        step_output = decoder(step_input, step_mask)
        print(f"Шаг {i}: вход {step_input.shape}, выход {step_output.shape}")

if __name__ == "__main__":
    main()
