"""
Пример использования MultiHeadAttention с визуализацией весов внимания
"""

import torch
import matplotlib.pyplot as plt
import os
from simple_llm.transformer.multi_head_attention import MultiHeadAttention

def plot_attention_heads(weights, num_heads, filename="multi_head_attention.png"):
    """Визуализация матриц внимания для всех голов"""
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    for i in range(num_heads):
        ax = axes[i]
        img = ax.imshow(weights[0, i].detach().numpy(), cmap='viridis')
        ax.set_title(f'Голова {i+1}')
        ax.set_xlabel('Key позиции')
        ax.set_ylabel('Query позиции') 
        fig.colorbar(img, ax=ax)
    plt.suptitle('Матрицы внимания по головам')
    plt.tight_layout()
    
    # Создаем папку если нет
    os.makedirs('example_output', exist_ok=True)
    plt.savefig(f'example_output/{filename}')
    plt.close()

def main():
    # Конфигурация
    num_heads = 4
    emb_size = 64
    head_size = 16
    seq_len = 10
    batch_size = 1

    # Инициализация
    torch.manual_seed(42)
    mha = MultiHeadAttention(
        num_heads=num_heads,
        emb_size=emb_size,
        head_size=head_size,
        max_seq_len=20
    )

    # Тестовые данные
    x = torch.randn(batch_size, seq_len, emb_size)

    # Прямой проход
    output = mha(x)
    # Получаем веса из всех голов
    attn_weights = torch.stack([head.get_attention_weights(x) for head in mha._heads], dim=1)
    
    print("Входная форма:", x.shape)
    print("Выходная форма:", output.shape) 
    print("Форма весов внимания:", attn_weights.shape)

    # Визуализация
    plot_attention_heads(attn_weights, num_heads)

    # Демонстрация работы механизма
    print("\nПример с ручными зависимостями:")
    x = torch.zeros(batch_size, 3, emb_size)
    x[:, 1, :] = 2.0  # Яркий токен
    
    # Получаем веса внимания
    weights = torch.stack([head.get_attention_weights(x) for head in mha._heads], dim=1)
    for head in range(num_heads):
        print(f"Голова {head+1} веса для токена 1:", 
              weights[0, head, 1].detach().round(decimals=3))

if __name__ == "__main__":
    main()
    print("\nГотово! Графики сохранены в example_output/multi_head_attention.png")
