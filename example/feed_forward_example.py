"""
Пример использования FeedForward слоя из архитектуры Transformer

Демонстрирует:
1. Базовое применение
2. Разницу между режимами train/eval
3. Визуализацию изменений внутри сети
"""

import torch
import matplotlib.pyplot as plt
import os
from simple_llm.transformer.feed_forward import FeedForward

def plot_layer_outputs(outputs, titles, filename):
    """Визуализация выходов разных слоев"""
    plt.figure(figsize=(15, 5))
    for i, (out, title) in enumerate(zip(outputs, titles)):
        plt.subplot(1, len(outputs), i+1)
        plt.imshow(out[0].detach().numpy(), cmap='viridis', aspect='auto')
        plt.title(title)
        plt.colorbar()
    plt.tight_layout()
    
    # Создаем папку если нет
    os.makedirs('example_output', exist_ok=True)
    plt.savefig(f'example_output/{filename}')
    plt.close()

def main():
    # Конфигурация
    emb_size = 128
    dropout = 0.1
    
    # Инициализация
    ff = FeedForward(emb_size, dropout)
    print(f"Архитектура сети:\n{ff.net}")

    # Тестовые данные
    x = torch.randn(1, 20, emb_size)  # [batch, seq_len, emb_size]
    
    # 1. Базовый forward pass
    output = ff(x)
    print(f"\nФорма входа: {x.shape} -> Форма выхода: {output.shape}")

    # 2. Сравнение режимов train/eval
    ff.train()
    train_out = ff(x)
    ff.eval()
    eval_out = ff(x)
    diff = torch.abs(train_out - eval_out).max().item()
    print(f"\nМаксимальное расхождение (train vs eval): {diff:.6f}")

    # 3. Визуализация преобразований
    with torch.no_grad():
        # Получаем выходы каждого слоя
        layer1_out = ff.net[0](x)
        relu_out = ff.net[1](layer1_out)
        layer2_out = ff.net[2](relu_out)
        
        plot_layer_outputs(
            outputs = [x, layer1_out, relu_out, layer2_out],
            titles = [
                'Входные данные', 
                'После первого Linear', 
                'После ReLU', 
                'После второго Linear'
            ],
            filename = 'feed_forward_layers.png'
        )

if __name__ == "__main__":
    main()
    print("\nГотово! Результаты сохранены в example_output/feed_forward_layers.png")
