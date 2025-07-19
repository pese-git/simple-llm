"""
Пример использования механизма внимания HeadAttention
с визуализацией матрицы внимания и анализом работы.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from simple_llm.transformer.head_attention import HeadAttention

import os
os.makedirs("example_output", exist_ok=True)

def plot_attention(weights, tokens=None, filename="attention_plot.png"):
    """Сохранение матрицы внимания в файл"""
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='viridis')
    
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
    
    plt.colorbar()
    plt.title("Матрица весов внимания")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.savefig(f"example_output/{filename}")
    plt.close()

def simulate_text_attention():
    """Пример с имитацией текстовых данных"""
    # Параметры
    emb_size = 64
    head_size = 32
    seq_len = 8
    
    # Имитация токенов
    tokens = ["[CLS]", "мама", "мыла", "раму", ",", "папа", "пил", "какао"]
    
    # Инициализация
    torch.manual_seed(42)
    attention = HeadAttention(emb_size, head_size, max_seq_len=seq_len)
    
    # Случайные эмбеддинги (в реальности - выход слоя токенизации)
    x = torch.randn(1, seq_len, emb_size)
    
    # Прямой проход + получение весов
    with torch.no_grad():
        output = attention(x)
        q, k = attention._q(x), attention._k(x)
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(head_size)
        weights = torch.softmax(scores, dim=-1).squeeze()
    
    # Визуализация
    print("\nПример для фразы:", " ".join(tokens))
    print("Форма выходного тензора:", output.shape)
    plot_attention(weights.numpy(), tokens)

def technical_demo():
    """Техническая демонстрация работы механизма"""
    print("\nТехническая демонстрация HeadAttention")
    attention = HeadAttention(emb_size=16, head_size=8, max_seq_len=10)
    
    # Создаем тензор с ручными значениями для анализа
    x = torch.zeros(1, 4, 16)
    x[0, 0, :] = 1.0  # Яркий токен
    x[0, 3, :] = 0.5  # Слабый токен
    
    # Анализ весов
    with torch.no_grad():
        output = attention(x)
        q = attention._q(x)
        k = attention._k(x)
        print("\nQuery векторы (первые 5 значений):")
        print(q[0, :, :5])
        
        print("\nKey векторы (первые 5 значений):")
        print(k[0, :, :5])
        
        weights = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(8), dim=-1)
        print("\nМатрица внимания:")
        print(weights.squeeze().round(decimals=3))

if __name__ == "__main__":
    print("Демонстрация работы HeadAttention")
    simulate_text_attention()
    technical_demo()
    print("\nГотово! Проверьте графики матрицы внимания.")
