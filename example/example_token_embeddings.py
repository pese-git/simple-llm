"""
Пример использования TokenEmbeddings для работы с векторными представлениями токенов

Содержит:
1. Базовый пример создания и использования эмбеддингов
2. Пример обучения эмбеддингов
3. Визуализацию похожих токенов
"""

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from simple_llm.embedding.token_embedings import TokenEmbeddings

def basic_example():
    """Базовый пример использования TokenEmbeddings"""
    print("\n=== Базовый пример ===")
    
    # Создаем слой эмбеддингов для словаря из 10 токенов с размерностью 3
    embedding_layer = TokenEmbeddings(vocab_size=10, emb_size=3)
    
    # Создаем тензор с индексами токенов (2 примера по 3 токена)
    tokens = torch.tensor([
        [1, 2, 3],  # Первая последовательность
        [4, 5, 6]   # Вторая последовательность
    ])
    
    # Получаем векторные представления
    embeddings = embedding_layer(tokens)
    
    print("Исходные индексы токенов:")
    print(tokens)
    print("\nВекторные представления (формат: [batch, sequence, embedding]):")
    print(embeddings)
    print(f"\nФорма выходного тензора: {embeddings.shape}")

def training_example():
    """Пример обучения эмбеддингов"""
    print("\n=== Пример обучения ===")
    
    # Инициализация
    embed = TokenEmbeddings(vocab_size=5, emb_size=2)
    optimizer = torch.optim.SGD(embed.parameters(), lr=0.1)
    
    # Токены для обучения (предположим, что 0 и 1 должны быть похожи)
    similar_tokens = torch.tensor([0, 1])
    
    print("Векторы ДО обучения:")
    print(embed(torch.arange(5)))  # Все векторы
    
    # Простейший "тренировочный" цикл
    for _ in range(50):
        optimizer.zero_grad()
        embeddings = embed(similar_tokens)
        loss = torch.dist(embeddings[0], embeddings[1])  # Минимизируем расстояние
        loss.backward()
        optimizer.step()
    
    print("\nВекторы ПОСЛЕ обучения:")
    print(embed(torch.arange(5)))
    print(f"\nРасстояние между токенами 0 и 1: {torch.dist(embed(torch.tensor([0])), embed(torch.tensor([1]))):.4f}")

def visualization_example():
    """Визуализация эмбеддингов в 2D пространстве"""
    print("\n=== Визуализация ===")
    
    # Создаем эмбеддинги для 100 токенов
    embed = TokenEmbeddings(vocab_size=100, emb_size=16)
    
    # Получаем все векторы
    all_embeddings = embed(torch.arange(100)).detach().numpy()
    
    # Уменьшаем размерность до 2D для визуализации
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    plt.title("Визуализация эмбеддингов токенов (PCA)")
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    
    # Подпишем некоторые точки
    for i in [0, 1, 2, 50, 51, 52, 98, 99]:
        plt.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.grid()
    plt.show()

if __name__ == "__main__":
    basic_example()
    training_example()
    visualization_example()
