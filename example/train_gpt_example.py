"""
Пример обучения модели GPT на синтетических данных
"""

import torch
from torch.utils.data import Dataset, DataLoader
from simple_llm.transformer.gpt import GPT

class SyntheticDataset(Dataset):
    """Синтетический датасет для демонстрации обучения GPT"""
    def __init__(self, vocab_size=100, seq_len=20, num_samples=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.roll(self.data, -1, dims=1)  # Сдвигаем на 1 для предсказания следующего токена
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_gpt():
    # Параметры модели
    VOCAB_SIZE = 100
    SEQ_LEN = 20
    EMB_SIZE = 128
    N_HEADS = 4
    HEAD_SIZE = 32
    N_LAYERS = 3
    
    # Инициализация модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQ_LEN,
        emb_size=EMB_SIZE,
        num_heads=N_HEADS,
        head_size=HEAD_SIZE,
        num_layers=N_LAYERS,
        device=device
    )
    
    # Создание датасета и загрузчика
    dataset = SyntheticDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Начало обучения на {device}...")
    print(f"Размер словаря: {VOCAB_SIZE}")
    print(f"Длина последовательности: {SEQ_LEN}")
    print(f"Размер батча: 32")
    print(f"Количество эпох: 5\n")
    
    # Обучение модели
    model.fit(
        train_loader=train_loader,
        valid_loader=None,  # Можно добавить валидационный набор
        num_epoch=5,
        learning_rate=0.001
    )
    
    # Сохранение модели
    model_path = "trained_gpt_model.pt"
    model.save(model_path)
    print(f"\nМодель сохранена в {model_path}")
    
    # Пример генерации текста
    print("\nПример генерации:")
    input_seq = torch.randint(0, VOCAB_SIZE, (1, 10)).to(device)
    generated = model.generate(
        x=input_seq,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7
    )
    print(f"Вход: {input_seq.tolist()[0]}")
    print(f"Сгенерировано: {generated.tolist()[0]}")

if __name__ == "__main__":
    train_gpt()
