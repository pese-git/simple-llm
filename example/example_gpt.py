"""
Пример использования GPT модели из simple_llm

1. Инициализация модели
2. Генерация текста
3. Сохранение/загрузка модели
"""

import torch
from simple_llm.transformer.gpt import GPT

def main():
    # Конфигурация модели
    config = {
        'vocab_size': 10000,  # Размер словаря
        'max_seq_len': 256,   # Макс. длина последовательности
        'emb_size': 512,      # Размерность эмбеддингов
        'num_heads': 8,       # Количество голов внимания
        'head_size': 64,      # Размер каждой головы внимания
        'num_layers': 6,      # Количество слоев декодера
        'dropout': 0.1,       # Dropout
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 1. Инициализация модели
    print("Инициализация GPT модели...")
    model = GPT(**config)
    print(f"Модель создана на устройстве: {config['device']}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Пример генерации с токенизатором
    try:
        from simple_llm.tokenizer.simple_bpe import SimpleBPE
        print("\nИнициализация токенизатора...")
        tokenizer = SimpleBPE()
        
        text = "Пример текста для генерации"
        print(f"Исходный текст: '{text}'")
        
        input_ids = tokenizer.encode(text)
        print(f"Токенизированный ввод: {input_ids}")
        
        input_seq = torch.tensor([input_ids], device=config['device'])
        generated = model.generate(input_seq, max_new_tokens=20)
        
        decoded_text = tokenizer.decode(generated[0].tolist())
        print(f"\nСгенерированный текст: '{decoded_text}'")
    except ImportError:
        print("\nТокенизатор не найден, используется числовая генерация...")
        input_seq = torch.randint(0, config['vocab_size'], (1, 10)).to(config['device'])
        print(f"Числовой ввод: {input_seq.tolist()[0]}")
        
        generated = model.generate(input_seq, max_new_tokens=20)
        print(f"Числовой вывод: {generated.tolist()[0]}")

    # 3. Сохранение и загрузка модели
    print("\nТест сохранения/загрузки...")
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        model.save(tmp.name)
        print(f"Модель сохранена во временный файл: {tmp.name}")
        
        loaded_model = GPT.load(tmp.name, device=config['device'])
        print("Модель успешно загружена")
        
        # Проверка работы загруженной модели
        test_output = loaded_model(input_seq)
        print(f"Тест загруженной модели - выходная форма: {test_output.shape}")

if __name__ == "__main__":
    main()
