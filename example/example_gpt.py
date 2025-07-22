"""
Пример использования GPT модели из simple_llm
"""

import torch
import os
from simple_llm.transformer.gpt import GPT

def use_numeric_generation(config, model):
    """Функция для числовой генерации"""
    input_seq = torch.randint(0, config['vocab_size'], (1, 10)).to(config['device'])
    print(f"\nЧисловой ввод: {input_seq.tolist()[0]}")
    
    print("\n=== Режимы генерации ===")
    
    # 1. Жадная генерация
    greedy_output = model.generate(input_seq.clone(), 
                                max_new_tokens=20,
                                do_sample=False)
    print("\n1. Жадная генерация (детерминированная):")
    print(greedy_output.tolist()[0])
    
    # 2. Сэмплирование с температурой
    torch.manual_seed(42)
    temp_output = model.generate(input_seq.clone(),
                               max_new_tokens=20,
                               do_sample=True,
                               temperature=0.7)
    print("\n2. Сэмплирование (температура=0.7):")
    print(temp_output.tolist()[0])
    
    # 3. Top-k сэмплирование
    torch.manual_seed(42)
    topk_output = model.generate(input_seq.clone(),
                               max_new_tokens=20,
                               do_sample=True,
                               top_k=50)
    print("\n3. Top-k сэмплирование (k=50):")
    print(topk_output.tolist()[0])
    
    # 4. Nucleus (top-p) сэмплирование
    try:
        torch.manual_seed(42)
        topp_output = model.generate(input_seq.clone(),
                                   max_new_tokens=20,
                                   do_sample=True,
                                   top_p=0.9)
        print("\n4. Nucleus сэмплирование (p=0.9):")
        print(topp_output.tolist()[0])
    except Exception as e:
        print(f"\nОшибка при nucleus сэмплировании: {str(e)}")
        print("Пропускаем этот режим генерации")

def main():
    # Конфигурация модели
    config = {
        'vocab_size': 10000,
        'max_seq_len': 256,
        'emb_size': 512,
        'num_heads': 8,
        'head_size': 64,
        'num_layers': 6,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 1. Инициализация модели
    print("Инициализация GPT модели...")
    model = GPT(**config)
    print(f"Модель создана на устройстве: {config['device']}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Пример генерации
    print("\nИспользуется числовая генерация...")
    use_numeric_generation(config, model)

    # 3. Сохранение и загрузка модели
    print("\nТест сохранения/загрузки...")
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        model.save(tmp.name)
        print(f"Модель сохранена во временный файл: {tmp.name}")
        
        loaded_model = GPT.load(tmp.name, device=config['device'])
        print("Модель успешно загружена")
        
        test_output = loaded_model(torch.randint(0, config['vocab_size'], (1, 5)).to(config['device']))
        print(f"Тест загруженной модели - выходная форма: {test_output.shape}")

if __name__ == "__main__":
    main()