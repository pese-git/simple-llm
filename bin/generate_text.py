#!/usr/bin/env python3
"""
Генерация текста (финальная версия)
"""
import argparse
import torch
from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.transformer.gpt import GPT

def main():
    parser = argparse.ArgumentParser()
    # Обязательные параметры
    parser.add_argument('--model', type=str, required=True,
                      help='Путь к файлу модели (.pth)')
    parser.add_argument('--tokenizer', type=str, required=True,
                      help='Путь к файлу токенизатора (.json)')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Начальный текст для генерации')
    
    # Параметры модели (должны соответствовать обучению)
    parser.add_argument('--seq-len', type=int, default=64,
                      help='Макс. длина последовательности (как при обучении)')
    parser.add_argument('--emb-size', type=int, default=64,
                      help='Размер эмбеддингов (как при обучении)')
    parser.add_argument('--num-heads', type=int, default=4,
                      help='Количество голов внимания (как при обучении)')
    parser.add_argument('--head-size', type=int, default=16,
                      help='Размер головы внимания (как при обучении)')
    parser.add_argument('--num-layers', type=int, default=2,
                      help='Количество слоёв (как при обучении)')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout (как при обучении)')
    
    # Параметры генерации
    parser.add_argument('--length', type=int, default=50,
                      help='Количество генерируемых токенов')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Температура сэмплинга (0.1-1.0)')
    
    args = parser.parse_args()

    # Загрузка
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    tokenizer = SimpleBPE.load(args.tokenizer)
    print(f"Загружен токенизатор (vocab_size={tokenizer.vocab_size})")
    
    # Инициализация модели
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        emb_size=args.emb_size,
        num_heads=args.num_heads,
        head_size=args.head_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device
    )
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Загружена модель с {sum(p.numel() for p in model.parameters()):,} параметрами")

    # Генерация
    print(f"\nГенерация текста для промта: '{args.prompt}'")
    tokens = tokenizer.encode(args.prompt)
    print(f"Токены промта: {tokens}")
    
    output = model.generate(
        x=torch.tensor([tokens], device=device),
        max_new_tokens=args.length,
        do_sample=True,
        temperature=args.temperature
    )
    
    print("\n=== Результат ===")
    print(tokenizer.decode(output[0].tolist()))

if __name__ == '__main__':
    main()
