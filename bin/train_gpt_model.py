#!/usr/bin/env python3
"""
Обучение GPT с CLI аргументами (исправленная версия)
"""
import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from simple_llm.data.get_data import GetData
from simple_llm.transformer.gpt import GPT
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=str, required=True,
                      help='Путь к токенизированным данным (.pkl)')
    parser.add_argument('--tokenizer', type=str, required=True,
                      help='Путь к файлу токенизатора (.json)')
    parser.add_argument('--output', type=str, required=True,
                      help='Путь для сохранения модели (.pth)')
    
    # Параметры модели
    parser.add_argument('--seq-len', type=int, default=64,
                      help='Максимальная длина последовательности')
    parser.add_argument('--emb-size', type=int, default=64,
                      help='Размер эмбеддингов')
    parser.add_argument('--num-heads', type=int, default=4,
                      help='Количество голов внимания')
    parser.add_argument('--head-size', type=int, default=16,
                      help='Размер головы внимания')
    parser.add_argument('--num-layers', type=int, default=2,
                      help='Количество слоёв декодера')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Вероятность dropout')
    
    # Параметры обучения
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Размер батча')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    
    args = parser.parse_args()

    # Проверяем и создаем директорию для сохранения
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Создаем директорию: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Загрузка данных
    with open(args.tokens, 'rb') as f:
        tokens = pickle.load(f)
    tokenizer = OptimizeBPE.load(args.tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Подготовка данных
    dataset = GetData(data=tokens, seq_len=args.seq_len, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Модель (уменьшенные параметры)
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

    # Обучение
    model.fit(
        train_loader=loader,
        num_epoch=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=output_dir
    )
    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    main()
