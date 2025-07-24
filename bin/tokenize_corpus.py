#!/usr/bin/env python3
"""
Токенизация корпуса с CLI аргументами
"""
import os
import argparse
import pickle
from pathlib import Path
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True,
                      help='Путь к директории с текстами')
    parser.add_argument('--tokenizer', type=str, required=True,
                      help='Путь к файлу токенизатора')
    parser.add_argument('--output', type=str, required=True,
                      help='Путь для сохранения токенизированных данных')
    parser.add_argument('--max-tokens', type=int, default=None,
                      help='Максимальное количество токенов (для тестов)')
    args = parser.parse_args()

    # Загрузка
    tokenizer = OptimizeBPE.load(args.tokenizer)
    corpus = []
    
    print(f"Чтение текстов из {args.corpus}...")
    for file in Path(args.corpus).glob('*.txt'):
        corpus.append(file.read_text(encoding='utf-8'))
    
    # Токенизация
    print("Токенизация...")
    all_tokens = []
    for text in corpus:
        tokens = tokenizer.encode(text)
        if args.max_tokens:
            tokens = tokens[:args.max_tokens]
        all_tokens.extend(tokens)
    
    # Сохранение
    # Проверяем и создаем директорию для сохранения
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Создаем директорию: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump(all_tokens, f)
    print(f"Сохранено {len(all_tokens)} токенов в {args.output}")

if __name__ == '__main__':
    main()
