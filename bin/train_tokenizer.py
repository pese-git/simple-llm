#!/usr/bin/env python3
"""
Обучение токенизатора с CLI аргументами
"""
import os
import argparse
from pathlib import Path
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True,
                      help='Путь к корпусу текстов')
    parser.add_argument('--output', type=str, required=True,
                      help='Путь для сохранения токенизатора')
    parser.add_argument('--vocab-size', type=int, default=4000,
                      help='Размер словаря')
    args = parser.parse_args()

    # Проверяем и создаем директорию для сохранения
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Создаем директорию: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка корпуса
    corpus = []
    for file in Path(args.corpus).glob('*.txt'):
        corpus.append(file.read_text(encoding='utf-8'))
    corpus = '\n'.join(corpus)

    # Обучение
    tokenizer = OptimizeBPE(vocab_size=args.vocab_size)
    tokenizer.fit(corpus)
    tokenizer.save(args.output)

if __name__ == '__main__':
    main()
