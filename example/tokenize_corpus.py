"""
Токенизация текстового корпуса с помощью обученного BPE-токенизатора
"""
from simple_llm.tokenizer.bpe import BPE
import pickle

if __name__ == "__main__":
    import torch
    # Определяем устройство
    #if torch.cuda.is_available():
    #    device = 'cuda'
    #elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    #    device = 'mps'  # Apple Silicon
    #else:
    #    device = 'cpu'
    device = 'cpu'
    print(f"Используется устройство: {device}")

    tokenizer = BPE.load('data/tokenizer/bpe_tokenizer.json')
    with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokenized = [tokenizer.encode(line) for line in lines]
    with open('data/tokens/corpus_tokens.pkl', 'wb') as f:
        pickle.dump(tokenized, f)
    print("Корпус токенизирован и сохранён в data/corpus_tokens.pkl")
