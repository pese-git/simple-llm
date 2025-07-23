"""
Обучение BPE-токенизатора на текстовом корпусе
"""
from simple_llm.tokenizer.bpe import BPE

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

    with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    tokenizer = BPE(vocab_size=5000)
    tokenizer.fit(" ".join(texts))
    tokenizer.save('data/tokenizer/bpe_tokenizer.json')
    print("Токенизатор обучен и сохранён в data/tokenizer/bpe_tokenizer.json")
