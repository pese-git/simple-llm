"""
Обучение GPT-модели на токенизированном корпусе
"""
import pickle
from torch.utils.data import DataLoader
from simple_llm.data.get_data import GetData
from simple_llm.transformer.gpt import GPT

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

    with open('data/tokens/corpus_tokens.pkl', 'rb') as f:
        tokenized = pickle.load(f)
    all_tokens = [token for line in tokenized for token in line]
    seq_len = 64
    dataset = GetData(data=all_tokens, seq_len=seq_len, device=device)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Загрузите токенизатор для определения размера словаря
    from simple_llm.tokenizer.bpe import BPE
    tokenizer = BPE.load('data/tokenizer/bpe_tokenizer.json')

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        emb_size=256,
        num_heads=4,
        head_size=64,
        num_layers=4,
        device='cpu'
    )

    model.fit(
        train_loader=loader,
        valid_loader=None,
        num_epoch=10,
        learning_rate=1e-4
    )
    print('Train loss:', model.train_loss)
    torch.save(model.state_dict(), 'data/model/simple_llm_gpt.pth')
    print("Модель обучена и сохранена в data/model/simple_llm_gpt.pth")
