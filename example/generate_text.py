"""
Генерация текста с помощью обученной GPT-модели и токенизатора
"""
import torch
from simple_llm.transformer.gpt import GPT
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

    # Загрузим токенизатор и модель
    tokenizer = BPE.load('data/tokenizer/bpe_tokenizer.json')
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        emb_size=256,
        num_heads=4,
        head_size=64,
        num_layers=4,
        device=device
    )
    model.load_state_dict(torch.load('data/model/simple_llm_gpt.pth', map_location=device))
    model.eval()

    # Введите начальный текст
    prompt = "Привет, мир! "
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Токены prompt: {prompt_tokens}")
    print(f"Размер словаря токенизатора: {tokenizer.vocab_size}")
    if any(idx >= tokenizer.vocab_size or idx < 0 for idx in prompt_tokens):
        print("ВНИМАНИЕ: В prompt есть токены с индексом вне диапазона словаря! Генерация невозможна.")
        exit(1)
    input_ids = torch.tensor([prompt_tokens], device=device)
    output = model.generate(
        x=input_ids,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0
    )
    result = tokenizer.decode(output[0].tolist())
    print("Сгенерированный текст:", result)
