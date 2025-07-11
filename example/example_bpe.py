from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE
import time

def tokenize_manually(text, vocab):
    """Простая ручная токенизация по словарю"""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        found = False
        # Ищем самый длинный возможный токен из словаря
        for l in range(min(4, n-i), 0, -1):  # проверяем токены длиной до 4 символов
            if text[i:i+l] in vocab:
                tokens.append(text[i:i+l])
                i += l
                found = True
                break
        if not found:  # если токен не найден, берем один символ
            tokens.append(text[i])
            i += 1
    return tokens

def run_example(text, vocab_size=30):
    print("\n=== Тестирование токенизаторов ===")
    print(f"Исходный текст: '{text}'\n")
    
    # Simple BPE
    start = time.time()
    simple_bpe = SimpleBPE(vocab_size=vocab_size)
    simple_bpe.fit(text)
    simple_time = time.time() - start
    
    print("SimpleBPE:")
    print(f"Время обучения: {simple_time:.4f} сек")
    print(f"Размер словаря: {len(simple_bpe.vocab)}")
    print(f"Словарь: {simple_bpe.vocab}")
    print(f"Ручная токенизация: {tokenize_manually(text, simple_bpe.vocab)}\n")
    
    # Optimize BPE
    start = time.time()
    opt_bpe = OptimizeBPE(vocab_size=vocab_size)
    opt_bpe.fit(text)
    opt_time = time.time() - start
    
    print("OptimizeBPE:")
    print(f"Время обучения: {opt_time:.4f} сек")
    print(f"Размер словаря: {len(opt_bpe.vocab)}")
    print(f"Словарь: {opt_bpe.vocab}")
    print(f"Ручная токенизация: {tokenize_manually(text, opt_bpe.vocab)}\n")
    
    if opt_time > 0:
        print(f"Оптимизированная версия быстрее в {simple_time/opt_time:.1f} раз\n")

if __name__ == "__main__":
    text1 = "мама мыла раму, папа пил какао"
    text2 = "коты бегают быстро, собаки лают громко"
    
    run_example(text1)
    run_example(text2)
