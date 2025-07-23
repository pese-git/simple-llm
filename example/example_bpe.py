from simple_llm.tokenizer.bpe import BPE
from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE
import time

def compare_tokenizers(text, vocab_size=50):
    """Сравнивает разные реализации BPE"""
    print(f"\n=== Анализ текста: '{text[:20]}...' ===")
    
    # 1. Базовая реализация BPE
    start = time.time()
    bpe = BPE(vocab_size=vocab_size)
    bpe.fit(text)
    base_time = time.time() - start
    
    print("\n[Базовая реализация BPE]")
    print(f"Время обучения: {base_time:.4f} сек")
    print(f"Размер словаря: {len(bpe.vocab)}")
    print("Примеры токенов:", list(bpe.vocab)[:10], "...")
    
    # 2. SimpleBPE
    start = time.time()
    simple_bpe = SimpleBPE(vocab_size=vocab_size)
    simple_bpe.fit(text)
    simple_time = time.time() - start
    
    print("\n[SimpleBPE]")
    print(f"Время обучения: {simple_time:.4f} сек")
    print(f"Размер словаря: {len(simple_bpe.vocab)}")
    
    # 3. OptimizeBPE
    start = time.time()
    opt_bpe = OptimizeBPE(vocab_size=vocab_size)
    opt_bpe.fit(text)
    opt_time = time.time() - start
    
    print("\n[OptimizeBPE]")
    print(f"Время обучения: {opt_time:.4f} сек")
    print(f"Размер словаря: {len(opt_bpe.vocab)}")
    
    # Сравнение производительности
    if opt_time > 0:
        print(f"\nОптимизированная версия быстрее SimpleBPE в {simple_time/opt_time:.1f} раз")
    
    # Демонстрация работы на примерах
    test_phrases = [
        text.split()[0],  # первое слово
        text[:10],       # часть текста
        "неизвестное_слово",  # OOV
        "спецсимволы: 123, !@#"
    ]
    
    print("\n=== Примеры кодирования/декодирования ===")
    for phrase in test_phrases:
        print(f"\nФраза: '{phrase}'")
        
        encoded = bpe.encode(phrase)
        decoded = bpe.decode(encoded)
        print(f"BPE: {encoded} -> '{decoded}'")
        
        encoded = simple_bpe.encode(phrase)
        decoded = simple_bpe.decode(encoded)
        print(f"SimpleBPE: {encoded} -> '{decoded}'")
        
        encoded = opt_bpe.encode(phrase)
        decoded = opt_bpe.decode(encoded)
        print(f"OptimizeBPE: {encoded} -> '{decoded}'")

def main():
    # Тестовые тексты разной сложности
    texts = [
        "мама мыла раму, папа пил какао",
        "коты бегают быстро, собаки лают громко",
        "искусственный интеллект меняет мир вокруг нас",
        "BPE (Byte Pair Encoding) - популярный алгоритм токенизации"
    ]
    
    for text in texts:
        compare_tokenizers(text)
        
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    main()
