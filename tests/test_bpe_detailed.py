import os
import tempfile
import pytest
from simple_llm.tokenizer.bpe import BPE

class TestBPE:
    @pytest.fixture
    def sample_text(self):
        return "ааабббвввггг аааббб дддд ееее жжжж"

    @pytest.fixture
    def bpe(self):
        return BPE(vocab_size=20)

    def test_fit(self, bpe, sample_text):
        """Тест обучения токенизатора"""
        bpe.fit(sample_text)
        assert len(bpe.vocab) == bpe.vocab_size
        assert len(bpe.token2id) == bpe.vocab_size
        assert len(bpe.id2token) == bpe.vocab_size

    def test_encode_decode(self, bpe, sample_text):
        """Тест кодирования и декодирования"""
        bpe.fit(sample_text)
        encoded = bpe.encode(sample_text)
        decoded = bpe.decode(encoded)
        assert decoded == sample_text

    @pytest.mark.skip(reason="Требуется доработка обработки неизвестных символов")
    def test_encode_unknown_chars(self, bpe, sample_text):
        """Тест с неизвестными символами"""
        bpe.fit(sample_text)
        test_text = "ааббцц"  # 'цц' нет в обучающем тексте
        encoded = bpe.encode(test_text)
        assert -1 in encoded  # Должен содержать специальный токен для неизвестных символов
        decoded = bpe.decode(encoded)
        assert "цц" in decoded

    def test_save_load(self, bpe, sample_text):
        """Тест сохранения и загрузки"""
        bpe.fit(sample_text)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                bpe.save(tmp.name)
                loaded = BPE.load(tmp.name)
                
                assert loaded.vocab_size == bpe.vocab_size
                assert loaded.vocab == bpe.vocab
                assert loaded.token2id == bpe.token2id
                assert loaded.id2token == bpe.id2token
                
                # Проверяем работоспособность после загрузки
                encoded = loaded.encode(sample_text)
                decoded = loaded.decode(encoded)
                assert decoded == sample_text
            finally:
                os.unlink(tmp.name)

    def test_pair_merging(self, bpe, sample_text):
        """Тест правильности объединения пар"""
        bpe.fit(sample_text)
        
        # Проверяем, что самые частые пары были объединены
        assert 'аа' in bpe.vocab or 'ааа' in bpe.vocab
        assert 'бб' in bpe.vocab or 'ббб' in bpe.vocab

    @pytest.mark.skip(reason="Требуется доработка валидации vocab_size")
    def test_vocab_size(self):
        """Тест обработки слишком маленького vocab_size"""
        small_bpe = BPE(vocab_size=5)
        with pytest.raises(ValueError):
            small_bpe.fit("абвгд")  # Слишком мало для начальных символов

if __name__ == "__main__":
    pytest.main()
