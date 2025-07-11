import pytest
from simple_llm.tokenizer.simple_bpe import SimpleBPE
from simple_llm.tokenizer.optimize_bpe import OptimizeBPE

@pytest.fixture(scope="session")
def large_text():
    """Генерирует большой текст для тестирования"""
    return " ".join(["мама мыла раму"] * 1000)

@pytest.fixture(params=[SimpleBPE, OptimizeBPE])
def bpe_class(request):
    """Возвращает классы BPE для тестирования"""
    return request.param
