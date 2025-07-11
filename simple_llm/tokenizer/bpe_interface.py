from abc import ABC, abstractmethod
from typing import List, Dict

class BPE(ABC):
    """
    Реализация алгоритма токенизации Byte Pair Encoding (BPE).

    BPE — это итеративный алгоритм, последовательно объединяющий наиболее частые пары символов/токенов,
    чтобы построить эффективный словарь для работы с текстом: токенизации, обучения языковой модели и т.п.

    Аргументы конструктора:
        vocab_size (int): Желаемый размер итогового словаря токенов (включая отдельные символы и составные токены).

    Атрибуты:
        vocab (List[str]): Список токенов в порядке их получения (сначала символы, затем новые пары).
        token2id (Dict[str, int]): Словарь преобразования токена в его индекс.
        id2token (Dict[int, str]): Обратный словарь преобразования индекса в токен.
    """
    def __init__(self, vocab_size: int):
        """
        Инициализация BPE токенизатора.

        Args:
            vocab_size (int): Размер словаря, к которому будет расширяться BPE.
        """
        self.vocab_size = vocab_size
        self.vocab: List[str] = []
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

    @abstractmethod
    def fit(self, text: str):
        pass

    def encode(self, text: str):
        raise NotImplementedError("Implement in subclass if needed.")

    def decode(self, ids: list[int]):
        raise NotImplementedError("Implement in subclass if needed.")
