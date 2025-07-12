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
        # 1. Разбиваем текст на токены-символы
        sequence = list(text)
        # 2. Инициализация пустого списка токенов
        tokens = []
        # 3. Установить i = 0
        i = 0
        while i < len(text):
            # 3.1 Найти все токены в словаре, начинающиеся с text[i]
            start_char = text[i]
            result = [token for token in self.vocab if token.startswith(start_char)]
            # 3.2 Выбрать самый длинный подходящий токен
            find_token = self._find_max_matching_token(text[i:], result)
            if find_token is None:
                # Обработка неизвестного символа
                tokens.append(text[i])  # Добавляем сам символ как токен
                i += 1
            else:
                # 3.3 Добавить токен в результат
                tokens.append(find_token)
                # 3.4 Увеличить i на длину токена
                i += len(find_token)

        # 4. Заменить токены на их ID
        return self._tokens_to_ids(tokens)

    def _find_max_matching_token(self, text: str, tokens: list):
        """Находит самый длинный токен из списка, с которого начинается текст"""
        matching = [token for token in tokens if text.startswith(token)]
        return max(matching, key=len) if matching else None

    def _tokens_to_ids(self, tokens):
        """Конвертирует список токенов в их ID с обработкой неизвестных токенов"""
        ids = []
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.append(-1)  # Специальное значение
        return ids

    def decode(self, ids: list[int]):
        return ''.join(self._ids_to_tokens(ids))

    def _ids_to_tokens(self, ids: list) -> list:
        """Конвертирует список Ids в их tokens"""
        tokens = []
        for id in ids:
            if id in self.id2token:
                tokens.append(self.id2token[id])
            else:
                tokens.append('')  # Специальное значение
        return tokens