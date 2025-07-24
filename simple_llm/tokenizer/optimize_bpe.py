from .bpe_interface import BPE
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict

class OptimizeBPE(BPE):

    def fit(self, text: str) -> None:
        """
        Обучает BPE-модель на предоставленном тексте.

        Последовательно расширяет словарь за счёт объединения наиболее частых пар токенов до достижения vocab_size.

        Args:
            text (str): Исходная строка для обучения токенизатора.
        """
        sequence = list(text)
        self._init_vocab(sequence)
        pair_freq, pair_first_occurrence = self._get_pair_stats(sequence)

        # Инициализация прогресс-бара
        with tqdm(total=self.vocab_size, desc="Building vocabulary") as pbar:
            pbar.update(len(self.vocab))  # Учитываем начальные токены

            while len(self.vocab) < self.vocab_size and pair_freq:
                pair_to_merge = self._select_pair_to_merge(pair_freq, pair_first_occurrence)
                new_token = pair_to_merge[0] + pair_to_merge[1]
                
                # Обновляем прогресс и логируем
                pbar.update(1)
                pbar.set_postfix({
                    'current_vocab': len(self.vocab),
                    'top_pair': f"{pair_to_merge[0]}{pair_to_merge[1]}",
                    'pair_freq': pair_freq[pair_to_merge]
                })
                print(f"\nТекущий размер словаря: {len(self.vocab)}/{self.vocab_size}")
                print(f"Самая частая пара: {pair_to_merge} (встречается {pair_freq[pair_to_merge]} раз)")
                print(f"Добавлен новый токен: '{new_token}'")

                if new_token in self.vocab:
                    # Защита от зацикливания: пара уже была добавлена как новый токен.
                    del pair_freq[pair_to_merge]
                    continue

                self.vocab.append(new_token)
                sequence, pair_freq, pair_first_occurrence = self._merge_pair(
                    sequence, pair_to_merge, new_token, pair_freq
                )

        self._build_token_dicts()

    def _init_vocab(self, sequence: List[str]) -> None:
        """
        Формирует стартовый словарь уникальных символов из последовательности, отсортированный по символам.

        Args:
            sequence (List[str]): Исходная последовательность символов.
        """
        self.vocab = sorted(set(sequence))

    def _get_pair_stats(self, sequence: List[str]) -> Tuple[Counter, Dict[Tuple[str, str], int]]:
        """
        Вычисляет частоты появления и индексы первого появления всех пар соседних токенов в последовательности.

        Args:
            sequence (List[str]): Текущая последовательность токенов.

        Returns:
            Tuple[Counter, Dict[Tuple[str, str], int]]:
                - Counter по всем парам (их частоты),
                - Словарь первых индексов появления каждой пары.
        """
        pair_freq = Counter()
        pair_first_occurrence = {}
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_freq[pair] += 1
            if pair not in pair_first_occurrence:
                pair_first_occurrence[pair] = i
        return pair_freq, pair_first_occurrence

    def _select_pair_to_merge(self, pair_freq: Counter, pair_first_occurrence: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
        """
        Выбирает следующую пару для слияния:
        приоритет — самая частая; если таких несколько — та, которая встречается раньше других (наименьший индекс появления).

        Args:
            pair_freq (Counter): Частоты всех пар.
            pair_first_occurrence (Dict[Tuple[str, str], int]): Индексы первых появлений каждой пары.

        Returns:
            Tuple[str, str]: Пара для слияния (двойка токенов).
        """
        pair_to_merge, _ = max(
            pair_freq.items(),
            key=lambda x: (x[1], -pair_first_occurrence.get(x[0], float('inf')))
        )
        return pair_to_merge

    def _merge_pair(
        self,
        sequence: List[str],
        pair_to_merge: Tuple[str, str],
        new_token: str,
        pair_freq: Counter
    ) -> Tuple[List[str], Counter, Dict[Tuple[str, str], int]]:
        """
        Выполняет слияние заданной пары токенов в новой последовательности, корректирует частоты пар и индексы первых появлений.

        Args:
            sequence (List[str]): Текущая последовательность токенов.
            pair_to_merge (Tuple[str, str]): Пара для слияния.
            new_token (str): Новый токен (результат слияния).
            pair_freq (Counter): Частоты текущих пар.

        Returns:
            Tuple[List[str], Counter, Dict[Tuple[str, str], int]]:
                - Новая последовательность,
                - Обновлённые частоты пар,
                - Обновлённые индексы первых появлений пар.
        """
        new_sequence = []
        i = 0
        pairs_to_decrement = Counter()
        pairs_to_increment = Counter()
        length = len(sequence)
        while i < length:
            if i < length - 1 and (sequence[i], sequence[i + 1]) == pair_to_merge:
                if i > 0:
                    pairs_to_decrement[(sequence[i - 1], sequence[i])] += 1
                    pairs_to_increment[(sequence[i - 1], new_token)] += 1
                if i + 2 < length:
                    pairs_to_decrement[(sequence[i + 1], sequence[i + 2])] += 1
                    pairs_to_increment[(new_token, sequence[i + 2])] += 1
                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        for pair, dec_count in pairs_to_decrement.items():
            pair_freq[pair] -= dec_count
            if pair_freq[pair] <= 0:
                del pair_freq[pair]
        for pair, inc_count in pairs_to_increment.items():
            pair_freq[pair] += inc_count
        # Пересчитываем первый индекс появления пар
        pair_first_occurrence = {}
        for idx in range(len(new_sequence) - 1):
            pair = (new_sequence[idx], new_sequence[idx + 1])
            if pair not in pair_first_occurrence:
                pair_first_occurrence[pair] = idx
        for pair in list(pair_freq.keys()):
            if pair not in pair_first_occurrence:
                del pair_freq[pair]
        return new_sequence, pair_freq, pair_first_occurrence

    def _build_token_dicts(self) -> None:
        """
        Формирует словари вида <токен, id> и <id, токен> по итоговому списку токенов.
        """
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}