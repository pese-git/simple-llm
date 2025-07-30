# Simple-LLM: Персональная языковая модель

## 🎯 Цель проекта

Simple-LLM - это минималистичная реализация языковой модели (LLM) с полным циклом:
- Обучение BPE-токенизатора на ваших данных
- Подготовка датасета для обучения модели
- Тренировка компактной GPT-архитектуры
- Генерация текста в заданном стиле

Проект создан для:
1. Образовательных целей - понимания работы современных LLM
2. Экспериментов с генерацией текста на небольших датасетах
3. Создания персонализированных языковых моделей

Полный цикл от обучения токенизатора до генерации текста

## 🛠 Установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/ваш-репозиторий/simple-llm.git
cd simple-llm

# 2. Создайте виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# 3. Установите зависимости
pip install torch
pip install dill tqdm  # Основные зависимости для работы

# Установка simple_llm пакета
pip install .
```

## 📂 Подготовка данных

Поместите текстовые файлы (.txt) в папку:
```
data/
└── corpus/
    └── sample/
        ├── text1.txt
        ├── text2.txt
        └── ...
```

## 🔄 Полный рабочий цикл

### 1. Обучение BPE-токенизатора
```bash
python bin/train_tokenizer.py \
  --corpus data/corpus/sample \
  --output data/tokenizer/bpe_model.json \
  --vocab-size 500
```

### 2. Токенизация данных
```bash
python bin/tokenize_corpus.py \
  --corpus data/corpus/sample \
  --tokenizer data/tokenizer/bpe_model.json \
  --output data/tokens/tokenized_corpus.pkl
```

### 3. Обучение GPT модели
```bash
python bin/train_gpt_model.py \
  --tokens data/tokens/tokenized_corpus.pkl \
  --tokenizer data/tokenizer/bpe_model.json \
  --output data/model/gpt_model.pth \
  --seq-len 32 \
  --batch-size 3 \
  --epochs 3 \
  --emb-size 64 \
  --num-heads 2 \
  --num-layers 2
```

### ✔️ Восстановление обучения с чекпоинта

Если обучение было прервано или вы хотите дообучить модель:
- Просто перезапустите команду обучения с теми же параметрами (`bin/train_gpt_model.py ...`).
- Скрипт сам определит последний checkpoint (checkpoint_epoch_X.pt) в папке модели.
- Обучение продолжится с нужной эпохи, параметры останутся неизменными.
- В консоли вы увидите сообщения вроде:
  ```
  ⚡ Восстанавливаем обучение с эпохи 12
  Восстановление обучения GPT
  ...
  Начало обучения GPT на 18 эпох (с 12 по 29)
  ```

### 4. Генерация текста
```bash
python bin/generate_text.py \
  --model data/model/gpt_model.pth \
  --tokenizer data/tokenizer/bpe_model.json \
  --seq-len 32 \
  --emb-size 64 \
  --num-heads 2 \
  --num-layers 2 \
  --prompt "Ваш текст для продолжения" \
  --length 100 \
  --temperature 0.7
```

## 🚀 Быстрый старт (минимальная конфигурация)
```bash
# Последовательно выполните:
./bin/train_tokenizer.py --corpus data/corpus/sample --output data/tokenizer/bpe.json
./bin/tokenize_corpus.py --corpus data/corpus/sample --tokenizer data/tokenizer/bpe.json
./bin/train_gpt_model.py --tokens data/tokens/corpus_tokens.pkl --tokenizer data/tokenizer/bpe.json
./bin/generate_text.py --model data/model/gpt_model.pth --tokenizer data/tokenizer/bpe.json --prompt "Привет"
```

## 🧠 Рекомендации по параметрам

| Параметр         | CPU (рекомендации) | GPU (рекомендации) |
|------------------|--------------------|--------------------|
| vocab-size       | 2000-5000          | 5000-10000         |
| seq-len          | 64-128             | 128-256            |
| batch-size       | 4-8                | 16-32              |
| emb-size         | 64-128             | 256-512            |
| num-layers       | 2-4                | 6-12               |

## ⚠️ Устранение проблем
1. **Ошибка памяти**:
   - Уменьшите `batch-size` и `seq-len`
   ```bash
   python bin/train_gpt_model.py --batch-size 2 --seq-len 64
   ```

2. **Плохая генерация**:
   - Увеличьте размер корпуса (>1MB текста)
   - Добавьте больше эпох обучения (`--epochs 15`)

3. **Медленная работа**:
   ```bash
   # Для GPU добавьте перед запуском:
   export CUDA_VISIBLE_DEVICES=0
   ```

## 👥 Участие в разработке

Мы приветствуем вклад в проект! Вот как вы можете помочь:

### 🛠 Как внести свой вклад:
1. Форкните репозиторий
2. Создайте ветку для вашего изменения (`git checkout -b feature/your-feature`)
3. Сделайте коммит ваших изменений (`git commit -am 'Add some feature'`)
4. Запушьте в ветку (`git push origin feature/your-feature`)
5. Создайте Pull Request

### 📌 Правила:
- Следуйте существующему стилю кода
- Пишите понятные сообщения коммитов
- Добавляйте тесты для новых функций
- Обновляйте документацию при изменении API

### 🐛 Сообщение об ошибках:
Открывайте Issue с описанием:
1. Шаги для воспроизведения
2. Ожидаемое поведение
3. Фактическое поведение
4. Версии ПО (Python, PyTorch и т.д.)

## 📜 Лицензия

Проект распространяется под лицензией MIT. Полный текст лицензии доступен в файле [LICENSE](LICENSE).

Основные положения:
- Разрешается свободное использование, модификация и распространение кода
- Обязательно указание авторства
- Лицензия предоставляется "как есть" без гарантий
- Авторы не несут ответственности за последствия использования

## 📌 Важно
- Все скрипты имеют встроенную помощь:
```bash
python bin/train_tokenizer.py --help
```
- Модель автоматически использует GPU если доступен
- Для выхода из виртуального окружения: `deactivate`
