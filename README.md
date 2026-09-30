# Text analysis pipeline

Программа обрабатывает текст с помощью языковой модели и возвращает JSON:
`summary`, `category`, `sentiment`, `key_points`, `final_answer`.

Реализованы этапы 1–3: клиент API, обработка текста, три варианта промпта,
JSON-схема и обработка ошибок валидации. Содержание ограничено 250 символами,
итоговый ответ — 400; ключевых мыслей должно быть ровно три. Схема отправляется
модели, затем полученный ответ повторно проверяется в Python.

Категории: `question` — вопрос, `request` — просьба выполнить действие,
`feedback` — отзыв, `other` — прочее. Настроение исходного текста:
`positive`, `neutral`, `negative`. По этим полям можно отбирать результаты.

## Структура

- `main.py` — запуск из командной строки и вывод результата в JSON.
- `llm_client.py` — вызов API модели.
- `prompts.py` — инструкции модели.
- `pipeline.py` — соединяет этапы и проверяет ошибки.
- `schemas.py` — схема результата.
- `compare_prompts.py` — сравнение вариантов на одинаковых входах.
- `sample_inputs/` — пять демонстрационных текстов.
- `tests/` — проверки обработки ответа модели без сетевых запросов.
- `docs/prompts.md` — варианты промптов и статус их выбора.

## Запуск

Нужны Python 3.10+ и ключ API. Создайте окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Впишите ключ в `.env` вместо `your_api_key_here`. Файл `.env` исключён из Git.
По умолчанию используется `gpt-4.1-mini`; при необходимости измените
`OPENAI_MODEL`. Для сервиса с поддержкой Responses API и JSON-схемы
задайте `OPENAI_BASE_URL`.

```bash
python main.py --demo
python main.py --demo --output output/demo.json
python main.py --text "Помоги составить план встречи"
python main.py --file /path/to/input.txt
python main.py --demo --prompt-variant baseline
python main.py --demo --category feedback --sentiment negative
```

Программа печатает массив JSON. При ошибке API или невалидном ответе модели
элемент получает поле `error`, а программа завершает работу с кодом 1 после
обработки остальных текстов. Сообщение указывает ошибку JSON или конкретное
неверное поле. Ошибки остаются видимыми и при фильтрации результатов.

Доступны варианты `baseline`, `explicit` (по умолчанию), `example`.
При необходимости автоматическое сравнение выполняется командой
`python compare_prompts.py`: 15 запросов, три варианта на пяти текстах.
`--repeats` задаёт число повторов; `--output` сохраняет отчёт в файл.
Файлы `output/` исключены из Git.

## Проверка

```bash
python -m unittest discover -s tests -v
```
