# avsec — захищене відео через наявний аналоговий відеотракт

Дослідницький стенд: відтворення алгоритму зі статті
Mardiyanto R., Suryoatmojo H., Setiawan F., Irfansyah A. N.
*Low Cost Analog Video Transmission Security of Unmanned Aerial Vehicle (UAV) based on
Linear Feedback Shift Register (LFSR)*, ISITIA 2021, с. 414–419,
DOI [10.1109/ISITIA52817.2021.9502241](https://doi.org/10.1109/ISITIA52817.2021.9502241) —
та побудова сильнішого захисту на стандартній криптографії з чесним експериментальним
порівнянням.

## Про яку архітектуру йдеться

**Це не повністю аналоговий шифратор.** Основний напрям проєкту — **гібридна система**:

* дані відео представляються **цифрово** і шифруються стандартним AEAD;
* захищені дані передаються **через наявний аналоговий композитний відеотракт**
  (рівні яскравості у видимій частині растру);
* аналоговими лишаються сигнал на композитному інтерфейсі та сам тракт передавання;
* уся обробка **до** передавача і **після** приймача — цифрова.

Базові методи `B0a`, `B1`, `B2` навпаки передають саме зображення аналоговим растром —
це опора для порівняння, а не захищений режим.

## Швидкий старт

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows; Linux: source .venv/bin/activate
pip install -e ".[dev]"
avsec info
```

Без встановлення пакета все працює і так:

```bash
set PYTHONPATH=src   &&  python -m avsec.cli info      # Windows cmd
PYTHONPATH=src python -m avsec.cli info                # bash
```

### Зручний веб-інтерфейс

```bash
avsec ui --port 8765
```

Локальна сторінка на `http://127.0.0.1:8765/` (на stdlib `http.server`, без вебфреймворків).
Вкладки: демонстрація, перестановка та атаки, порівняння B0–B4/P, залежність від
спотворень, добір параметрів, абляції, CVBS рівня B, бюджет каналу, звіт.
Ліворуч — спільна конфігурація: розмір кадру, профіль каналу з повзунками
(шум, підсилення, зміщення, смуга, довжина й частота пакетів пошкоджень),
спільний бюджет, набір методів, профілі B4 і P, параметри перестановки.
Можна завантажити власне зображення та будь-який файл із `configs/`.

### Команди

Усі приклади нижче перевірені на цьому коді.

```bash
avsec info                                                  # версія і середовище
avsec gendata --output data/generated                       # створити тестові дані
avsec budget --config configs/comparison.yaml               # бюджет каналу і затримка
avsec demo --config configs/smoke.yaml --output runs/demo   # один кадр через усі методи
avsec scramble --config configs/attack.yaml --output runs/scramble   # B1 і назад
avsec attack --config configs/attack.yaml --output runs/attack       # атаки + AEAD
avsec transmit --config configs/comparison.yaml --output runs/comparison
avsec transmit --config configs/comparison_bursty.yaml --output runs/bursty  # вирішальний прогін гіпотези
avsec simulate --config configs/cvbs.yaml --preset mild --output runs/cvbs
avsec sweep --config configs/comparison.yaml --output runs/sweeps
avsec ablate --config configs/comparison.yaml --output runs/ablations
avsec tune --config configs/tuning.yaml --quick --output runs/tuning
avsec benchmark --config configs/comparison.yaml --output runs/benchmark
avsec report --input runs/benchmark --output reports/benchmark
avsec hardware --device 0 --output runs/hardware            # тільки з реальним пристроєм
avsec ui --port 8765
```

Дослідницький конвеєр — від плану до згенерованої документації:

```bash
avsec dataset validate --config configs/research_main.yaml   # дублікати, витік між split
avsec protocol-check --config configs/research_main.yaml     # 35 перевірок AEAD/фреймінгу
avsec matrix --plan configs/research_main.yaml --output runs/main --dry-run
avsec matrix --plan configs/research_main.yaml --output runs/main --resume --workers 16
python scripts/run_program.py configs/research_main.yaml runs/main 16   # E01–E10
avsec analyze --input runs/main                              # ефекти, CI, Holm
avsec plots --input runs/main --formats png svg pdf          # каталог G01–G43
python scripts/publish.py runs/main                          # згенерувати docs/ і README
```

`avsec matrix` відновлюється після переривання і **не дублює** вже виконані
завдання: ідентифікатор завдання походить від конфігурації, хешу кліпу і seed,
а не від позиції у списку. `--dry-run` вимірює вартість одного кадру **для
кожного методу окремо** і оцінює прискорення від worker-ів за виміряною
ефективністю, а не діленням на їх кількість.

`avsec hardware` **не підмінює** відсутній пристрій синтетичними даними: якщо плати
захоплення немає, команда повідомляє про це і повертає код 2.

### Бібліотечний API

CLI і UI — тонкі оболонки над бібліотекою; той самий код можна вбудувати:

```python
from avsec.config import config_from_dict
from avsec.experiments import build_methods, build_sources
from avsec.utils import experiment_rng

cfg = config_from_dict({"methods": ["B4"], "channel": {"preset": "bursty"}})
method = build_methods(cfg)["B4"]
frame = build_sources(cfg)[0].frames[0]
result = method.process(frame, 0, experiment_rng(cfg.seed, "demo"))
print(result.metrics.psnr_full, result.metrics.coverage)
```

## Методи, які порівнюються

| Метод | Що це | Автентифікація |
| --- | --- | --- |
| `B0a` | незахищене зображення через аналоговий растр | немає |
| `B0d` | той самий цифровий транспорт **без** криптографії (діагностика) | немає |
| `B1` | перестановка блоків на LFSR — реконструкція статті 2021 | немає |
| `B2` | та сама перестановка з криптографічним генератором | немає |
| `B3` | AEAD цілого кадру однією одиницею, фрагментованою в тракті | є |
| `B4` | незалежно захищені смуги, один опис, налаштовані FEC і перемежування | є |
| `P`  | кілька незалежних описів смуги + BAWP + спільний добір параметрів | є |

`B4` і `P` — це **той самий клас конвеєра** з різними конфігураціями: різниця у
порівнянні — конфігурація, а не якість реалізації.

## Структура

| Каталог | Призначення |
| --- | --- |
| `src/avsec/sources` | тестовий матеріал, файли, адаптер плати захоплення |
| `src/avsec/source_coding` | смуги, описи, незалежні кодеки, збирання кадру |
| `src/avsec/crypto` | ключі, сеанси, ChaCha20-Poly1305/AES-GCM, nonce, захист від повторів |
| `src/avsec/framing` | канонічний бінарний формат і суворий обмежений парсер |
| `src/avsec/fec` | Reed–Solomon з явним обліком блоків, padding і стирань |
| `src/avsec/modem` | растровий модем, синхронізація, пілоти; `modem/cvbs.py` — рівень B |
| `src/avsec/interleaving` | послідовне, блочне і запропоноване розміщення (BAWP) |
| `src/avsec/channel` | модель прийнятого растру (рівень A) |
| `src/avsec/transmitter.py`, `src/avsec/receiver` | передавач і приймач |
| `src/avsec/baselines` | B0–B4 і P за спільним інтерфейсом |
| `src/avsec/attacks` | лабораторний криптоаналіз перестановок, перевірки протоколу |
| `src/avsec/optimization` | спільний бюджет, простір параметрів, добір із відсіканням |
| `src/avsec/evaluation` | метрики, агрегування, довірчі інтервали, графіки |
| `src/avsec/budget.py` | місткість каналу, накладні витрати, модельна затримка |
| `src/avsec/ui` | локальний веб-інтерфейс |

## Документація

* [docs/architecture.md](docs/architecture.md) — архітектура і два рівні моделювання
* [docs/protocol.md](docs/protocol.md) — специфікація формату та ключового контексту
* [docs/threat_model.md](docs/threat_model.md) — модель загроз і межі тверджень
* [docs/reconstruction_2021.md](docs/reconstruction_2021.md) — що саме взято зі статті, а що добудовано
* [docs/related_work.md](docs/related_work.md) — найближчі роботи і можливий внесок
* [docs/experiments.md](docs/experiments.md) — протокол експериментів і розділення даних
* [docs/hardware.md](docs/hardware.md) — вимоги до обладнання, кошторис, невиконані перевірки
* [docs/results.md](docs/results.md) — **підсумок фактично виконаних експериментів**
* [docs/defects_fixed.md](docs/defects_fixed.md) — журнал виправлених дефектів ревʼю
* [results/](results/) — сирі результати, графіки і згенеровані звіти, що стоять за цими числами

## Перевірки

```bash
PYTHONPATH=src python -m pytest -q
```

## Головний результат

<!-- AVSEC:RESULTS:BEGIN -->

**Перевага запропонованої схеми залежить від каналу.** `P` значуще краща за `B4` на bursty і значуще гірша на clean, mild. Загального виграшу немає, і це головний результат прогону, а не застереження до нього.

_Усі числа нижче походять з одного прогону:_ `run_id=07b257405f757223`, commit `26dcb4a75436e5ac0db9c369cbef8ff20128a44a`, Windows-11-10.0.26200-SP0, Python 3.12.10, NumPy 2.1.2. Матеріал: 34 кліпів / 34 сцен усього, з них 20 у цьому прогоні (split test); **синтетичні дані**.

- **Обсяг:** 62675 оброблених кадрів, 20 незалежних сцен, 5 профілів каналу, 8 методів.
- **`clean`:** P − B4 = -3.46 дБ [-3.89; -2.95] за 19 сценами — значуще, перевірене покриття 1.000 проти 1.000.
- **`mild`:** P − B4 = -3.46 дБ [-3.89; -2.95] за 19 сценами — значуще, перевірене покриття 1.000 проти 1.000.
- **`moderate`:** P − B4 = -0.006 дБ [-0.534; +0.590] за 19 сценами — різниця не встановлена, перевірене покриття 0.957 проти 0.860.
- **`bursty`:** P − B4 = +2.35 дБ [+1.47; +3.26] за 19 сценами — значуще, перевірене покриття 0.807 проти 0.686.
- **`harsh`:** P − B4 = -0.001 дБ [-0.002; -0.000] за 19 сценами — методи дали практично тотожний результат (|Δ| < 0,01 дБ), перевірене покриття 0.000 проти 0.000.
- **Перевірки протоколу:** 35/35 пройдено.
- **Бюджет:** B4 — 0.512 Мбіт/с корисних даних (ефективність 0.485), P — 0.384 Мбіт/с (0.363); модельна затримка 159 мс, логічний буфер 68.5 кБ.
- **Рисунки:** 43 з 43 каталогу G01–G43 побудовано; решта позначені `pending` з причиною у [docs/figures.md](docs/figures.md).
- **Апаратних вимірювань немає** (E11 не виконано): усе нижче — модель.

Повні таблиці: [docs/results.md](docs/results.md).  Сирі дані: `runs/main/`.
<!-- AVSEC:RESULTS:END -->

> Числа з попередніх прогонів перенесені до
> [docs/historical_results.md](docs/historical_results.md) і не змішуються з
> поточними. Найважливіша змістовна зміна після переходу на єдиний матричний
> прогін: перевага `P` виявилася **залежною від каналу**, а не загальною.

## Що цей репозиторій НЕ доводить

* Немає вимірювань на фізичному передавачі, приймачі чи платі захоплення.
* Немає моделі радіочастотного FM-тракту; рівень B — це композитний сигнал у
  основній смузі.
* Немає виміряних енергоспоживання, вартості чи маси.
* Правильна інтеграція AEAD, підтверджена перевірками, не доводить відсутності
  інших помилок протоколу.
* Усі поставлені в комплекті послідовності — синтетичні; це видно у полі
  `provenance` кожного джерела і у звіті.
* Перевага узгодженого добору параметрів встановлена **лише** для профілю
  каналу `bursty`, одного бюджету і синтетичних даних; перенесення на інші
  сімейства пошкоджень і на реальний тракт не перевірене.
* На чистому й слабко пошкодженому каналі запропонована схема **програє**
  простішому `B4`: вона платить за кілька описів і сильніший FEC якістю, яку
  там нічим не компенсувати. Це виміряно і показано, а не обійдене.
* Дві з двадцяти тестових сцен (`grid_fast`, `text_fast`) **не вміщуються** у
  спільний бюджет для цифрових методів. Вони позначені як capacity-відмови й
  відсутні в середніх цих методів; це властивість обраного профілю, а не збій.
* Перевага BAWP над глибоким блочним перемежуванням не є однозначною: за
  пошкодженням окремого кодового слова глибоке блочне краще, а BAWP виграє за
  ймовірністю втратити обидва описи однієї смуги. Обидві величини наведені.

## Ліцензія

MIT. Криптографічні примітиви беруться з бібліотеки `cryptography`, код
Reed–Solomon — з `reedsolo`; вони не переписуються тут.
