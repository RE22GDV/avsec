# avsec — звіт за прогоном comparison (канал moderate)

## Умови запуску

- конфігурація: `comparison`
- Python 3.12.10, Windows-11-10.0.26200-SP0
- пакети: PIL 11.0.0, cryptography 50.0.1, cv2 4.13.0, matplotlib 3.9.2, numpy 2.1.2, reedsolo unknown, scipy 1.16.0, yaml 6.0.2
- git commit: `репозиторій git відсутній`
- seed (невідтворювані секрети не входять): `20240909`
- ключі: режим `lab` (lab = відтворюваний ключ лише для бенчмарків)

## Бюджет каналу та затримка

**[ЗАПУСК]** Розрахунок за фактичними параметрами профілю.

- нестиснений опорний потік: 3.28 Мбіт/с (нестиснене 256x192, 8 біт, 8.333 кадр/с; ця величина НЕ вважається автоматично доступною в обраному тракті)

| метод | корисних Б/одиницю | Б у каналі/одиницю | одиниць/растр | корисна Мбіт/с | повна Мбіт/с | ефективність | накопич. рядків | модельна затримка, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0d | 640 | 1236 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| B3 | 640 | 1236 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| B4 | 640 | 1236 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| P | 320 | 820 | 6 | 0.384 | 1.056 | 0.363 | 278 | 333.600 |


**[МОДЕЛЬ]** Наведена затримка - віртуальний розклад моделі, а не вимірювання апаратури.

## Порівняння методів

> **Увага.** усі джерела процедурно згенеровані (SYNTHETIC DATA): висновки стосуються синтетичного стенда, а не реального відеотракту.

**[ЗАПУСК]** Спільний бюджет: {"rasters_per_frame": 3, "raster_rate_hz": 25.0, "source_fps": 8.333, "effective_display_fps": 8.333333333333334, "max_accumulation_rows": 300, "max_virtual_latency_s": 0.8, "max_receiver_buffer_kb": 512, "raster": "720x576", "active_window": [24, 8, 696, 568], "amplitude_window": [40, 216], "note": "однакові геометрія растру, частота растрів і амплітудне вікно для всіх методів; порядок модуляції та розмір комірки символу входять до простору пошуку, тому жодна схема не отримує непомітно додаткового часу, смуги чи потужності сигналу"}

| метод | автентифіковано | послідовностей | PSNR, дБ | 95% ДІ | перевірене покриття |
| --- | --- | --- | --- | --- | --- |
| B0a | ні | 21 | 21.315 | [20.50; 22.13] | 1.000 |
| B0d | ні | 21 | 26.310 | [23.92; 28.70] | 0.899 |
| B1 | ні | 21 | 20.363 | [19.31; 21.42] | 1.000 |
| B2 | ні | 21 | 19.990 | [18.97; 21.01] | 1.000 |
| B3 | так | 21 | 25.031 | [22.03; 28.03] | 0.726 |
| B4 | так | 21 | 24.673 | [22.34; 27.01] | 0.838 |
| P | так | 21 | 25.338 | [23.37; 27.31] | 0.914 |


### Парні порівняння (PSNR, за послідовностями)

| пара | середня різниця, дБ | 95% ДІ | значуще на рівні 95% |
| --- | --- | --- | --- |
| B0a - B0d | -4.995 | [-7.17; -2.82] | так |
| B0a - B1 | 0.952 | [0.02; 1.88] | так |
| B0a - B2 | 1.325 | [0.39; 2.27] | так |
| B0a - B3 | -3.715 | [-6.87; -0.56] | так |
| B0a - B4 | -3.358 | [-5.88; -0.83] | так |
| B0a - P | -4.023 | [-5.84; -2.21] | так |
| B0d - B1 | 5.946 | [3.45; 8.44] | так |
| B0d - B2 | 6.320 | [4.04; 8.60] | так |
| B0d - B3 | 1.279 | [-2.47; 5.03] | ні |
| B0d - B4 | 1.637 | [-2.12; 5.39] | ні |
| B0d - P | 0.971 | [-0.94; 2.89] | ні |
| B1 - B2 | 0.374 | [-0.61; 1.35] | ні |
| B1 - B3 | -4.667 | [-8.21; -1.12] | так |
| B1 - B4 | -4.310 | [-6.70; -1.92] | так |
| B1 - P | -4.975 | [-7.01; -2.94] | так |
| B2 - B3 | -5.041 | [-8.19; -1.89] | так |
| B2 - B4 | -4.683 | [-7.25; -2.12] | так |
| B2 - P | -5.349 | [-7.62; -3.08] | так |
| B3 - B4 | 0.358 | [-2.70; 3.41] | ні |
| B3 - P | -0.308 | [-3.97; 3.35] | ні |
| B4 - P | -0.666 | [-3.83; 2.50] | ні |


**[ІНЖЕНЕРНЕ ПРИПУЩЕННЯ]** Для аналогових базових методів `coverage = 1` означає лише «зображення відображено», а не «дані перевірені»: у B0a/B1/B2 криптографічної перевірки немає.

## Атаки на перестановочні базові методи

**[ЗАПУСК]** Параметри LFSR - наша реконструкція: `{"width": 16, "taps": [16, 15, 13, 4], "polynomial": "x^16 + x^15 + x^13 + x^4 + 1", "form": "fibonacci", "seed": 44257, "zero_state_policy": "force_one", "state_space": 65535, "note": "reconstruction: the 2021 paper does not specify these parameters"}`

| атака | вміст | успіх | точність перестановки | суміжність | PSNR відновлення, дБ | с |
| --- | --- | --- | --- | --- | --- | --- |
| chosen_plaintext_permutation_recovery | - | так | 1.000 | n/a | n/a | 0.002 |
| known_plaintext_tile_matching | smooth | так | 1.000 | n/a | n/a | 0.034 |
| boundary_compatibility_reassembly | smooth | ні | 0.000 | 0.949 | 24.681 | 0.103 |
| known_plaintext_tile_matching | edges | так | 0.333 | n/a | n/a | 0.053 |
| boundary_compatibility_reassembly | edges | ні | 0.010 | 0.034 | 6.033 | 0.107 |
| known_plaintext_tile_matching | text | так | 0.536 | n/a | n/a | 0.039 |
| boundary_compatibility_reassembly | text | ні | 0.000 | 0.031 | 13.027 | 0.117 |
| known_plaintext_tile_matching | texture | так | 1.000 | n/a | n/a | 0.046 |
| boundary_compatibility_reassembly | texture | ні | 0.005 | 0.674 | 14.620 | 0.155 |
| multi_frame_variance_fingerprint | - | так | 0.510 | n/a | n/a | 0.008 |
| lfsr_seed_bruteforce | - | так | n/a | n/a | n/a | 5.771 |


> відновлення однієї сталої перестановки нічого не говорить про незалежно перегенеровану перестановку; невдала атака не є доказом безпеки

### «Схожість» у сенсі статті 2021 року

**[ЗАПУСК]** Кореляція оригіналу зі скремблованим і з відновленим зображенням, |r|·100%.

| seed | вміст | оригінал vs скрембл, % | оригінал vs відновлено, % | точне відновлення |
| --- | --- | --- | --- | --- |
| 44257 | smooth | 5.805 | 100.000 | так |
| 44257 | edges | 5.079 | 100.000 | так |
| 44257 | texture | 2.281 | 100.000 | так |
| 1234 | smooth | 6.315 | 100.000 | так |
| 1234 | edges | 4.769 | 100.000 | так |
| 1234 | texture | 2.205 | 100.000 | так |
| 1111 | smooth | 3.576 | 100.000 | так |
| 1111 | edges | 6.673 | 100.000 | так |
| 1111 | texture | 0.899 | 100.000 | так |


**[МОДЕЛЬ]** Низька кореляція між кадрами не є доказом конфіденційності: атака за сумісністю меж і атака за відомим кадром працюють попри неї.

### Перевірки протоколу AEAD

| перевірка | очікувано | результат | пройдено |
| --- | --- | --- | --- |
| valid unit | accept | ACCEPTED | так |
| replay of an accepted unit | reject | REJECTED (ReplayDetected) | так |
| modified ciphertext | reject | REJECTED (AuthenticationFailed) | так |
| modified tag | reject | REJECTED (AuthenticationFailed) | так |
| modified authenticated coordinates | reject | REJECTED (AuthenticationFailed) | так |
| unit moved to another stripe | reject | REJECTED (AuthenticationFailed) | так |
| unit moved to another frame | reject | REJECTED (AuthenticationFailed) | так |
| unit from another session | reject | REJECTED (AuthenticationFailed) | так |
| wrong master key | reject | REJECTED (AuthenticationFailed) | так |
| unknown protocol version | reject | REJECTED (UnknownVersion) | так |
| out-of-range payload length | reject | REJECTED (FramingError) | так |
| geometry outside the frame | reject | REJECTED (FramingError) | так |
| header CRC corruption | reject | REJECTED (FramingError) | так |
| modification fully repaired by FEC | not a forgery (identical protected message restored) | restored | так |
| modification beyond FEC capability | rejected by FEC or by AEAD | uncorrectable, discarded by FEC | так |


## Абляції запропонованого методу

**[ЗАПУСК]** криптографічна міцність і довжина тега (16 байтів) зафіксовані в усіх варіантах; зменшення захисту ніколи не використовується як спосіб показати виграш

| варіант | допустимий | PSNR, дБ | покриття | растрів/кадр | причина відхилення |
| --- | --- | --- | --- | --- | --- |
| P (proposed) | так | 25.453 | 0.969 | 3.000 |  |
| single description | так | 26.906 | 0.961 | 2.375 |  |
| sequential placement | так | 24.523 | 0.781 | 3.000 |  |
| plain block interleaving | так | 23.912 | 0.848 | 3.000 |  |
| deep block interleaving | так | 25.244 | 0.961 | 3.000 |  |
| smaller AEAD unit | ні |  |  |  | CapacityExceeded: frame 0 needs 4 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| larger AEAD unit | ні |  |  |  | CapacityExceeded: frame 0 needs 8 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| weaker FEC | так | 20.504 | 0.758 | 2.000 |  |
| stronger FEC | ні |  |  |  | CapacityExceeded: frame 0 needs 4 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| shallow BAWP window | так | 25.495 | 0.965 | 3.000 |  |


## Залежність від сили спотворень

| канал | метод | PSNR, дБ | покриття | растрів/кадр | кадрів |
| --- | --- | --- | --- | --- | --- |
| clean | B0a | 54.424 | 1.000 | 1.000 | 16 |
| clean | B0d | 29.338 | 1.000 | 2.000 | 16 |
| clean | B1 | 54.424 | 1.000 | 1.000 | 16 |
| clean | B2 | 54.424 | 1.000 | 1.000 | 16 |
| clean | B3 | 29.338 | 1.000 | 1.375 | 16 |
| clean | B4 | 29.338 | 1.000 | 2.000 | 16 |
| clean | P | 25.614 | 1.000 | 3.000 | 16 |
| mild | B0a | 29.053 | 1.000 | 1.000 | 16 |
| mild | B0d | 29.338 | 1.000 | 2.000 | 16 |
| mild | B1 | 27.595 | 1.000 | 1.000 | 16 |
| mild | B2 | 27.556 | 1.000 | 1.000 | 16 |
| mild | B3 | 29.338 | 1.000 | 1.375 | 16 |
| mild | B4 | 29.338 | 1.000 | 2.000 | 16 |
| mild | P | 25.614 | 1.000 | 3.000 | 16 |
| moderate | B0a | 22.040 | 1.000 | 1.000 | 16 |
| moderate | B0d | 28.180 | 0.953 | 2.000 | 16 |
| moderate | B1 | 20.464 | 1.000 | 1.000 | 16 |
| moderate | B2 | 20.197 | 1.000 | 1.000 | 16 |
| moderate | B3 | 23.704 | 0.688 | 1.375 | 16 |
| moderate | B4 | 24.690 | 0.867 | 2.000 | 16 |
| moderate | P | 25.533 | 0.961 | 3.000 | 16 |
| bursty | B0a | 16.913 | 1.000 | 1.000 | 16 |
| bursty | B0d | 17.283 | 0.523 | 2.000 | 16 |
| bursty | B1 | 13.975 | 1.000 | 1.000 | 16 |
| bursty | B2 | 15.961 | 1.000 | 1.000 | 16 |
| bursty | B3 | 22.919 | 0.625 | 1.375 | 16 |
| bursty | B4 | 15.773 | 0.555 | 2.000 | 16 |
| bursty | P | 21.983 | 0.738 | 3.000 | 16 |
| harsh | B0a | 16.040 | 1.000 | 1.000 | 16 |
| harsh | B0d | 11.990 | 0.000 | 2.000 | 16 |
| harsh | B1 | 14.231 | 1.000 | 1.000 | 16 |
| harsh | B2 | 14.989 | 1.000 | 1.000 | 16 |
| harsh | B3 | 11.990 | 0.000 | 1.375 | 16 |
| harsh | B4 | 11.990 | 0.000 | 2.000 | 16 |
| harsh | P | 11.990 | 0.000 | 3.000 | 16 |


![psnr_vs_channel](../../runs/benchmark/psnr_vs_channel.png)

![coverage_vs_channel](../../runs/benchmark/coverage_vs_channel.png)

## Наскрізна демонстрація через модель CVBS (рівень B)

**[ЗАПУСК]** Профіль каналу: `mild`, метод `B4`.

- якість: PSNR 36.48 дБ, перевірене покриття 1.000
- статуси одиниць: {"verified": 8}

- растр: відновлено рядків 576/576, полів 2, фронтів синхронізації 927
- растр: відновлено рядків 576/576, полів 2, фронтів синхронізації 926

**[ІНЖЕНЕРНЕ ПРИПУЩЕННЯ]** спрощена структура 625/50 - перелік відмінностей у CVBSProfile.describe()['simplifications']; це не сертифікована реалізація стандарту і вона не містить моделі радіочастотного тракту

Спрощення моделі CVBS:

- vertical interval: 5 equalising + 5 broad + 5 equalising half-lines only
- no teletext / VITS / VITC lines
- no colour subcarrier, no burst, no PAL phase alternation
- no RF or FM link model

## Демонстрація на одному кадрі

| метод | PSNR, дБ | SSIM | покриття | перевірено одиниць | надіслано одиниць | растрів |
| --- | --- | --- | --- | --- | --- | --- |
| B0a | 24.637 | 0.531 | 1.000 | 0 | 0 | 1 |
| B0d | 36.483 | 0.887 | 1.000 | 8 | 8 | 2 |
| B1 | 22.654 | 0.423 | 1.000 | 0 | 0 | 1 |
| B2 | 23.039 | 0.420 | 1.000 | 0 | 0 | 1 |
| B3 | 36.483 | 0.887 | 1.000 | 1 | 1 | 1 |
| B4 | 14.335 | 0.776 | 0.000 | 0 | 8 | 2 |
| P | 34.768 | 0.882 | 0.812 | 13 | 16 | 3 |


### B0a
- original: ![B0a-original](../../runs/benchmark/images/B0a/original.png)
- transmitted: ![B0a-transmitted](../../runs/benchmark/images/B0a/transmitted.png)
- received: ![B0a-received](../../runs/benchmark/images/B0a/received.png)
- reconstructed: ![B0a-reconstructed](../../runs/benchmark/images/B0a/reconstructed.png)
- availability: ![B0a-availability](../../runs/benchmark/images/B0a/availability.png)

### B0d
- original: ![B0d-original](../../runs/benchmark/images/B0d/original.png)
- transmitted: ![B0d-transmitted](../../runs/benchmark/images/B0d/transmitted.png)
- received: ![B0d-received](../../runs/benchmark/images/B0d/received.png)
- reconstructed: ![B0d-reconstructed](../../runs/benchmark/images/B0d/reconstructed.png)
- availability: ![B0d-availability](../../runs/benchmark/images/B0d/availability.png)

### B1
- original: ![B1-original](../../runs/benchmark/images/B1/original.png)
- transmitted: ![B1-transmitted](../../runs/benchmark/images/B1/transmitted.png)
- received: ![B1-received](../../runs/benchmark/images/B1/received.png)
- reconstructed: ![B1-reconstructed](../../runs/benchmark/images/B1/reconstructed.png)
- availability: ![B1-availability](../../runs/benchmark/images/B1/availability.png)

### B2
- original: ![B2-original](../../runs/benchmark/images/B2/original.png)
- transmitted: ![B2-transmitted](../../runs/benchmark/images/B2/transmitted.png)
- received: ![B2-received](../../runs/benchmark/images/B2/received.png)
- reconstructed: ![B2-reconstructed](../../runs/benchmark/images/B2/reconstructed.png)
- availability: ![B2-availability](../../runs/benchmark/images/B2/availability.png)

### B3
- original: ![B3-original](../../runs/benchmark/images/B3/original.png)
- transmitted: ![B3-transmitted](../../runs/benchmark/images/B3/transmitted.png)
- received: ![B3-received](../../runs/benchmark/images/B3/received.png)
- reconstructed: ![B3-reconstructed](../../runs/benchmark/images/B3/reconstructed.png)
- availability: ![B3-availability](../../runs/benchmark/images/B3/availability.png)

### B4
- original: ![B4-original](../../runs/benchmark/images/B4/original.png)
- transmitted: ![B4-transmitted](../../runs/benchmark/images/B4/transmitted.png)
- received: ![B4-received](../../runs/benchmark/images/B4/received.png)
- reconstructed: ![B4-reconstructed](../../runs/benchmark/images/B4/reconstructed.png)
- availability: ![B4-availability](../../runs/benchmark/images/B4/availability.png)

### P
- original: ![P-original](../../runs/benchmark/images/P/original.png)
- transmitted: ![P-transmitted](../../runs/benchmark/images/P/transmitted.png)
- received: ![P-received](../../runs/benchmark/images/P/received.png)
- reconstructed: ![P-reconstructed](../../runs/benchmark/images/P/reconstructed.png)
- availability: ![P-availability](../../runs/benchmark/images/P/availability.png)

## Що НЕ підтверджено цим звітом

**[НЕ ВИКОНАНО]** Вимірювання на фізичному передавачі/приймачі (TS5828, video grabber, Raspberry Pi) не проводилися.
**[НЕ ВИКОНАНО]** Радіочастотний FM-тракт не змодельовано; результати рівня B стосуються лише композитного відеосигналу в основній смузі.
**[НЕ ВИКОНАНО]** Енергоспоживання, вартість і маса апаратури не вимірювалися.
**[ГІПОТЕЗА]** Твердження про перевагу узгодженого добору параметрів дійсне лише в межах перевірених конфігурацій і моделей каналу; за їх межами воно не встановлене.
**[МОДЕЛЬ]** Правильна інтеграція AEAD, підтверджена перевірками, не доводить відсутності інших помилок протоколу.
