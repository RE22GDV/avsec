# avsec — звіт за прогоном comparison (канал moderate)

## Умови запуску

- конфігурація: `comparison`
- Python 3.12.10, Windows-11-10.0.26200-SP0
- пакети: PIL 11.0.0, cryptography 50.0.1, cv2 4.13.0, matplotlib 3.9.2, numpy 2.1.2, reedsolo unknown, scipy 1.16.0, yaml 6.0.2
- git commit: `22a831f41bdff151337cebb124fe6dd6cf460d72`
- seed (невідтворювані секрети не входять): `20240909`
- ключі: режим `lab` (lab = відтворюваний ключ лише для бенчмарків)

## Бюджет каналу та затримка

**[ЗАПУСК]** Розрахунок за фактичними параметрами профілю.

- нестиснений опорний потік: 3.28 Мбіт/с (нестиснене 256x192, 8 біт, 8.333 кадр/с; ця величина НЕ вважається автоматично доступною в обраному тракті)

| метод | корисних Б/одиницю | Б у каналі/одиницю | одиниць/растр | корисна Мбіт/с | повна Мбіт/с | ефективність | накопич. рядків | модельна затримка, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0d | 640 | 1288 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| B3 | 640 | 1288 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| B4 | 640 | 1288 | 4 | 0.512 | 1.056 | 0.485 | 278 | 333.600 |
| P | 320 | 872 | 6 | 0.384 | 1.056 | 0.363 | 278 | 333.600 |


**[МОДЕЛЬ]** Наведена затримка - віртуальний розклад моделі, а не вимірювання апаратури.

## Порівняння методів

> **Увага.** усі джерела процедурно згенеровані (SYNTHETIC DATA): висновки стосуються синтетичного стенда, а не реального відеотракту.

**[ЗАПУСК]** Спільний бюджет: {"rasters_per_frame": 3, "raster_rate_hz": 25.0, "source_fps": 8.333, "effective_display_fps": 8.333333333333334, "max_accumulation_rows": 300, "max_virtual_latency_s": 0.8, "max_receiver_buffer_kb": 512, "raster": "720x576", "active_window": [24, 8, 696, 568], "amplitude_window": [40, 216], "note": "однакові геометрія растру, частота растрів і амплітудне вікно для всіх методів; порядок модуляції та розмір комірки символу входять до простору пошуку, тому жодна схема не отримує непомітно додаткового часу, смуги чи потужності сигналу"}

| метод | автентифіковано | послідовностей | PSNR, дБ | 95% ДІ | перевірене покриття |
| --- | --- | --- | --- | --- | --- |
| B0a | ні | 21 | 21.222 | [20.22; 22.23] | 1.000 |
| B0d | ні | 21 | 26.116 | [23.11; 29.12] | 0.875 |
| B1 | ні | 21 | 19.965 | [18.88; 21.05] | 1.000 |
| B2 | ні | 21 | 19.945 | [18.88; 21.01] | 1.000 |
| B3 | так | 21 | 25.813 | [22.35; 29.28] | 0.750 |
| B4 | так | 21 | 25.682 | [22.70; 28.66] | 0.842 |
| P | так | 21 | 25.050 | [22.98; 27.12] | 0.935 |


### Парні порівняння (PSNR, за послідовностями)

| пара | середня різниця, дБ | 95% ДІ | значуще на рівні 95% |
| --- | --- | --- | --- |
| B0a - B0d | -4.894 | [-7.73; -2.06] | так |
| B0a - B1 | 1.258 | [0.82; 1.70] | так |
| B0a - B2 | 1.277 | [0.88; 1.67] | так |
| B0a - B3 | -4.590 | [-7.78; -1.40] | так |
| B0a - B4 | -4.459 | [-7.25; -1.67] | так |
| B0a - P | -3.828 | [-5.75; -1.91] | так |
| B0d - B1 | 6.152 | [3.34; 8.96] | так |
| B0d - B2 | 6.171 | [3.33; 9.01] | так |
| B0d - B3 | 0.304 | [-2.64; 3.25] | ні |
| B0d - B4 | 0.435 | [-0.61; 1.48] | ні |
| B0d - P | 1.066 | [-0.33; 2.47] | ні |
| B1 - B2 | 0.020 | [-0.07; 0.11] | ні |
| B1 - B3 | -5.848 | [-9.21; -2.48] | так |
| B1 - B4 | -5.717 | [-8.45; -2.99] | так |
| B1 - P | -5.086 | [-7.03; -3.15] | так |
| B2 - B3 | -5.868 | [-9.22; -2.52] | так |
| B2 - B4 | -5.737 | [-8.49; -2.99] | так |
| B2 - P | -5.105 | [-7.07; -3.14] | так |
| B3 - B4 | 0.131 | [-2.91; 3.17] | ні |
| B3 - P | 0.763 | [-2.04; 3.57] | ні |
| B4 - P | 0.632 | [-1.02; 2.28] | ні |


**[ІНЖЕНЕРНЕ ПРИПУЩЕННЯ]** Для аналогових базових методів `coverage = 1` означає лише «зображення відображено», а не «дані перевірені»: у B0a/B1/B2 криптографічної перевірки немає.

## Атаки на перестановочні базові методи

**[ЗАПУСК]** Параметри LFSR - наша реконструкція: `{"width": 16, "taps": [16, 15, 13, 4], "polynomial": "x^16 + x^15 + x^13 + x^4 + 1", "form": "fibonacci", "seed": 44257, "zero_state_policy": "force_one", "state_space": 65535, "note": "reconstruction: the 2021 paper does not specify these parameters"}`

| атака | вміст | успіх | точність перестановки | суміжність | PSNR відновлення, дБ | с |
| --- | --- | --- | --- | --- | --- | --- |
| chosen_plaintext_permutation_recovery | - | так | 1.000 | n/a | n/a | 0.004 |
| known_plaintext_tile_matching | smooth | так | 1.000 | n/a | n/a | 0.056 |
| boundary_compatibility_reassembly | smooth | ні | 0.000 | 0.949 | 24.681 | 0.180 |
| known_plaintext_tile_matching | edges | так | 0.333 | n/a | n/a | 0.058 |
| boundary_compatibility_reassembly | edges | ні | 0.010 | 0.034 | 6.033 | 0.169 |
| known_plaintext_tile_matching | text | так | 0.536 | n/a | n/a | 0.060 |
| boundary_compatibility_reassembly | text | ні | 0.000 | 0.031 | 13.027 | 0.186 |
| known_plaintext_tile_matching | texture | так | 1.000 | n/a | n/a | 0.058 |
| boundary_compatibility_reassembly | texture | ні | 0.005 | 0.674 | 14.620 | 0.223 |
| multi_frame_variance_fingerprint | - | так | 0.510 | n/a | n/a | 0.009 |
| lfsr_seed_bruteforce | - | так | n/a | n/a | n/a | 9.182 |


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
| valid unit (positive control) | accept | ACCEPTED | так |
| replay of an accepted unit | ReplayDetected | REJECTED (ReplayDetected) | так |
| modified ciphertext | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified tag | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| truncated ciphertext | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'frame_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'stripe_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'desc_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'seg_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'codec_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'profile_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'payload_len' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'flags' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'session_epoch' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'stream_id' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'n_segs' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated field 'n_descs' | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| modified authenticated geometry | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| unit replayed into another session | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| unit replayed into another epoch of the same session | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| wrong master key | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| unit replayed into another stream | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| negative control: empty AAD instead of the header | AuthenticationFailed | REJECTED (AuthenticationFailed) | так |
| counters are allocated once and never repeat | strictly increasing, no caller-chosen counter | 0 then 1; seal() takes no counter argument | так |
| counter beyond the profile limit | NonceExhausted | REJECTED (NonceExhausted) | так |
| unknown protocol version | UnknownVersion | REJECTED (UnknownVersion) | так |
| unknown transport profile | UnknownProfile | REJECTED (UnknownProfile) | так |
| zero payload length | FramingError | REJECTED (FramingError) | так |
| payload length above the profile bound | FramingError | REJECTED (FramingError) | так |
| segment index outside the segment count | FramingError | REJECTED (FramingError) | так |
| corrupted header CRC | FramingError | REJECTED (FramingError) | так |
| geometry outside the frame | FramingError | REJECTED (FramingError) | так |
| geometry phase inconsistent with the step | FramingError | REJECTED (FramingError) | так |
| modification fully repaired by FEC | not a forgery: the identical protected message is restored | restored | так |
| modification beyond the FEC capability | rejected by FEC or by AEAD | uncorrectable, discarded by FEC | так |


## Абляції запропонованого методу

**[ЗАПУСК]** криптографічна міцність і довжина тега (16 байтів) зафіксовані в усіх варіантах; зменшення захисту ніколи не використовується як спосіб показати виграш

| варіант | допустимий | PSNR, дБ | покриття | растрів/кадр | причина відхилення |
| --- | --- | --- | --- | --- | --- |
| P (proposed) | так | 25.548 | 0.973 | 3.000 |  |
| single description | так | 27.654 | 1.000 | 2.375 |  |
| sequential placement | так | 24.841 | 0.895 | 3.000 |  |
| plain block interleaving | так | 24.610 | 0.910 | 3.000 |  |
| deep block interleaving | так | 25.617 | 0.992 | 3.000 |  |
| smaller AEAD unit | ні |  |  |  | CapacityExceeded: frame 0 needs 4 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| larger AEAD unit | ні |  |  |  | CapacityExceeded: frame 0 needs 8 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| weaker FEC | так | 23.620 | 0.812 | 2.000 |  |
| stronger FEC | ні |  |  |  | CapacityExceeded: frame 0 needs 4 rasters, the profile allows 3; lower the quality/resolution or raise the capacity |
| shallow BAWP window | так | 25.529 | 0.973 | 3.000 |  |


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
| mild | B0a | 29.163 | 1.000 | 1.000 | 16 |
| mild | B0d | 29.338 | 1.000 | 2.000 | 16 |
| mild | B1 | 27.639 | 1.000 | 1.000 | 16 |
| mild | B2 | 27.607 | 1.000 | 1.000 | 16 |
| mild | B3 | 29.338 | 1.000 | 1.375 | 16 |
| mild | B4 | 29.338 | 1.000 | 2.000 | 16 |
| mild | P | 25.614 | 1.000 | 3.000 | 16 |
| moderate | B0a | 21.454 | 1.000 | 1.000 | 16 |
| moderate | B0d | 28.007 | 0.953 | 2.000 | 16 |
| moderate | B1 | 20.077 | 1.000 | 1.000 | 16 |
| moderate | B2 | 19.989 | 1.000 | 1.000 | 16 |
| moderate | B3 | 28.727 | 0.938 | 1.375 | 16 |
| moderate | B4 | 27.294 | 0.938 | 2.000 | 16 |
| moderate | P | 25.529 | 0.973 | 3.000 | 16 |
| bursty | B0a | 15.375 | 1.000 | 1.000 | 16 |
| bursty | B0d | 15.343 | 0.562 | 2.000 | 16 |
| bursty | B1 | 15.319 | 1.000 | 1.000 | 16 |
| bursty | B2 | 15.267 | 1.000 | 1.000 | 16 |
| bursty | B3 | 24.301 | 0.625 | 1.375 | 16 |
| bursty | B4 | 15.343 | 0.562 | 2.000 | 16 |
| bursty | P | 21.345 | 0.777 | 3.000 | 16 |
| harsh | B0a | 14.591 | 0.875 | 1.000 | 16 |
| harsh | B0d | 11.990 | 0.000 | 2.000 | 16 |
| harsh | B1 | 14.026 | 0.875 | 1.000 | 16 |
| harsh | B2 | 14.014 | 0.875 | 1.000 | 16 |
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
| B0a | 24.118 | 0.513 | 1.000 | 0 | 0 | 1 |
| B0d | 36.483 | 0.887 | 1.000 | 8 | 8 | 2 |
| B1 | 23.218 | 0.419 | 1.000 | 0 | 0 | 1 |
| B2 | 23.083 | 0.417 | 1.000 | 0 | 0 | 1 |
| B3 | 36.483 | 0.887 | 1.000 | 1 | 1 | 1 |
| B4 | 36.483 | 0.887 | 1.000 | 8 | 8 | 2 |
| P | 34.605 | 0.869 | 1.000 | 16 | 16 | 3 |


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
