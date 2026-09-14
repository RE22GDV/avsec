# Набори даних

<!-- DOCNAV -->
> **Призначення документа.** Паспорт кожного набору: походження, ліцензії, спосіб виготовлення
> кліпів, одиниця незалежності та межі того, що на ньому можна довести.
>
> **Цільовий читач.** Читач, який оцінює межі узагальнення висновків.
>
> **Пов'язані документи:** [протокол експериментів](experiments.md) · [реєстр тверджень](claims.md) · [road map](../ROADMAP.md) · [README](../README.md)
<!-- /DOCNAV -->

<!-- TOC -->
<details>
<summary><b>Зміст</b></summary>

- [1. `main` — процедурний матеріал](#1-main--процедурний-матеріал)
- [2. `drone` — одна фотографія, 17 послідовностей](#2-drone--одна-фотографія-17-послідовностей)
- [3. `natural` — 24 незалежних джерел](#3-natural--24-незалежних-джерел)
  - [Як добиралися](#як-добиралися)
  - [Що це за файли](#що-це-за-файли)
  - [Повний перелік джерел](#повний-перелік-джерел)
- [4. Що жоден із трьох наборів не дає](#4-що-жоден-із-трьох-наборів-не-дає)

</details>
<!-- /TOC -->

Три набори, і вони відповідають на **різні** питання. Плутати їх — головна
пастка цієї роботи, і вона спрацювала один раз (дефект
[R07](revision_history.md#r07-сімнадцять-вирізок-з-однієї-фотографії-рахувалися-як-сімнадцять-джерел)).

| | `main` | `drone` | `natural` |
| --- | --- | --- | --- |
| **Матеріал** | процедурні патерни | одна фотографія з БпЛА | 24 фотографій з БпЛА |
| **Кліпів** | 34 | 17 | 24 |
| **Сцен** | 34 | 17 | 24 |
| **Незалежних джерел** | 34 | **1** | **24** |
| **Одиниця незалежності** | сцена = генератор | одна фотографія | вихідний знімок |
| **Splits (джерел)** | 8 / 6 / 20 | 0 / 0 / 1 | 5 / 5 / 14 |
| **Узагальнюється на** | нову процедурну сцену | нову вирізку **цього** знімка | **новий запис** |

> [!WARNING]
> Інтервал, порахований по набору `drone`, оцінює невизначеність **усередині
> однієї фотографії**. Він не має права стосуватися наступного польоту, і в
> жодній таблиці так не використовується. Для цього існує `natural`.

---

## 1. `main` — процедурний матеріал

34 сцен, згенерованих кодом: горизонт, дорога, цілі,
текстура, краї, хмари, сітка, слабке освітлення, текст. Кожна має власну точку
зйомки й експозицію.

**Навіщо.** Повний контроль над вмістом, детермінованість байт у байт і
можливість зробити сцену **навмисно важкою**. Дві сцени (`grid_fast`,
`text_fast`) не вміщуються у спільний бюджет для цифрових методів — і це
корисна властивість, а не вада: вони виявилися важчими за будь-який природний
кадр у наборі.

**Межа.** Процедурні патерни — не фотографії. Статистика країв, шуму й
кореляції в них інша, і саме тому потрібні два наступні набори.

```bash
avsec dataset validate --config configs/research_main.yaml
```

---

## 2. `drone` — одна фотографія, 17 послідовностей

| | |
| --- | --- |
| **Джерело** | Curonian Spit NP, аерофотознімок біля дюни Ефа, 12.05.2017 |
| **Автор** | A.Savin ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Curonian_Spit_NP_05-2017_img17_aerial_view_at_Epha_Dune.jpg)) |
| **Камера** | DJI FC6310 — сенсор Phantom 4 Pro |
| **Роздільність** | 4213 × 2633, **камерний оригінал**, не перекодований тут |
| **Ліцензія** | Free Art License 1.3 |
| **Файл** | `data/real/curonian_spit_epha_dune.jpg` — **закомічений** у репозиторій |

Кожен кліп — вікно кадрування, що панорамує знімком між двома оголошеними
наперед прямокутниками.

**Що виміряно про сам набір.** 63 пар
вирізок зі 136 **геометрично перетинаються**.
Це не декларація, а результат `measure_crop_overlap()`, записаний у маніфест.
Раніше в документації стояло «вирізки непересічні» — це було неправдою.

**Чому весь набір — `test`.** Ділити одну фотографію на
calibration/validation/test не має сенсу: усі три частини поділяли б сенсор,
освітлення й місцевість. Валідатор тепер позначає таку спробу як **помилку**.
Нічого на цьому знімку не налаштовувалося — конфігурації прийшли з `main`.

```bash
python scripts/fetch_drone_photo.py     # перевірити або завантажити
```

---

## 3. `natural` — 24 незалежних джерел

Щоб висновок міг стосуватися **нового запису**, потрібні різні записи.

| | |
| --- | --- |
| **Джерел** | 24 фотографій, 24 різних авторів |
| **Камер** | 9 моделей DJI: DJI Mavic Air, DJI Mavic Air 2, DJI Mavic Pro, DJI Mini 2, DJI Mini 3 Pro, DJI Phantom 4, DJI Phantom 4 Pro, DJI Phantom 4 Pro V2.0, DJI Zenmuse X5S (Inspire 2) |
| **Ліцензії** | CC BY-SA 2.0, CC BY-SA 4.0, CC0 |
| **Кліпів на джерело** | 1 — «більше даних» тут означає «більше знімків» |
| **Splits** | за **джерелом**: 5 calibration / 5 validation / 14 test |
| **Категорії** | призначені за **виміряною** градієнтною енергією знімка, до будь-якого результату |
| **Реєстр** | [`data/real/sources.json`](../data/real/sources.json) — URL, ліцензія, автор, камера, sha256 |
| **Файли** | **не** комітяться (0 МБ); завантажуються скриптом |

### Як добиралися

Детерміновано, і критерії зафіксовані в коді
([`scripts/select_natural_sources.py`](../scripts/select_natural_sources.py)):

1. запит до Wikimedia Commons за категоріями «Taken with DJI *model*»;
2. фільтр: JPEG, ≥ 3000×2000 px, вільна ліцензія, вказаний автор;
3. **одна фотографія на автора** — два акаунти однієї людини зводяться разом
   за відсортованими літерами імені;
4. квота розподіляється **між моделями камер**, щоб набір не виявився одним
   сенсором і одним об'єктивом;
5. порядок — за `pageid`, перший придатний виграє. Повторний запуск дає той
   самий реєстр.

### Що це за файли

Wikimedia просить не тягнути камерні оригінали пакетно і відповідає `429`.
Тому кожен знімок береться через `Special:FilePath?width=…` — це **рендер
thumbnailer-а з оригіналу**, а не сам оригінал, і манiфести кажуть саме так.

На вимірювання це не впливає: кліп — це вирізка близько тисячі пікселів
завширшки, усереднена до 256×192 ще до кодека, тобто далеко нижче роздільності
рендера.

```bash
python scripts/fetch_natural_sources.py           # завантажити й звірити хеші
python scripts/fetch_natural_sources.py --verify  # лише звірити
avsec dataset validate --config configs/research_natural.yaml
```

### Повний перелік джерел

| Знімок | Автор | Апарат | Ліцензія | Роздільність | Split | Категорія | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1062-mu-koh-lanta-national-park-04.jpg](https://commons.wikimedia.org/wiki/File:1062-mu-koh-lanta-national-park-04.jpg) | Wanjak Atikomchakorn | DJI Phantom 4 | CC BY-SA 4.0 | 3991×2245 | calibration | low-detail | `b6a8ba1749b8…` |
| [149DJI 0001.jpg](https://commons.wikimedia.org/wiki/File:149DJI_0001.jpg) | Max Bond | DJI Phantom 4 | CC BY-SA 4.0 | 4000×3000 | calibration | low-detail | `23962cd7f571…` |
| [1 pano xinping yangshupo (cropped).jpg](https://commons.wikimedia.org/wiki/File:1_pano_xinping_yangshupo_(cropped).jpg) | Chensiyuan | DJI Phantom 4 Pro | CC BY-SA 4.0 | 5905×3324 | calibration | low-detail | `8e64cabb883d…` |
| [2017 07 11 - RUFFIEU.jpg](https://commons.wikimedia.org/wiki/File:2017_07_11_-_RUFFIEU.jpg) | Lilo01260 | DJI Mavic Pro | CC BY-SA 4.0 | 4000×2250 | calibration | low-detail | `3290c71cb9dd…` |
| [99% water and 1% land. (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:99%25_water_and_1%25_land._(Unsplash).jpg) | Mifxal Latheef droneworxmv | DJI Phantom 4 Pro | CC0 | 4864×3648 | calibration | low-detail | `f50a220794a4…` |
| [Aerial composition. (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Aerial_composition._(Unsplash).jpg) | Sweet Ice Cream Photography  | DJI Phantom 4 Pro | CC0 | 5464×3070 | validation | low-detail | `81c22c19818a…` |
| [2017 06 Ali- 00977.jpg](https://commons.wikimedia.org/wiki/File:2017_06_Ali-_00977.jpg) | Alimdaihli | DJI Mavic Pro | CC BY-SA 4.0 | 4000×3000 | validation | low-detail | `2b0ee41becf9…` |
| [1933·上海公共租界工部局宰牲场·上海虹口·（正面俯拍）.jpg](https://commons.wikimedia.org/wiki/File:1933%C2%B7%E4%B8%8A%E6%B5%B7%E5%85%AC%E5%85%B1%E7%A7%9F%E7%95%8C%E5%B7%A5%E9%83%A8%E5%B1%80%E5%AE%B0%E7%89%B2%E5%9C%BA%C2%B7%E4%B8%8A%E6%B5%B7%E8%99%B9%E5%8F%A3%C2%B7%EF%BC%88%E6%AD%A3%E9%9D%A2%E4%BF%AF%E6%8B%8D%EF%BC%89.jpg) | Legolas1024 | DJI Mavic Pro | CC BY-SA 4.0 | 3874×2903 | validation | low-detail | `4cf578059a6a…` |
| [Broadwood's Tower.jpg](https://commons.wikimedia.org/wiki/File:Broadwood%27s_Tower.jpg) | Balazs Mocsar | DJI Phantom 4 Pro V2.0 | CC BY-SA 4.0 | 5472×3078 | validation | low-detail | `b065c69f5efc…` |
| [Country Pond.jpg](https://commons.wikimedia.org/wiki/File:Country_Pond.jpg) | Mnsesq | DJI Phantom 4 Pro V2.0 | CC BY-SA 4.0 | 5219×3477 | validation | low-detail | `2f0db3dd3c7a…` |
| [Idongandnamhae.jpg](https://commons.wikimedia.org/wiki/File:Idongandnamhae.jpg) | Chemsty | DJI Phantom 4 Pro V2.0 | CC BY-SA 4.0 | 5472×3078 | test | low-detail | `ebd6dd3ff5bf…` |
| [ITC Royal Bengalz.jpg](https://commons.wikimedia.org/wiki/File:ITC_Royal_Bengalz.jpg) | Chiranjit.das.official | DJI Zenmuse X5S (Inspire 2) | CC BY-SA 4.0 | 5250×3322 | test | low-detail | `434e479f812f…` |
| [15 Bern Photo by Giles Laurent.jpg](https://commons.wikimedia.org/wiki/File:15_Bern_Photo_by_Giles_Laurent.jpg) | Giles Laurent | DJI Mavic Air | CC BY-SA 4.0 | 4056×3040 | test | low-detail | `c79620fc3cc7…` |
| [2021-01-15-Sankt-Fides-Soelden.jpg](https://commons.wikimedia.org/wiki/File:2021-01-15-Sankt-Fides-Soelden.jpg) | Thomas Berwing | DJI Mini 2 | CC BY-SA 4.0 | 4000×2250 | test | low-detail | `bd420c265c7f…` |
| [Chateau de la Barolliere DJI 0001.jpg](https://commons.wikimedia.org/wiki/File:Chateau_de_la_Barolliere_DJI_0001.jpg) | Kywam | DJI Zenmuse X5S (Inspire 2) | CC BY-SA 4.0 | 6016×4008 | test | low-detail | `07f45bf7a471…` |
| [01-front.jpg](https://commons.wikimedia.org/wiki/File:01-front.jpg) | Sharkyaloha2017 | DJI Zenmuse X5S (Inspire 2) | CC BY-SA 4.0 | 5240×3928 | test | low-detail | `eddbd5865338…` |
| [02FuenteLavadero Guarnizo.jpg](https://commons.wikimedia.org/wiki/File:02FuenteLavadero_Guarnizo.jpg) | Luis Fermín TURIEL PEREDO | DJI Mini 2 | CC BY-SA 4.0 | 4000×2250 | test | low-detail | `93df8a2fa1de…` |
| ["Big Beach" (^387 explore 08-17-2021) - Flic…](https://commons.wikimedia.org/wiki/File:%22Big_Beach%22_(%5E387_explore_08-17-2021)_-_Flickr_-_Kirt_Edblom.jpg) | Kirt Edblom from Kihei, Hi,  | DJI Mini 2 | CC BY-SA 2.0 | 3720×2025 | test | low-detail | `b324d4f7e39a…` |
| [20220327耕文路站工地.jpg](https://commons.wikimedia.org/wiki/File:20220327%E8%80%95%E6%96%87%E8%B7%AF%E7%AB%99%E5%B7%A5%E5%9C%B0.jpg) | MasaneMiyaPA | DJI Mavic Air 2 | CC BY-SA 4.0 | 4000×3000 | test | low-detail | `9709c101aa19…` |
| [2022 Daly City BART.jpg](https://commons.wikimedia.org/wiki/File:2022_Daly_City_BART.jpg) | InvadingInvader | DJI Mavic Air 2 | CC BY-SA 4.0 | 4000×2250 | test | low-detail | `1c502306864d…` |
| [Hôpital Donka, vue aérien en 2022 02.jpg](https://commons.wikimedia.org/wiki/File:H%C3%B4pital_Donka,_vue_a%C3%A9rien_en_2022_02.jpg) | Aboubacarkhoraa | DJI Mavic Air 2 | CC BY-SA 4.0 | 4000×3000 | test | low-detail | `03960dd67408…` |
| [5ed Gorymdaith YesCymru - the 5th YesCymru a…](https://commons.wikimedia.org/wiki/File:5ed_Gorymdaith_YesCymru_-_the_5th_YesCymru_and_AUOB_march_in_Cardiff_1_Oct_2022_05.jpg) | Llywelyn2000 | DJI Mini 3 Pro | CC BY-SA 4.0 | 4032×2268 | test | low-detail | `155688b71ca5…` |
| [48-20230521McLoughlinHSAerial.jpg](https://commons.wikimedia.org/wiki/File:48-20230521McLoughlinHSAerial.jpg) | FriendlyToaster | DJI Mini 3 Pro | CC0 | 4242×2828 | test | low-detail | `eb1677119470…` |
| [2023-09-09 Itertal DJI (45).jpg](https://commons.wikimedia.org/wiki/File:2023-09-09_Itertal_DJI_(45).jpg) | Ladislaus Hoffner | DJI Mini 3 Pro | CC BY-SA 4.0 | 4032×3024 | test | low-detail | `05342627abc1…` |

---

## 4. Що жоден із трьох наборів не дає

* **Це не польотні записи.** Вікно, що рухається по нерухомому знімку, не має
  паралакса, скошування рядків, адаптації експозиції й руху в сцені. Жодне
  твердження в репозиторії на польотну динаміку не спирається.
* **Це не вибірка «усіх БпЛА».** Усі камери — побутові DJI; усі знімки — денні,
  переважно з висоти кількох десятків метрів.
* **Немає запису з реального відеотракту.** Канал — програмна модель; знімки
  входять у неї як ідеальні кадри, а не як те, що вже пройшло аналоговий тракт.

Наступний крок, який це закриває, — [E11 і польотні записи](../ROADMAP.md#чого-немає--і-що-саме-це-означає).
