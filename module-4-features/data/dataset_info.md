# Описание датасета: Online Shoppers Purchasing Intention

## Источник

- **Kaggle:** https://www.kaggle.com/datasets/roshansharma/online-shoppers-intention
- **UCI ML Repository:** https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
- **Оригинальная статья:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893–6908 (2019).

## Общая информация

| Параметр | Значение |
|---|---|
| Всего записей | 12 330 |
| Признаков | 17 + целевая `Revenue` |
| Классов | 2 (True = покупка, False = нет) |
| Доля покупок (True) | ~15.5% (1 908 сессий) |
| Доля без покупки (False) | ~84.5% (10 422 сессии) |
| Тип задачи | Бинарная классификация |
| Дисбаланс классов | 84.5 / 15.5 — сильный |

## Структура признаков

| Тип признака | Количество | Примеры |
|---|---|---|
| Числовые (административные) | 2 | `Administrative`, `Administrative_Duration` |
| Числовые (информационные) | 2 | `Informational`, `Informational_Duration` |
| Числовые (товарные) | 2 | `ProductRelated`, `ProductRelated_Duration` |
| Числовые (метрики качества) | 4 | `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Категориальные (технические) | 4 | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| Категориальные (поведенческие) | 2 | `VisitorType`, `Weekend` |
| Временные | 1 | `Month` |
| Целевая | 1 | `Revenue` |

## Описание ключевых столбцов

| Столбец | Тип | Описание | Пропуски |
|---|---|---|---|
| `Administrative` | int | Число посещённых административных страниц | 0% |
| `Administrative_Duration` | float | Суммарное время на административных страницах (сек) | 0% |
| `Informational` | int | Число посещённых информационных страниц | 0% |
| `Informational_Duration` | float | Суммарное время на информационных страницах (сек) | 0% |
| `ProductRelated` | int | Число посещённых товарных страниц | 0% |
| `ProductRelated_Duration` | float | Суммарное время на товарных страницах (сек) | 0% |
| `BounceRates` | float | Средний показатель отказов по страницам сессии | 0% |
| `ExitRates` | float | Средний показатель выходов по страницам сессии | 0% |
| `PageValues` | float | Средняя ценность страниц, посещённых до транзакции | 0% |
| `SpecialDay` | float | Близость сессии к празднику (0 = далеко, 1 = близко) | 0% |
| `Month` | object | Месяц сессии (`Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`) | 0% |
| `OperatingSystems` | int | ID операционной системы пользователя | 0% |
| `Browser` | int | ID браузера пользователя | 0% |
| `Region` | int | ID региона пользователя | 0% |
| `TrafficType` | int | ID типа трафика | 0% |
| `VisitorType` | object | Тип посетителя (`Returning_Visitor`, `New_Visitor`, `Other`) | 0% |
| `Weekend` | bool | Признак выходного дня | 0% |
| `Revenue` | bool | **Целевая**: True = сессия завершилась покупкой | 0% |

## Распределение классов

- **False** (нет покупки): 10 422 сессии (84.5%)
- **True** (покупка): 1 908 сессий (15.5%)

Дисбаланс сильный — это требует использования `class_weight='balanced'`, `scale_pos_weight` (XGBoost), `auto_class_weights` (CatBoost), а также метрик F1 и ROC-AUC вместо Accuracy.

## Особенности данных

- **Пропусков нет** — все 12 330 строк заполнены полностью.
- **Нет временных меток** в явном виде — есть только `Month` (месяц), из которого можно извлечь номер месяца (`MonthNum`) для модели.
- **Категориальные ID** (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) не несут порядкового смысла — их нужно кодировать (Label/One-Hot).
- **`PageValues` — сильнейший предиктор** покупки: у купивших сессий медиана заметно выше.
- **`BounceRates` и `ExitRates`** сильно коррелируют между собой — учитывать при интерпретации Feature Importance и SHAP.

## Созданные признаки (Feature Engineering)

В ноутбуке созданы новые признаки на основе поведения пользователя:

| Признак | Формула | Зачем |
|---|---|---|
| `TotalPages` | `Administrative + Informational + ProductRelated` | Общая активность в сессии |
| `TotalDuration` | `Administrative_Duration + Informational_Duration + ProductRelated_Duration` | Общее время на сайте |
| `AvgPageValue` | `PageValues / (TotalPages + 1)` | Нормированная ценность страницы |
| `BounceExitRatio` | `BounceRates / (ExitRates + 0.001)` | Отношение отказов к выходам |
| `IsNewVisitor` | `(VisitorType == 'New_Visitor').astype(int)` | Бинарный флаг нового посетителя |
| `IsWeekend` | `Weekend.astype(int)` | Бинарный флаг выходного дня |
| `MonthNum` | `Month.map({...})` | Числовой номер месяца |
| `IsHighTraffic` | `(TrafficType > 10).astype(int)` | Флаг «высокого» трафика |

## Файл данных

- **Путь в репозитории:** `module-4-features/data/online_shoppers_intention.csv`
- **Путь в Colab:** `online_shoppers_intention.csv` (в корне файловой системы)
- **Кодировка:** UTF-8
- **Разделитель:** запятая
