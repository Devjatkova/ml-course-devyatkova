# Описание датасета Telco Customer Churn

**Источник:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Структура

- Всего примеров: 7043
- Признаков: 19 (+ `customerID` + целевая `Churn`)
- Числовых признаков: 3 (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Категориальных признаков: 16

## Описание ключевых столбцов

| Столбец | Тип | Описание | Пропуски |
|---|---|---|---|
| customerID | object | Уникальный ID | 0% |
| gender | object | Пол | 0% |
| SeniorCitizen | int (0/1) | Пенсионер | 0% |
| tenure | int | Месяцев с компанией | 0% |
| MonthlyCharges | float | Ежемесячная плата | 0% |
| TotalCharges | object | Общая сумма | **~0.2%** |
| Contract | object | Тип контракта | 0% |
| Churn | object | **Целевая** | 0% |

## Распределение классов

- No (остался): 5174 (73.5%)
- Yes (ушёл): 1869 (26.5%) — **умеренный дисбаланс**.

## Файл данных в репозитории

`module-3-churn/data/churn_info.md`