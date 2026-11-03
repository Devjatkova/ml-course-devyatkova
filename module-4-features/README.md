# 🛒 Module 4 — Online Shoppers: Инженерия признаков

**Автор:** Девяткова А.А., АСОиУб-23-2
**Дисциплина:** Машинное обучение и ИИ
**Дата:** 2026-11-03

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.87 | 0.58 | 0.74 | 0.69 | 0.94 |
| XGBoost | 0.87 | 0.58 | 0.73 | 0.67 | 0.93 |
| LightGBM | 0.87 | 0.57 | 0.77 | 0.65 | 0.92 |
| CatBoost | 0.87 | 0.55 | 0.79 | 0.66 | 0.93 |

**Лучшая модель:** LightGBM.

## 🎯 Ключевые инсайты (SHAP + Feature Importance)

- **PageValues** — сильнейший предиктор покупки: чем выше ценность страниц, тем выше вероятность конверсии.
- **TotalDuration** — общее время на сайте положительно связано с покупкой.
- **ExitRates / BounceRates** — высокие значения снижают вероятность покупки.
- **ProductRelated** — число товарных страниц положительно влияет на конверсию.
- **IsNewVisitor** — новые посетители покупают реже, чем возвращающиеся.
- **MonthNum** — есть сезонность: пики покупок в отдельные месяцы (ноябрь, декабрь).

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/Devjatkova/ml-course-devyatkova/main/module-4-features"

# Загрузка лучшей модели
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))

# Загрузка списка признаков
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

print(f"Модель загружена, признаков: {len(feature_cols)}")