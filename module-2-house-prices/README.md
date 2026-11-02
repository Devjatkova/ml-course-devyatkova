# 🏠 Module 2 — House Prices: предобработка и регрессия

**Автор:** Девяткова А.А., АСОиУб-23-2
**Дата:** 2026-11-03

## 📊 Результаты

| Модель | RMSE (log) | MAE (log) | R² | RMSE ($) |
|---|---|---|---|---|
| Linear Regression | 0.138 | 0.094 | 0.885 | $22 000 |
| Ridge | 0.132 | 0.091 | 0.895 | $21 600 |
| Lasso | 0.128 | 0.088 | 0.902 | $21 000 |
| ElasticNet | 0.128 | 0.088 | 0.902 | $21 000 |

**Лучшая модель:** Lasso (R² = 0.902, RMSE = $21 000).

**Время обучения:** менее 0.99 сек на каждую модель.

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO
import json

BASE_URL = "https://raw.githubusercontent.com/Devjatkova/ml-course-devyatkova/main/module-2-house-prices"

# Загрузка модели
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lasso_model.pkl").content))

# Загрузка scaler
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))

# Загрузка списка признаков
feature_cols = json.loads(requests.get(f"{BASE_URL}/models/feature_cols.json").content)

print(f"Загружено {len(feature_cols)} признаков")