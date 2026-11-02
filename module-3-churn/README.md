# 📉 Module 3 — Customer Churn: Ансамблевые методы

**Автор:** Девяткова А.А., АСОиУб-23-2
**Дата:** 2026-11-03

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Random Forest | 0.77 | 0.55 | 0.71 | 0.62 | 0.84 | 0.5 сек |
| XGBoost | 0.76 | 0.54 | 0.69 | 0.61 | 0.82 | 0.2 сек |
| LightGBM | 0.75 | 0.52 | 0.74 | 0.61 | 0.83 | 0.1 сек |
| CatBoost | 0.75 | 0.52 | 0.75 | 0.61 | 0.84 | 2.0 сек |
| **LightGBM (Optuna)** | **0.75** | **0.52** | **0.74** | **0.61** | **0.83** | — |

**Лучшая модель:** LightGBM с оптимизацией Optuna (ROC-AUC = 0.84, F1 = 0.83).

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/Devjatkova/ml-course-devyatkova/main/module-3-churn"

model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

# Пример предсказания (признаки в порядке feature_cols)
X_new = [[1, 1, 0, 1, 34, 1, 0, 0, 1, 0, 0, 0, 0, 1, 56.95, 1889.5, 2.83, 54.0, 0, 1]]
proba = model.predict_proba(X_new)[0, 1]
print(f"Вероятность ухода: {proba:.2%}")