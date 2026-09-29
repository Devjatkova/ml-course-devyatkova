# Module 1 — Titanic: EDA + бинарная классификация

**Автор:** Девяткова Анастасия Александровна
**Группа:** АСОиУб-23-2
**Дата:** 2026-09-29
**Дисциплина:** Машинное обучение и ИИ

---

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время обучения |
|--------|----------|-----------|--------|-----|---------|----------------|
| Logistic Regression | 0.8045 | 0.7833 | 0.6812 | 0.7287 | **0.8486** | ~0.07 сек |
| Decision Tree | 0.7821 | 0.7419 | 0.6667 | 0.7023 | 0.8132 | ~0.03 сек |

**Лучшая модель:** Logistic Regression (ROC-AUC = 0.8486).

**Ключевой вывод:** Пол (`Sex`) — самый сильный предиктор выживания. Женщины выживали в 74% случаев, мужчины — только в 19%.

---

## 📁 Структура проекта
module-1-titanic/
├── README.md # Этот файл
├── notebook.ipynb # Полный отчёт (14 разделов)
├── requirements.txt # Зависимости проекта
├── data/
│ ├── train.csv # Обучающая выборка (891 пассажир)
│ ├── test.csv # Тестовая выборка (418 пассажиров)
│ ├── gender_submission.csv
│ └── titanic_info.md # Подробное описание датасета
├── models/
│ ├── lr_model.pkl # Обученная Logistic Regression
│ ├── dt_model.pkl # Обученный Decision Tree
│ ├── scaler.pkl # StandardScaler (для LR)
│ ├── le_sex.pkl # LabelEncoder для Sex
│ ├── feature_cols.json # Список признаков в правильном порядке
│ └── metrics.json # Метрики обеих моделей
└── examples/
├── eda_plots.png # Визуализации из EDA
├── roc_curves.png # ROC-кривые моделей
└── confusion_matrices.png # Матрицы ошибок


## 🛠️ Обработка данных

- **`Age`** (19.87% пропусков) — заполнен медианой по группам `Pclass + Sex`.
- **`Embarked`** (0.22% пропусков) — заполнен модой (`'S'`).
- **`Cabin`** (77.10% пропусков) — **удалён** (восстановить данные невозможно).
- **`Fare`** (0.22% пропусков в test) — заполнен медианой из train.

### Feature Engineering

- `Family_Size = SibSp + Parch + 1` — размер семьи.
- `Is_Alone = (Family_Size == 1)` — бинарный признак одиночества.
- `Age_Group` — биннинг через `pd.cut()` (использовался в EDA).

### Кодирование

- `Sex` → **Label Encoding** (`female` → 1, `male` → 0).
- `Embarked` → **One-Hot Encoding** (`drop_first=True`, столбцы `Embarked_Q`, `Embarked_S`).

### Финальный список признаков

```python
feature_cols = [
    'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch',
    'Fare', 'Family_Size', 'Is_Alone', 'Embarked_Q', 'Embarked_S'
]

import joblib
import requests
from io import BytesIO
import json

BASE_URL = "https://raw.githubusercontent.com/Devjatkova/ml-course-devyatkova/main/module-1-titanic"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
feature_cols = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

print(f" Модель загружена. Признаков: {len(feature_cols)}")
