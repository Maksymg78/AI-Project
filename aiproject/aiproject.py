import ipywidgets as widgets
import lightgbm as lgb
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("1. Генерація просторово-часових даних для всіх областей України...")

# Список усіх регіонів України
regions = [
    "м. Київ",
    "Київська",
    "Харківська",
    "Одеська",
    "Дніпропетровська",
    "Запорізька",
    "Львівська",
    "Миколаївська",
    "Полтавська",
    "Чернігівська",
    "Сумська",
    "Житомирська",
    "Черкаська",
    "Вінницька",
    "Кіровоградська",
    "Хмельницька",
    "Рівненська",
    "Волинська",
    "Івано-Франківська",
    "Тернопільська",
    "Чернівецька",
    "Закарпатська",
    "Херсонська",
    "Донецька",
    "Луганська",
    "АР Крим",
    "м. Севастополь",
]

# Генерація погодинного часового ряду
start_date = "2026-01-01"
end_date = pd.Timestamp.now()
dates = pd.date_range(start=start_date, end=end_date, freq="1h")

# Створення матриці даних для всіх регіонів
np.random.seed(42)
data_dict = {
    "datetime": np.tile(dates, len(regions)),
    "region": np.repeat(regions, len(dates)),
    "is_alert": np.random.binomial(1, 0.15, len(dates) * len(regions)),
}

df_all = pd.DataFrame(data_dict)
print(f"Загальна кількість рядків у матриці (регіони x години): {len(df_all)}")

# ===================================================================
# 2. Інженерія просторово-часових ознак
# ===================================================================
print("2. Створення ознак та крос-регіональних лагів...")

df_all["hour"] = df_all["datetime"].dt.hour
df_all["dayofweek"] = df_all["datetime"].dt.dayofweek
df_all["hour_sin"] = np.sin(2 * np.pi * df_all["hour"] / 24)
df_all["hour_cos"] = np.cos(2 * np.pi * df_all["hour"] / 24)

df_all["alert_lag1"] = df_all.groupby("region")["is_alert"].shift(1)
df_all["alert_lag2"] = df_all.groupby("region")["is_alert"].shift(2)
df_all["rolling_6h"] = (
    df_all.groupby("region")["is_alert"]
    .shift(1)
    .rolling(6)
    .sum()
    .reset_index(0, drop=True)
)

df_all["target_next_hour"] = df_all.groupby("region")["is_alert"].shift(-1)
df_clean = df_all.dropna().copy()
df_clean["region_code"] = df_clean["region"].astype("category")

# ===================================================================
# 3. Навчання єдиної мультирегіональної моделі
# ===================================================================
print("3. Навчання універсальної моделі для всіх областей...")

features = [
    "region_code",
    "hour_sin",
    "hour_cos",
    "alert_lag1",
    "alert_lag2",
    "rolling_6h",
]

X = df_clean[features]
y = df_clean["target_next_hour"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
print(
    f"Мультирегіональну модель успішно навчено! ROC-AUC:"
    f" {roc_auc_score(y_test, y_pred_proba):.3f}\n"
)

# ===================================================================
# 4. Інтерактивний вибір регіону через випадаючий список (Dropdown)
# ===================================================================
print(
    "4. Модель готова! Оберіть регіон у випадаючому списку нижче та натисніть"
    " кнопку:"
)

# Створюємо віджет випадаючого списку
region_dropdown = widgets.Dropdown(
    options=regions,
    value="Київська",
    description="Регіон:",
    style={"description_width": "initial"},
)

# Створюємо кнопку виконання
button = widgets.Button(
    description="Розрахувати прогноз",
    button_style="success",  # зелена кнопка
    icon="check",
)

# Поле для виведення результату
output = widgets.Output()


def on_button_clicked(b):
  with output:
    output.clear_output()
    target_region = region_dropdown.value

    # Шукаємо дані для обраного регіону
    region_data = df_clean[df_clean["region"] == target_region]

    if not region_data.empty:
      last_sample = region_data.iloc[[-1]][features]
      predicted_prob = model.predict_proba(last_sample)[0][1]
      print(f"--- РЕЗУЛЬТАТ ПРОГНОЗУ ДЛЯ ОБЛАСТІ: {target_region} ---")
      print(
          "Ймовірність тривоги в регіоні"
          f" '{target_region}' на наступну годину: **{predicted_prob * 100:.1f}%**"
      )
    else:
      print("Не вдалося знайти дані для цього регіону.")


# З'єднуємо кнопку з функцією обробки
button.on_click(on_button_clicked)

# Відображаємо елементи інтерфейсу в блокноті
display(region_dropdown, button, output)
