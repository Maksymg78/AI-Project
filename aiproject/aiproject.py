import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Генерація демонстраційних даних (симуляція часового ряду)
# ------------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-06-01", freq="1h")
df = pd.DataFrame({"datetime": dates})

# Симулюємо статус тривоги (1 - є тривога, 0 - немає) для Києва та сусідніх областей
# Враховуємо географічну кореляцію: якщо тривога в сусідній області, зростає шанс у Києві
df["alert_kyiv_region"] = np.random.binomial(1, 0.2, len(df))
df["alert_zhytomyr"] = np.random.binomial(1, 0.15, len(df))
df["alert_chernihiv"] = np.random.binomial(1, 0.18, len(df))

# Київська міська тривога сильно залежить від Київської області та сусідів
prob_kyiv = 0.5 * df["alert_kyiv_region"] + 0.2 * df["alert_chernihiv"] + 0.1
df["alert_kyiv"] = np.random.binomial(1, prob_kyiv.clip(0, 1))

# ------------------------------------------------------------------
# 2. Інженерія ознак (Feature Engineering)
# ------------------------------------------------------------------

# А. Циклічні ознаки часу (година доби та день тижня)
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# Б. Лаги (статус тривог у сусідніх областях 1 та 2 години тому)
for col in ["alert_kyiv_region", "alert_zhytomyr", "alert_chernihiv"]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

# В. Інтенсивність (кількість тривог у Києві за останні 6 та 24 години)
df["kyiv_rolling_6h"] = df["alert_kyiv"].shift(1).rolling(window=6).sum()
df["kyiv_rolling_24h"] = df["alert_kyiv"].shift(1).rolling(window=24).sum()

# Г. Цільова змінна (Target): чи буде тривога в Києві у наступну годину (Shift -1)
df["target_kyiv_next_hour"] = df["alert_kyiv"].shift(-1)

# Видаляємо строки з порожніми значеннями (NaN), що виникли через shift/rolling
df_clean = df.dropna().copy()

# ------------------------------------------------------------------
# 3. Підготовка даних та навчання моделі
# ------------------------------------------------------------------

# Вибираємо всі ознаки для навчання
features = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "alert_kyiv_region_lag1",
    "alert_kyiv_region_lag2",
    "alert_zhytomyr_lag1",
    "alert_zhytomyr_lag2",
    "alert_chernihiv_lag1",
    "alert_chernihiv_lag2",
    "kyiv_rolling_6h",
    "kyiv_rolling_24h",
]

X = df_clean[features]
y = df_clean["target_kyiv_next_hour"]

# Важливо для часових рядів: split робимо без перемішування (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Модель градієнтного бустингу
model = LGBMClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 4. Оцінка якості та прогноз ймовірності
# ------------------------------------------------------------------

# Отримуємо ймовірності (predict_proba повертає [ймовірність_0, ймовірність_1])
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Оцінка метрики ROC-AUC (кільки добре модель розділяє класи)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Якість моделі (ROC-AUC Score): {auc_score:.3f}\n")

# Симуляція прогнозу для останнього відомого моменту часу
last_sample = X_test.iloc[[-1]]
last_time = df_clean.iloc[X_test.index[-1]]["datetime"]

predicted_prob = model.predict_proba(last_sample)[0][1]

print("--- Прогноз ймовірності тривоги ---")
print(f"Дата та час аналізу: {last_time}")
print(
    f"Ймовірність оголошення тривоги у Києві в наступну годину: {predicted_prob * 100:.1f}%"
)
