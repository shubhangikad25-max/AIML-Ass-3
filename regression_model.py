# ==========================================================
# PART A: HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 1. Create dataset
data = pd.DataFrame({
    "area": [
        800, 900, 1000, 1100, 1200,
        1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2200, 2400,
        2600, 2800, 3000, 3200, 3500
    ],

    "bedrooms": [
        1, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        3, 4, 4, 4, 4,
        4, 5, 5, 5, 6
    ],

    "bathrooms": [
        1, 1, 1, 2, 2,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        3, 4, 4, 5, 5
    ],

    "price": [
        40, 45, 50, 58, 65,
        72, 78, 85, 92, 100,
        108, 118, 128, 142, 155,
        170, 185, 205, 225, 250
    ]
})


print("HOUSE PRICE DATASET")
print(data)


# 2. Select features and target
X = data[["area", "bedrooms", "bathrooms"]]
y = data["price"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create model
model = LinearRegression()


# 5. Train model
model.fit(X_train, y_train)


# 6. Prediction
y_pred = model.predict(X_test)


# 7. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\n===== MODEL PERFORMANCE =====")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))


# 8. Actual vs Predicted
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": np.round(y_pred, 2)
})

print("\n===== ACTUAL VS PREDICTED =====")
print(results)


# 9. Graph
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.show()
