# ==========================================================
# PART B: GRADIENT DESCENT FOR LINEAR REGRESSION
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. House area and price
X = np.array([
    8, 9, 10, 11, 12,
    13, 14, 15, 16, 17,
    18, 19, 20, 22, 24,
    26, 28, 30, 32, 35
], dtype=float)

y = np.array([
    40, 45, 50, 58, 65,
    72, 78, 85, 92, 100,
    108, 118, 128, 142, 155,
    170, 185, 205, 225, 250
], dtype=float)


# 2. Standardize input
X_mean = np.mean(X)
X_std = np.std(X)

X_scaled = (X - X_mean) / X_std


# 3. Initialize parameters
w = 0.0
b = 0.0


# 4. Hyperparameters
learning_rate = 0.01
iterations = 1000


# 5. Store cost
cost_history = []

n = len(X_scaled)


# 6. Gradient Descent
for i in range(iterations):

    # Prediction
    y_pred = w * X_scaled + b

    # Error
    error = y_pred - y

    # Cost
    cost = np.mean(error ** 2)

    cost_history.append(cost)

    # Gradients
    dw = (2 / n) * np.sum(X_scaled * error)

    db = (2 / n) * np.sum(error)

    # Update parameters
    w = w - learning_rate * dw

    b = b - learning_rate * db


# 7. Final prediction
y_pred = w * X_scaled + b


# 8. Performance metrics
mae = np.mean(np.abs(y - y_pred))

mse = np.mean((y - y_pred) ** 2)

rmse = np.sqrt(mse)

ss_total = np.sum((y - np.mean(y)) ** 2)

ss_residual = np.sum((y - y_pred) ** 2)

r2 = 1 - (ss_residual / ss_total)


# 9. Display results
print("===== GRADIENT DESCENT RESULTS =====")

print("Learning Rate :", learning_rate)
print("Iterations    :", iterations)

print("Final Weight  :", round(w, 4))
print("Final Bias    :", round(b, 4))

print("Final Cost    :", round(cost_history[-1], 4))

print("\n===== PERFORMANCE METRICS =====")

print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R2   :", round(r2, 4))


# 10. Actual vs Predicted
results = pd.DataFrame({
    "Area": X,
    "Actual Price": y,
    "Predicted Price": np.round(y_pred, 2)
})

print("\n===== ACTUAL VS PREDICTED =====")

print(results)


# 11. Cost vs Iterations
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.title("Gradient Descent - Cost vs Iterations")

plt.show()


# 12. Regression graph
plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    y_pred,
    label="Gradient Descent Prediction"
)

plt.xlabel("House Area (100 sq.ft)")

plt.ylabel("House Price (Lakhs)")

plt.title("Linear Regression using Gradient Descent")

plt.legend()

plt.show()
