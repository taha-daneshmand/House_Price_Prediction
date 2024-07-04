import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing(as_frame=True)
data = housing.data
target = housing.target

print(data.head())

X = data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

for i in range(5):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")