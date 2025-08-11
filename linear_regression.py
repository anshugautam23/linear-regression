# Linear Regression Example in Python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (Square footage vs Price)
X = np.array([650, 800, 1000, 1200, 1500, 1800, 2000, 2300, 2500]).reshape(-1, 1)
y = np.array([70000, 85000, 100000, 120000, 150000, 180000, 200000, 230000, 250000])

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict prices
y_pred = model.predict(X)

# Print results
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Prediction for 2100 sqft:", model.predict([[2100]])[0])

# Plot the results
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Best Fit Line")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()
