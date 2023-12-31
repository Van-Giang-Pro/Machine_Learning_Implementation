import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Read data from csv file
house_data = pd.read_csv('house_prices.csv')
print(house_data)
size = house_data['sqft_living']
price = house_data['price']
print(size)
print(price)

# Machine learning handle arrays not dataframes
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# We use linear regression model to train our model
model = LinearRegression()
model.fit(x, y)

# MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

# We can get the B value after model fit
# B zero
print(model.coef_[0])
# B one
print(model.intercept_[0])

# Visualize the dataset with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# Predicting the price
print("Prediction by the model: ", model.predict([[2000]]))
