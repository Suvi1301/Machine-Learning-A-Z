# Polynomial Regression

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X is just the 'level' as its equivalent to the position
# 1:2 to make sure X is a "Matrix" not a vector.
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values # Salary

# NOTE: Wont split into training and testing set as dataset is quite small

# Feature scaling will be done by the linear regression library itself

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
# X_poly will end up having the column of 1's (b0) automatically
# We are essentially getting the x1 and x1^2 terms here
X_poly = poly_regressor.fit_transform(X)

# To include the polynomial fit into our multiple linear regression model
lin_regressor2= LinearRegression()
lin_regressor2.fit(X_poly, y)

# Visualising the Linear Regression results
# To plot more points in the range
X_grid = np.arange(min(X), max(X), 0.1)
# Make it a matrix
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_regressor.predict(X_grid), color='blue', label='Linear')
plt.title("Salary Truth or bluff")
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')

# Visualising the Polynomial Regression results
plt.plot(X_grid, lin_regressor2.predict(poly_regressor.fit_transform(X_grid)),
        color='purple', label='Polynomial (degree 4)')
plt.legend()
# plt.show()

# Predicting a new result with the Linear Regression
print("Predicted salary for Level=6.5 with Linear Regression:",
      lin_regressor.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print("Predicted salary for Level=6.5 with Linear Regression:", 
      lin_regressor2.predict(poly_regressor.fit_transform([[6.5]])))
