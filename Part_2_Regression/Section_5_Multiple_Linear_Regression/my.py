# Multiple Linear Regression

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Dummy encode the State column (4th column)
column_transformer = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [3])],
         remainder = 'passthrough')
X = np.array(column_transformer.fit_transform(X))


# Avoiding the dummy variable trap
# Remove the first column of X
# Which is now the first state dummy variable
# NOTE: Not REQUIRED to be done manually as the library takes care of it
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_prediction = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Add a column of 1's at the front (represent the x0 in the regression equation)
# y = b0*x0 + b1*x1 + ... + bn*xn
# This is needed for the statsmodels library. Otherwise it will treat this as
# y = b1*x1 + ... + bn*xn
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
# X = sm.add_constant(X)

# Initialise this as the matrix containing all columns of independent variables

p_value = 1.0
significance_level = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
while (p_value > significance_level):
    # OLS = Ordinary Least squares model
    backwards_regressor_OLS = sm.OLS(y.astype(float), X_optimal.astype(float)).fit()
    p_value = np.amax(backwards_regressor_OLS.pvalues)
    highest_pval_index = np.where(backwards_regressor_OLS.pvalues == p_value)
    X_optimal = np.delete(X_optimal, highest_pval_index[0][0], axis=1)

print(backwards_regressor_OLS.summary())


