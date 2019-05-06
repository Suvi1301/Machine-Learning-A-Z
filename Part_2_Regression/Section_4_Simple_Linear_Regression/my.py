# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X_experience = dataset.iloc[:, :-1].values
y_salary = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_experience, y_salary, test_size = 1/3, random_state = 0)

# Feature scaling will be taken care by the LinearRegression class
# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# This is where the library "learns" the correlation and produces the line of regression
regressor.fit(X_train, y_train)

# Predicting the Test set results
''' The predictor method will use the regression fit
    to make predictions on the provided data set '''
y_prediction = regressor.predict(X_test)

# Visualise the Training, Testing and predicted results set results 
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary v Experience')
plt.xlabel('Experience (years)')
plt.ylabel('Salary ($)')
plt.show()