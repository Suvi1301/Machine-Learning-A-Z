# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # indexes from 3 to 12
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable - Converting Country and Gender into numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
# Create dummy variables
# For the gender encoded data, we don't need dummy variable because its only 0 and 1.
# So we just use one column to avoid dummy variable trap.
# Whereas, for Country, we use 2 of the 3 dummy variable to avoid the DV trap.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - is required because we don't want one independent variable dominating another.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now lets make the Artificial Neural Network
# ANN relevant libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
# This classifier object is the Neural Net
classifier = Sequential()

# Adding the input and first hidden layer
# output_dim: no. of nodes in the hidden layer (avg of input and output nodes)
# init: initalises the weights "uniformly" randomly close to zero.
# activation: relu is the rectifier activation function
# input_dim: tells the network how many input nodes there are for that layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
# output_dim: no. of classes for the dependent variable
# activation: 'sigmoid' for 2 classes. If more than 2 classes: use 'soft'
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN - applying Stochastic Gradient Descent on the ANN
# optimizer: 'adam' is an efficient stochastic gradient descent algorithm
# loss: use the Logarithmic Loss function for sigmoid activation function
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_predict = classifier.predict(X_test)
y_predict = (y_predict > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_predict)
print(f'Confusion Matrix: {confusionMatrix}')
accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1]) / len(y_test)
print(f'Accuracy of test prediction: {accuracy*100}%')