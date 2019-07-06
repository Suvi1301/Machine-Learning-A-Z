# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
# rbf kernel: Gaussian (had better prediction)
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predict = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_predict)
import ipdb; ipdb.set_trace()
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_train_set, y_train_set, X_test_set, y_test_set = X_train, y_train, X_test, y_test

# Create the grid using the data set. Using 0.01 pixel steps.
X1, X2 = np.meshgrid(np.arange(start = X_train_set[:, 0].min() - 1, stop = X_train_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_train_set[:, 1].min() - 1, stop = X_train_set[:, 1].max() + 1, step = 0.01))

# Use the prediction on the training set to generate a classification boundary line. Creates the classifcation areas
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())

# Plot the training set
for i, j in enumerate(np.unique(y_train_set)):
    plt.scatter(X_train_set[y_train_set == j, 0], X_train_set[y_train_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = 'train %s' % j)

# Plot the testing set
for i, j in enumerate(np.unique(y_test_set)):
    plt.scatter(X_test_set[y_test_set == j, 0], X_test_set[y_test_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = 'test %s' % j,
                marker = 'x')

plt.title('Support Vector Machine (Training (o) and Testing (x) Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()