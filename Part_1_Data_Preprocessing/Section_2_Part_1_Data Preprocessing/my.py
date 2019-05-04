# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # All columns except last
y = dataset.iloc[:, 3].values # Last (3rd) column


# Handling missing data - Take mean of the column approach
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3]) # upper bound is excluded hence, 1:3 for column
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# =============================================================================
# # We can convert the labels to integers first and then use OneHotEncoder.
# # But we use ColumnTransformer instead (other shit is deprecated)
# labelencoder_X = LabelEncoder()
# # Fit the Country column to encode it. We lose the original values ofc.
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) 
# one_hot_encoder = OneHotEncoder(categorical_features = [0])
# # Dont need to specify index transform since we already defined the index in categorical_features
# X = one_hot_encoder.fit_transform(X).toarray()
# =============================================================================


# DUMMY Encoding
from sklearn.compose import ColumnTransformer

# Dummy encode the Country column
column_transformer = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],
         remainder = 'passthrough')
X = np.array(column_transformer.fit_transform(X))

# Here we encode the 'Purchased' column.
# We dont need OneHotEncoder because its a dependent variable.
# The values will be unique
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
# User model_selection library instead of cross_validation (deprecated)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""