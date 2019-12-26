import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing

data = pd.read_csv('./Data.csv')									############### CSV Data read with pandas

X = data.iloc[:, :-1]												############### Store all columns in the input expect the last which are in the independent vars
Y = data.iloc[:, 3]													############### Store only the last column which is the dependent var

imputer = Imputer(missing_values=np.nan, strategy='mean')			############### Use the Imputer class with imputer object to fill the nan values in few cols with a strategy (here mean)
imputer = imputer.fit(X.iloc[:, 1:3])								############### Fit the imputer opject to the X matrix
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])					############### Transform the X with the mean values from the imputer object

# Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
columntransformer = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
X = columntransformer.fit_transform(X)

#encoding the y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Dividing the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.2, random_state = 0)

# print("X_train: {}\tY_train: {}\nX_test: {}\tY_test: {}".format(X_train, Y_train, X_test, Y_test))

# Feature scaling the features (The output categories are not scaled because here they are categorical data and not a range of data values)
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print("X_train: {}\tY_train: {}\nX_test: {}\tY_test: {}".format(X_train, Y_train, X_test, Y_test))