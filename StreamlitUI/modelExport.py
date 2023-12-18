import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
dataFilePath = os.path.join(script_dir, 'warsaw.csv')

# 1. Load the data.
dataset = pd.read_csv(dataFilePath, sep=",")


dataset['DATE'] = pd.to_datetime(dataset['DATE'])
dataset['Year'] = dataset['DATE'].dt.year
dataset['Month'] = dataset['DATE'].dt.month
dataset['Day'] = dataset['DATE'].dt.day


features = ['Year', 'Month', 'Day', 'PRCP', 'SNWD']
target = 'TAVG'

X = dataset[features]
y = dataset[target]

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns

# Exclude non-numeric columns from imputation
numeric_features = list(set(features) - set(non_numeric_cols))

# Define transformers for numeric and non-numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', 'passthrough', non_numeric_cols)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Create and train the kNN model with preprocessing pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=4))
])

model.fit(X_train, y_train)

with open('knn_model.pkl', 'wb') as file:
    pickle.dump(model, file)
