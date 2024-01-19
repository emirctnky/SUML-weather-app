import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from pylint.lint import Run


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
dataFilePath = os.path.join(script_dir, 'warsaw.csv')

# 1. Load the data
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

# Exclude non-numeric columns
numeric_features = list(set(features) - set(non_numeric_cols))

# Define transformers for features
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=101)

# Create and train the RandomForest model with preprocessing pipeline
n_estimators = 156

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=101))
])

model.fit(X_train, y_train)

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)


scriptName = "modelExport.py"
# Pylint results
results = Run([scriptName])
print(results.linter.stats.global_note)
