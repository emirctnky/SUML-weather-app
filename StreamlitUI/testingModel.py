import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pylint.lint import Run


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
dataFilePath = os.path.join(script_dir, 'warsaw.csv')

# 1. Load the data
dataframe = pd.read_csv(dataFilePath, sep=",")

# 2. Fill (or remove) all missing data
dataset_preprocessed = dataframe.fillna(dataframe.mean(numeric_only=True))

# splitting plain date into columns for the kNN model
dataset_preprocessed['DATE'] = pd.to_datetime(dataset_preprocessed['DATE'])
dataset_preprocessed['Year'] = dataset_preprocessed['DATE'].dt.year
dataset_preprocessed['Month'] = dataset_preprocessed['DATE'].dt.month
dataset_preprocessed['Day'] = dataset_preprocessed['DATE'].dt.day

dataset_preprocessed = dataset_preprocessed.drop(['DATE'], axis=1)

# 3. Select the features and target variable
features = ['LATITUDE', 'LONGITUDE', 'ELEVATION',
            'Year', 'Month', 'Day', 'PRCP', 'SNWD']
target = 'TAVG'

X = dataset_preprocessed[features]
y = dataset_preprocessed[target]


# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=101)


# looking for the best "k" for the model

# Initialize a list to store the MSE values for different k
mse_values = []

# Get the MSE for K values between 1 and 100
for k in range(1, 100):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Plot the MSE values to choose the best K
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), mse_values, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('MSE for Different K Values')
plt.xlabel('K Value')
plt.ylabel('Mean Squared Error')

plt.show()

# Find the K value with the lowest MSE
best_k = mse_values.index(min(mse_values)) + 1
print(f"Minimum MSE: {min(mse_values)} at K = {best_k}")


scriptName = "testingModel.py"
# Pylint results
results = Run([scriptName])
print(results.linter.stats.global_note)
