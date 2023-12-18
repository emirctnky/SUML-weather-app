import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
dataFilePath = os.path.join(script_dir, 'warsaw.csv')

# 1. Load the data.
dataframe = pd.read_csv(dataFilePath, sep=",")

# 2. Fill (or remove) all missing data.
#  df2_removed = dataframe.dropna()
dataset_preprocessed = dataframe.fillna(dataframe.mean(numeric_only=True))

# changing the "DATE" field into columns with numerical values so kNN can work with them
dataset_preprocessed['DATE'] = pd.to_datetime(dataset_preprocessed['DATE'])
dataset_preprocessed['Year'] = dataset_preprocessed['DATE'].dt.year
dataset_preprocessed['Month'] = dataset_preprocessed['DATE'].dt.month
dataset_preprocessed['Day'] = dataset_preprocessed['DATE'].dt.day

dataset_preprocessed = dataset_preprocessed.drop(['DATE'], axis=1)

# 3. Select your features and target variable.
# Here, you need to replace 'YourFeatures' and 'YourTarget' with the names of your features and target variable.
# Select your features and target variable.
features = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'Year', 'Month', 'Day', 'PRCP', 'SNWD']
target = 'TAVG'

X = dataset_preprocessed[features]
y = dataset_preprocessed[target]


# 4. Split the data (test size = 0.1; random state = 101).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# 5. Create and train the kNN model.
model = KNeighborsRegressor(n_neighbors=3)  # Number of nearest neighbours considered
model.fit(X_train, y_train)

# 6. Make predictions.
y_pred = model.predict(X_test)

# 7. Print the classification report.
# print(classification_report(y_test, y_pred)) # can't use classification for regression

# min_temp = dataset_preprocessed['TAVG'].min()
# max_temp = dataset_preprocessed['TAVG'].max()
#
# print('Min Temperature:', min_temp)
# print('Max Temperature:', max_temp)

# 7. Print the Mean Squared Error.
# MSE and RMSE are kinda the same, RMSE is just MSE^1/2(square root). It means that the model was on average 3.67 units away
# from the "target"(correct prediction). RMSE squares errors before calculations meaning that larger errors have bigger
# impact on the score
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)

# This is like RMSE but this calculates the ABSOLUTE error, without considering how large each error was and assigning
# weights based on that
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)


# A measure of how much the underlying model explains the input variables, not sure how to understand it yet
r2 = r2_score(y_test, y_pred)
print('R^2 Score:', r2)




# looking for the best "k" for the model

# Initialize a list to store the MSE values for different k
mse_values = []

# Calculate MSE for K values between 1 and 30
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








