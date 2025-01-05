
# prompt: load the parkinsons data set into a pandas f

import pandas as pd

df = pd.read_csv('/content/parkinsons.csv')

features = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']
target = 'status'

x = df[features]
y = df[target]


# prompt: i want to scale my data using minmax scaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# prompt: i want to split my data into train and test

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)



# prompt: i want to train the GridSearchCV model

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Create an SVC model
svc = SVC()

# Create GridSearchCV object
grid = GridSearchCV(svc, param_grid, refit=True, verbose=3)

# Fit the GridSearchCV object to the training data
grid.fit(x_train, y_train)

# Print the best parameters and best score
print(grid.best_params_)
print(grid.best_estimator_)

# Make predictions on the test set using the best model found by GridSearchCV
grid_predictions = grid.predict(x_test)
