
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


# prompt: i want to train the GridSearchCV model, with these parameters SVC(C=10, gamma=0.1, kernel='rbf')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Define the parameter grid
param_grid = {'C': [10], 'gamma': [0.1], 'kernel': ['rbf']}

# Create the SVC model
svc = SVC()

# Create the GridSearchCV object
grid_search = GridSearchCV(svc, param_grid, cv=3) # cv is the number of cross-validation folds

# Train the model
grid_search.fit(x_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


