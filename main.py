
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



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVC model
model = SVC(kernel='linear', C=1) # You can experiment with different kernels and C values

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



