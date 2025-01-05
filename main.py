import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml

# Load your dataset
data = pd.read_csv('parkinsons.csv')

# Selecting relevant features based on the paper
# Using the provided column names
features = ['HNR', 'RPDE']
output = 'PPE'

X = data[features]
y = data[output]

# Scaling the features to [-1, 1] as suggested in the paper
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Building the SVM regression model with a radial basis function (RBF) kernel
svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')

# Training the model
svm_model.fit(X_train, y_train.ravel())

# Making predictions
y_pred = svm_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Save the trained model to a file
model_filename = 'svm_model.joblib'
joblib.dump(svm_model, model_filename)
print(f"Model saved as {model_filename}")

# Create and save the config.yaml file
config = {
    'features': features,
    'path': model_filename
}

with open('config.yaml', 'w') as config_file:
    yaml.dump(config, config_file)
print("Configuration file 'config.yaml' created.")
