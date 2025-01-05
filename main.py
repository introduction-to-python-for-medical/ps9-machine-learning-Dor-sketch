import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml

# Load your dataset
data = pd.read_csv('parkinsons.csv')

# Selecting only two relevant features and the target
features = ['HNR', 'RPDE']
output = 'status'  # Assuming 'status' is the categorical target for classification

X = data[features]
y = data[output]

# Scaling the features to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building the SVM classification model with tuned parameters and balanced class weight
svm_model = SVC(kernel='rbf', C=10.0, gamma=0.1, class_weight='balanced')  # Tuned parameters

# Training the model
svm_model.fit(X_train, y_train)

# Making predictions
y_pred = svm_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
model_filename = 'svc_model_2features.joblib'
joblib.dump(svm_model, model_filename)
print(f"Model saved as {model_filename}")

# Create and save the config.yaml file
config = {
    'features': features,
    'path': "../" + model_filename  # Adjusted for relative path
}

with open('config.yaml', 'w') as config_file:
    yaml.dump(config, config_file)
print("Configuration file 'config.yaml' created.")
