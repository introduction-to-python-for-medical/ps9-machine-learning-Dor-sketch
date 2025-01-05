import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml

# Load dataset
data = pd.read_csv('parkinsons.csv')

# Based on the paper's findings (Fig 6), using PPE and DFA features
features = ['PPE', 'DFA']  # Changed feature combination
output = 'status'

X = data[features]
y = data[output]

# Scale features to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create SVM model with optimized parameters
svm_model = SVC(
    kernel='rbf',
    C=10.0,  # Increased C for better accuracy
    gamma='auto',
    random_state=42
)

# Train the model
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
model_filename = 'svc_model.joblib'
joblib.dump(svm_model, model_filename)

# Create and save config
config = {
    'features': features,
    'path': model_filename
}

with open('config.yaml', 'w') as file:
    yaml.dump(config, file)