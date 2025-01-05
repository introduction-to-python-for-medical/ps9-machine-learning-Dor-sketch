import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml

# Load dataset
data = pd.read_csv('parkinsons.csv')

# Select the best features based on the paper's findings (Figure 6)
features = ['HNR', 'DFA']  # Changed from ['HNR', 'RPDE'] to better features
output = 'status'

X = data[features]
y = data[output]

# Scale features to [-1, 1] as shown in the paper
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Split the dataset (no SMOTE needed as per paper's methodology)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create SVM model with parameters from the paper
svm_model = SVC(
    kernel='rbf',
    C=1.0,  # Default regularization
    gamma='scale',  # Auto-scaling of gamma
    random_state=42,
    probability=True  # Enable probability estimates
)

# Train the model
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_filename = 'svc_model.joblib'
joblib.dump(svm_model, model_filename)

# Create config
config = {
    'features': features,
    'path': model_filename
}

with open('config.yaml', 'w') as file:
    yaml.dump(config, file)