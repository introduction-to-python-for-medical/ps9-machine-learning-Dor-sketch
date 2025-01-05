import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import yaml

def test_model():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    path = config['path']
    features = config['features']
    assert len(features) == 2

    df_test = pd.read_csv(".tests/test.data")
    X = df_test[features]
    y = df_test["status"] # categorial
    model = joblib.load(path)
    prediction = model.predict(X)
    score = accuracy_score(y, prediction)
    assert score > 0.75
