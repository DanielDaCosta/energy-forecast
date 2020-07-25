import joblib
import pandas as pd


def predict(model, dataset):
    model = joblib.load(model)
    dataset = pd.read_csv(dataset)
    predictions = model.predict(dataset)
    return predictions


if __name__ == "__main__":
    predict('./model/xgb_model.sav', 'test_data.csv')
