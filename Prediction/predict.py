import joblib
import pandas as pd


def predict(model, dataset):
    model = joblib.load(model)
    dataset = pd.read_csv(dataset)
    predictions = model.predict(dataset)
    results = pd.Series(data=predictions, index=dataset.index)
    return results


if __name__ == "__main__":
    predict('./model/xgb_model.sav', 'test_data.csv')
