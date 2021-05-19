import pandas as pd
import requests

PATH_TO_DATASET = 'heart.csv'
TARGET_COLUMN = 'target'
ENDPOINT_URL = 'http://0.0.0.0:8000/predict/'


if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_DATASET).drop(columns=TARGET_COLUMN)
    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist()
    }
    response = requests.get(ENDPOINT_URL, json=request_params)
    print(response.status_code)
    print(response.json())
