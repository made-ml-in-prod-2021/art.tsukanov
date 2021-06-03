import os
import logging
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sklearn.linear_model import LogisticRegression

DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
TRAIN_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

logger = logging.getLogger(__name__)
app = FastAPI()
model: Optional[LogisticRegression] = None


def load_model(path: str) -> LogisticRegression:
    with open(path, 'rb') as fin:
        return pickle.load(fin)


class HeartDiseaseModel(BaseModel):
    data: List[List]
    features: List[str]


class HeartDiseaseResponse(BaseModel):
    disease: List[int]


def make_prediction(
    model: LogisticRegression,
    data: List[List[Union[int, float]]],
    features: List[str]
) -> List[int]:
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    return predicts.tolist()


@app.on_event('startup')
def startup():
    global model
    model_path = os.getenv('PATH_TO_MODEL')
    if model_path is None:
        err = 'PATH_TO_MODEL is None'
        logger.error(err)
        raise RuntimeError(err)
    model = load_model(model_path)


@app.get('/')
def root():
    return "This is an entry point of Heart Disease classifier. " \
           "To make prediction use '/predict/' endpoint."


def validate_features_and_data(request_model: HeartDiseaseModel):
    if len(request_model.features) != len(TRAIN_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f'Number of features should be equal to {len(TRAIN_FEATURES)} '
                   f'as well as in the following list: {TRAIN_FEATURES}'
        )
    if not any(isinstance(x, (int, float)) for row in request_model.data for x in row):
        raise HTTPException(
            status_code=400,
            detail="All elements in 'data' should be either int or float"
        )


@app.get('/predict/', response_model=HeartDiseaseResponse)
def predict(request_model: HeartDiseaseModel):
    validate_features_and_data(request_model)
    predicts = make_prediction(model, request_model.data, request_model.features)
    return {'disease': predicts}


if __name__ == '__main__':
    uvicorn.run('app:app', host=DEFAULT_HOST, port=os.getenv('PORT', DEFAULT_PORT))
