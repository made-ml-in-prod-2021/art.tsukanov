import os
import pickle
import json

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def validate_model(input_dir: str, models_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "X_val.csv"))
    target = pd.read_csv(os.path.join(input_dir, "y_val.csv"))

    with open(os.path.join(models_dir, "logreg.pth"), "rb") as fin:
        model = pickle.load(fin)

    preds = model.predict(data)
    metrics = {
        "accuracy": accuracy_score(preds, target)
    }

    with open(os.path.join(models_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout)


@click.command("validate_model")
@click.option("--input-dir")
@click.option("--models-dir")
def main(input_dir: str, models_dir: str):
    validate_model(input_dir, models_dir)


if __name__ == '__main__':
    main()
