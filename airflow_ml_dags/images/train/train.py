import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(input_dir: str, models_dir: str, random_state: int):
    data = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    target = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    model = LogisticRegression(random_state=random_state)  # should not be hardcoded
    model.fit(data, target)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "logreg.pth"), "wb") as fout:
        pickle.dump(model, fout)


@click.command("train_model")
@click.option("--input-dir")
@click.option("--models-dir")
@click.option("--random-state", type=int)
def main(input_dir: str, models_dir: str, random_state: int):
    train_model(input_dir, models_dir, random_state)


if __name__ == '__main__':
    main()
