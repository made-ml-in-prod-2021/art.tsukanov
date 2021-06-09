import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


def make_predictions(input_dir: str, output_dir: str, model_path: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    with open(model_path, "rb") as fin:
        model = pickle.load(fin)

    preds = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(preds).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-path")
def main(input_dir: str, output_dir: str, model_path: str):
    make_predictions(input_dir, output_dir, model_path)


if __name__ == "__main__":
    main()
