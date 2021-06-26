import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8


def split_data(input_dir: str, output_dir: str, random_state: int):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_val, y_train, y_val = train_test_split(
        data, target, train_size=TRAIN_SIZE, random_state=random_state
    )

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)


@click.command("slit_data")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--random-state", type=int)
def main(input_dir: str, output_dir: str, random_state: int):
    split_data(input_dir, output_dir, random_state)


if __name__ == '__main__':
    main()
