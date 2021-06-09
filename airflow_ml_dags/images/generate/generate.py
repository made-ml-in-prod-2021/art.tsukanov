import os

import click
from sklearn.datasets import load_iris
import numpy as np

SUBSET_SIZE = 0.9


def generate_data(output_dir: str):
    X, y = load_iris(return_X_y=True, as_frame=True)

    indeces = X.index.to_numpy()
    np.random.shuffle(indeces)
    indeces = indeces[:int(len(indeces) * SUBSET_SIZE)]

    os.makedirs(output_dir, exist_ok=True)
    X.iloc[indeces].to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.iloc[indeces].to_csv(os.path.join(output_dir, "target.csv"), index=False)


@click.command("generate_data")
@click.option("--output_dir")
def main(output_dir: str):
    generate_data(output_dir)


if __name__ == "__main__":
    main()
