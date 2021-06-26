import os

import click
import pandas as pd

NUM_COLS = 4


def process_data(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    assert NUM_COLS == len(data.columns), f"number of columns should be {NUM_COLS}, not {len(data.columns)}"
    # some preprocessing (no need for iris dataset)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


@click.command("process_data")
@click.option("--input-dir")
@click.option("--output-dir")
def main(input_dir: str, output_dir: str):
    process_data(input_dir, output_dir)


if __name__ == '__main__':
    main()
