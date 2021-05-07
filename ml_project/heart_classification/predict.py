import logging
import logging.config
import pickle

import hydra
from omegaconf import OmegaConf
import yaml

from heart_classification.entities import PredictionParams
from heart_classification.data import read_data
from heart_classification.features import build_transformer, make_features
from heart_classification.models import predict_model

LOGGING_CONFIG_PATH = '../configs/logging_config.yaml'


def predict(prediction_params: PredictionParams):
    """
    * Read data
    * Build transformer
    * Extract features
    * Predict model
    """
    input_df = read_data(prediction_params.data_path)
    logging.debug(f'dataset shape: {input_df.shape}')

    transformer = build_transformer(prediction_params.feature_params)
    transformer.fit(input_df)

    features = make_features(input_df, transformer)
    logging.debug(f'features shape: {features.shape}')

    with open(prediction_params.model_path, 'rb') as fin:
        model = pickle.load(fin)

    predicts = predict_model(model, features)
    predicts.tofile(prediction_params.output_path, sep=',')
    logging.info(f'model output saved to {prediction_params.output_path}')


def setup_logging(logging_config_path):
    """Set logging config."""
    with open(logging_config_path, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))


@hydra.main(config_path='../configs', config_name='predict_config')
def predict_command(config: PredictionParams):
    """Run train pipeline."""
    setup_logging(config.logging_config_path)
    logging.info(f'prediction started with params:\n{{\n{OmegaConf.to_yaml(config)}}}')
    predict(config)
    logging.info('prediction finished successfully')


if __name__ == '__main__':
    predict_command()  # pylint: disable=E1120
