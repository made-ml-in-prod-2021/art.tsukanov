import logging
import logging.config
import json
import pickle

import hydra
from omegaconf import OmegaConf
import yaml

from heart_classification.entities import TrainingPipelineParams
from heart_classification.data import (
    read_data,
    split_data,
)
from heart_classification.features import (
    build_transformer,
    make_features,
    extract_target,
)
from heart_classification.models import (
    train_model,
    predict_model,
    evaluate_model,
)

LOGGING_CONFIG_PATH = '../configs/logging_config.yaml'


def training_pipeline(training_pipeline_params: TrainingPipelineParams):
    """
    * Read data
    * Split data into train and validation parts
    * Build transformer
    * Extract features
    * Extract targets
    * Train and serialize model
    * Predict and evaluate model
    """
    input_df = read_data(training_pipeline_params.input_data_path)
    logging.debug(f'input dataset shape: {input_df.shape}')

    train_df, val_df = split_data(input_df, training_pipeline_params.splitting_params)
    logging.debug(f'train dataset shape: {train_df.shape}')
    logging.debug(f'validation dataset shape: {val_df.shape}')

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    train_features = make_features(train_df, transformer)
    val_features = make_features(val_df, transformer)
    logging.debug(f'train features shape: {train_features.shape}')
    logging.debug(f'validation feature shape: {val_features.shape}')

    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    model = train_model(train_features, train_target, training_pipeline_params.training_params)
    with open(training_pipeline_params.output_model_path, 'wb') as fout:
        pickle.dump(model, fout)

    predicts = predict_model(model, val_features)
    metrics = evaluate_model(predicts, val_target)
    logging.info(f'metrics: {metrics}')
    with open(training_pipeline_params.metric_path, 'w') as fout:
        json.dump(metrics, fout)


@hydra.main(config_path='../configs', config_name='train_config')
def training_pipeline_command(config: TrainingPipelineParams):
    """Run train pipeline."""
    setup_logging(config.logging_config_path)
    logging.info(f'train pipeline started with params:\n{{\n{OmegaConf.to_yaml(config)}}}')
    training_pipeline(config)
    logging.info('train pipeline finished successfully')


def setup_logging(logging_config_path):
    """Set logging config."""
    with open(logging_config_path, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))


if __name__ == '__main__':
    training_pipeline_command()  # pylint: disable=E1120
