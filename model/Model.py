import yaml
import logging
from .ModelPipeline import ModelPipeline


def load_config(config_file="config_files/modelConfig.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_model(config_path: str):
    logging.info("Initiating Model Setup...")
    logging.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    model_pipeline = ModelPipeline(config)

    logging.info("Loading data...")
    model_pipeline.load_data()

    logging.info("Building model pipeline ...")
    model_pipeline.get_preprocessor()
    model_pipeline.build_pipeline()

    logging.info("Training model...")
    model_pipeline.train_model()

    logging.info("Evaluating model...")
    model_pipeline.evaluate_model()

    logging.info("Model Setup Complete")
    return model_pipeline


if __name__ == "__main__":
    train_model("config_files/modelConfig.yaml")
