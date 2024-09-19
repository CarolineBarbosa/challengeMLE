import yaml
import logging
from model.ModelPipeline import ModelPipeline
from datetime import datetime

log_filename = datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S.log')

logging.basicConfig(filename=f'log/{log_filename}', 
       level=logging.INFO, 
       format='%(levelname)s %(asctime)s %(message)s', 
       datefmt='%m/%d/%Y %H:%M:%S')


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path: str):
    logging.info('Initiating Model Setup...')
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

    logging.info('Model Setup Complete')
    
    logger = logging.getLogger('Log')


if __name__ == "__main__":
    main("config.yaml")
