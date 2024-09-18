import yaml
from model.ModelPipeline import ModelPipeline


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path: str):
    config = load_config(config_path)

    model_pipeline = ModelPipeline(config)
    model_pipeline.load_data()
    model_pipeline.get_preprocessor()
    model_pipeline.build_pipeline()
    model_pipeline.train_model()
    model_pipeline.evaluate_model()


if __name__ == "__main__":
    main("config.yaml")
