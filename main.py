import pandas as pd
import numpy as np
import yaml
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(train_path: str, test_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the train and test data into pandas DataFrames
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def print_metrics(predictions, target):
    print("RMSE: ", np.sqrt(mean_squared_error(predictions, target)))
    print("MAPE: ", mean_absolute_percentage_error(predictions, target))
    print("MAE : ", mean_absolute_error(predictions, target))


def get_preprocessor(categorical_cols: list):
    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_transformer, categorical_cols)]
    )
    return preprocessor


def build_pipeline(preprocessor, model_params: dict):
    steps = [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(**model_params),
        ),
    ]

    pipeline = Pipeline(steps)
    return pipeline


def train_model(pipeline, target, train_cols, train):
    pipeline.fit(train[train_cols], train[target])


def evaluate_model(pipeline, test, train_cols, target):
    test_predictions = pipeline.predict(test[train_cols])
    test_target = test[target].values

    type(test_predictions), type(test_target)

    print_metrics(test_predictions, test_target)


def main(config_path: str):
    config = load_config(config_path)
    ## DATA LOADING
    train_path = config["FILES"]["TRAIN"]["PATH"]
    test_path = config["FILES"]["TEST"]["PATH"]
    train, test = load_data(train_path, test_path)
    # DATA PREP
    train_cols = [col for col in train.columns if col not in ["id", "target"]]

    categorical_cols = config["categorical_columns"]
    target = config["target_column"]

    preprocessor = get_preprocessor(categorical_cols)

    pipeline = build_pipeline(preprocessor, config["model_params"])

    train_model(pipeline, target, train_cols, train)

    evaluate_model(pipeline, test, train_cols, target)


if __name__ == "__main__":
    main("config.yaml")
