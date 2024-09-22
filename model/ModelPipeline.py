import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
import logging
from .DataLoad import DataLoad


class ModelPipeline:
    def __init__(self, config_file: dict):
        self.config = config_file
        self.train_data = None
        self.test_data = None
        self.preprocessor = None
        self.pipeline = None
        self.test_prediction = None
        self.test_target = None
        self.train_cols = None
        self.test_cols = None

    def load_data(self):
        train_loader = DataLoad(self.config["FILES"]["TRAIN"]["PATH"])
        self.train_data = train_loader.load_file()
        test_loader = DataLoad(self.config["FILES"]["TEST"]["PATH"])
        self.test_data = test_loader.load_file()

    def get_preprocessor(self):
        categorical_cols = self.config["categorical_columns"]
        categorical_transformer = TargetEncoder()

        self.preprocessor = ColumnTransformer(
            transformers=[("categorical", categorical_transformer, categorical_cols)]
        )

    def build_pipeline(self):
        model_params = self.config["model_params"]

        steps = [
            ("preprocessor", self.preprocessor),
            (
                "model",
                GradientBoostingRegressor(**model_params),
            ),
        ]

        self.pipeline = Pipeline(steps)

    def _columns_selection(self):
        self.train_cols = [
            col for col in self.train_data.columns if col not in ["id", "target"]
        ]

    def train_model(self):
        self._columns_selection()
        target = self.config["target_column"]

        self.pipeline.fit(self.train_data[self.train_cols], self.train_data[target])

    def _print_metrics(self):
        rmse = np.sqrt(mean_squared_error(self.test_prediction, self.test_target))
        mape = mean_absolute_percentage_error(self.test_prediction, self.test_target)
        mae = mean_absolute_error(self.test_prediction, self.test_target)
        logging.info("MODEL ERROR METRICS")
        logging.info(f"RMSE: {rmse}")
        logging.info(f"MAPE: {mape}")
        logging.info(f"MAE : {mae}")

    def evaluate_model(self):
        print(self.test_data)
        self.test_target = self.test_data[self.config["target_column"]].values
        self.test_cols = list(self.train_cols)
        self.test_cols.remove(self.config["target_column"])
        self.test_prediction = self.pipeline.predict(self.test_data[self.test_cols])
        self._print_metrics()

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        test_cols = [col for col in self.train_cols if col in test_data.columns]
        predictions = self.pipeline.predict(test_data[test_cols])
        return predictions
