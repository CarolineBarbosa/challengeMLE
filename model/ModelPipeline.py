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

    def load_data(self):
        self.train_data = pd.read_csv(self.config["FILES"]["TRAIN"]["PATH"])
        self.test_data = pd.read_csv(self.config["FILES"]["TEST"]["PATH"])

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
        print(
            "RMSE: ",
            np.sqrt(mean_squared_error(self.test_prediction, self.test_target)),
        )
        print(
            "MAPE: ",
            mean_absolute_percentage_error(self.test_prediction, self.test_target),
        )
        print("MAE : ", mean_absolute_error(self.test_prediction, self.test_target))

    def evaluate_model(self):
        self.test_target = self.test_data[self.config["target_column"]].values
        test_cols = list(self.train_cols) 
        test_cols.remove(self.config["target_column"])
        
        self.test_prediction = self.pipeline.predict(self.test_data[test_cols])

        self._print_metrics()

