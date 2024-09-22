import pandas as pd
from sqlalchemy import create_engine


class DataLoad:
    def __init__(self, path_or_query: str):
        self.source = path_or_query

    def load_file(self):
        if self.source.endswith(".csv"):
            file = pd.read_csv(self.source)
        elif self.source.endswith(".xlsx"):
            file = pd.read_excel(self.source)
        else:
            engine = create_engine("YOUR CONNECTION PATH HERE")
            file = pd.read_sql(self.source, engine)
        return file
