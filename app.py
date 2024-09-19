from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import yaml
import logging
from model.ModelPipeline import ModelPipeline
from datetime import datetime

def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

log_filename = datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S.log')

logging.basicConfig(filename=f'log/{log_filename}', 
       level=logging.INFO, 
       format='%(levelname)s %(asctime)s %(message)s', 
       datefmt='%m/%d/%Y %H:%M:%S')

app = FastAPI()

config_path="config.yaml"
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

# Define the input data model for prediction requests
class PredictionRequest(BaseModel):
    type:str
    sector: str
    net_usable_area:  float
    net_area:  float
    n_rooms:  float
    n_bathroom:  float
    latitude:  float
    longitude:float
    # Add other feature columns as needed

@app.post("/predict/")
def predict(request: PredictionRequest):
    logging.info("Making predictions...")

    input_data = pd.DataFrame([request.dict()])
    
    # Perform prediction using the pipeline
    prediction = model_pipeline.predict(input_data)
    
    return {"predicted_price": prediction[0]}

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}