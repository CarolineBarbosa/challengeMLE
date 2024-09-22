from fastapi import FastAPI, Depends
from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from typing import Annotated
from pydantic import BaseModel
import pandas as pd
import logging
from main import main, load_config
from datetime import datetime

### INITIATING LOG
log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(
    filename=f"log/{log_filename}",
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filemode="a",
)

### INITIATING APP
app = FastAPI()

model_pipeline = main("config.yaml")

### API KEY AUTH
SECRETS = load_config("secrets.yaml")  # Read API key from environment variable
API_KEY = SECRETS["API_KEY"]
API_KEY_NAME = "access-token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key in API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


class PredictionRequest(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float


@app.post("/predict/")
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    logging.info("Received prediction request: %s", request.json())
    input_data = pd.DataFrame([request.dict()])

    prediction = model_pipeline.predict(input_data)
    prediction = round(prediction[0], 2)
    logging.info("Prediction request returned: CLP $ %s", prediction)
    return {"Predicted Price CLP $ ": prediction}


@app.get("/")
def read_root():
    return {"Hello": "World"}
