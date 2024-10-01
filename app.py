from fastapi import FastAPI, Depends
from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import logging
from datetime import datetime
from model.Model import train_model, load_config
from fastapi.responses import HTMLResponse


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
config_file_name = "config_files/modelConfig.yaml"
model_pipeline = train_model(config_file_name)

### API KEY AUTH
config = load_config(config_file_name)
secret_file_name = config["FILES"]["API_KEY"]["PATH"]
SECRETS = load_config(secret_file_name)
API_KEY = SECRETS["API_KEY"]
API_KEY_NAME = "access-token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key in API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# class PredictionRequest(BaseModel):
#     type: str
#     sector: str
#     net_usable_area: float
#     net_area: float
#     n_rooms: float
#     n_bathroom: float
#     latitude: float
#     longitude: float


# @app.post("/predict/")
# async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
#     logging.info("Received prediction request: %s", request.json())
#     input_data = pd.DataFrame([request.dict()])

#     prediction = model_pipeline.predict(input_data)
#     prediction = round(prediction[0], 2)
#     logging.info("Prediction request returned: CLP $ %s", prediction)
#     return {"Predicted Price CLP $ ": prediction}


@app.post("/Price Prediction/")
async def predict(
    type: str,
    sector: str,
    net_usable_area: float,
    net_area: float,
    n_rooms: float,
    n_bathroom: float,
    latitude: float,
    longitude: float,
    api_key: str = Depends(get_api_key),
):
    input_data = pd.DataFrame(
        [
            {
                "type": type,
                "sector": sector,
                "net_usable_area": net_usable_area,
                "net_area": net_area,
                "n_rooms": n_rooms,
                "n_bathroom": n_bathroom,
                "latitude": latitude,
                "longitude": longitude,
            }
        ]
    )

    prediction = model_pipeline.predict(input_data)
    prediction = round(prediction[0], 2)
    logging.info("Prediction request returned: CLP $ %s", prediction)
    return {f"Predicted Price CLP $ {prediction}"}


@app.get("/", response_class=HTMLResponse)
def read_root():
    msg = """
    <h2>Welcome to the Property-Friends Real Estate API!</h2>
    <p>To get a property price prediction, go to the <a href="/docs">/docs</a> endpoint for an interactive API documentation.</p>
    <p>To directly make predictions, use the <b>Price Prediction Post/</b> endpoint. Select 'Try it out' and include the necessary property details in the request body before executing the price prediction.</p>
    <p>To get a property price prediction, you need an API key. Please, make sure you have access and include your API key to get autorization when making requests.</p>
    """
    return msg
