# ccpp.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# Serve static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

with open("CCPP_Systems.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]

class CCPPFeatures(BaseModel):
    AT: float
    V: float
    AP: float
    RH: float

@app.get("/")
async def home():
    """Serve the main HTML file"""
    return FileResponse("index.html")

@app.post("/predict")
async def predict(features: CCPPFeatures):
    """Predict net hourly electricity generation (PE)"""
    input_data = pd.DataFrame([features.model_dump()])
    print("Input data:", input_data)
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return {"predicted_energy": float(prediction[0])}