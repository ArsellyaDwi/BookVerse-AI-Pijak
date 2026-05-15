from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

personality_app = FastAPI()

personality_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "trained_model" / "personality_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
    'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',
    'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',
    'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
    'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'
]

class PersonalityRequest(BaseModel):
    answers: dict

@personality_app.get("/")
def root():
    return {"status": "running", "model_loaded": True}

@personality_app.post("/predict")
def predict(request: PersonalityRequest):
    input_dict = {name: request.answers.get(name, 3) for name in FEATURE_NAMES}
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)
    
    return {
        "extroversion": round(float(prediction[0][0]) * 20, 2),
        "neuroticism": round(float(prediction[0][1]) * 20, 2),
        "agreeableness": round(float(prediction[0][2]) * 20, 2),
        "conscientiousness": round(float(prediction[0][3]) * 20, 2),
        "openness": round(float(prediction[0][4]) * 20, 2)
    }