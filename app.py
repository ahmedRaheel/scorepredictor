from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import tensorflow as tf
from predict import predict_score
model = tf.keras.models.load_model("models/score_model.keras")
preprocessor = joblib.load("models/score_preprocessor.pkl")

app = FastAPI()


class ScoreRequest(BaseModel):
    match_id: int
    overs_played: float
    wickets_lost: int
    run_rate: float
    home_away: str
    opponent_strength: int
    pitch_condition: str
    weather: str

@app.post("/predict")
def predict(request: ScoreRequest):

    predicted_score = predict_score(
                                    request.overs_played,
                                    request.wickets_lost,
                                    request.run_rate,
                                    request.home_away,
                                    request.opponent_strength,
                                    request.pitch_condition,
                                    request.weather        
                  )
    return {
        "predictedScore": round(predicted_score, 2)
    }
