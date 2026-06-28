import pandas as pd
import joblib

model = joblib.load("models/sklearn_score_model.pkl")

sample = pd.DataFrame([
    {
        "Overs Played": 7,
        "Wickets Lost": 1,
        "Run Rate": 11.04,
        "Opponent Strength": 9,
        "Home/Away": "Home",
        "Pitch Condition": "Batting",
        "Weather": "Sunny"
    }
])

prediction = model.predict(sample)

print(f"Predicted Score: {prediction[0]:.2f}")