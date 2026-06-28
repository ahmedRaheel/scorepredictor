import joblib
import pandas as pd
import tensorflow as tf


NUMERIC_COLUMNS = [
    "Overs Played",
    "Wickets Lost",
    "Run Rate",
    "Opponent Strength"
]

CATEGORICAL_COLUMNS = [
    "Home/Away",
    "Pitch Condition",
    "Weather"
]


model = tf.keras.models.load_model("models/score_model.keras")
preprocessor = joblib.load("models/score_preprocessor.pkl")


def predict_score(   
    overs_played,
    wickets_lost,
    run_rate,
    home_away,
    opponent_strength,
    pitch_condition,
    weather
):
    row = pd.DataFrame([{
        "Overs Played": overs_played,
        "Wickets Lost": wickets_lost,
        "Run Rate": run_rate,
        "Home/Away": home_away,
        "Opponent Strength": opponent_strength,
        "Pitch Condition": pitch_condition,
        "Weather": weather
    }])

    row = row[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]

    row_processed = preprocessor.transform(row)

    prediction = model.predict(row_processed, verbose=0)

    return float(prediction[0][0])


if __name__ == "__main__":
    score = predict_score(
        overs_played=7,
        wickets_lost=1,
        run_rate=11.04,
        home_away="Home",
        opponent_strength=9,
        pitch_condition="Batting",
        weather="Sunny"
    )

    print(f"Predicted Score: {score:.2f}")