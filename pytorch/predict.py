import pandas as pd
import torch

from model_manager import ModelManager


def predict(
    overs_played,
    wickets_lost,
    run_rate,
    home_away,
    opponent_strength,
    pitch_condition,
    weather
):
    preprocessor = ModelManager.load_preprocessor()

    sample = pd.DataFrame([
        {
            "Overs Played": overs_played,
            "Wickets Lost": wickets_lost,
            "Run Rate": run_rate,
            "Opponent Strength": opponent_strength,
            "Home/Away": home_away,
            "Pitch Condition": pitch_condition,
            "Weather": weather
        }
    ])

    X = preprocessor.transform(sample)

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    input_size = X.shape[1]

    model = ModelManager.load_model(input_size)
    model.eval()

    with torch.no_grad():
        prediction = model(X)

    return prediction.item()


if __name__ == "__main__":
    score = predict(
        overs_played=7,
        wickets_lost=1,
        run_rate=11.04,
        home_away="Home",
        opponent_strength=5,
        pitch_condition="Bowling",
        weather="Sunny"
    )

    print(f"Predicted Score: {score:.2f}")