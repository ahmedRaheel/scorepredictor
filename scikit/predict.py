import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

TARGET_COLUMN = "Predicted Score"

df = pd.read_csv("data/score.csv")

if "Match ID" in df.columns:
    df.drop(columns=["Match ID"], inplace=True)

X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
y = df[TARGET_COLUMN]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")

joblib.dump(model, "models/sklearn_score_model.pkl")
print("Scikit-learn model saved.")