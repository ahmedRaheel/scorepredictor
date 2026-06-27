import os
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERIC_COLUMNS = [
    "Match ID",
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


def train():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/score.csv")

    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    input_shape = X_train.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=4,
        callbacks=[early_stop],
        verbose=1
    )

    loss, mae = model.evaluate(X_test, y_test)

    print(f"Test Loss: {loss}")
    print(f"Test MAE : {mae}")

    model.save("models/score_model.keras")
    joblib.dump(preprocessor, "models/score_preprocessor.pkl")

    print("Model and preprocessor saved successfully.")


if __name__ == "__main__":
    train()