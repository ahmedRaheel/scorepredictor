import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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


class ScoreDataset(Dataset):

    def __init__(self, csvPath):
        df = pd.read_csv(csvPath)

        df.drop(columns=["Match ID"], inplace=True)

        X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        y = df[[TARGET_COLUMN]]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_COLUMNS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
            ]
        )

        X_processed = self.preprocessor.fit_transform(X)

        self.X = torch.tensor(
            X_processed,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y.values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]