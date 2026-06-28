from model import ScorePredictorModel
from dataset import  ScoreDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_manager import ModelManager


def train():
    data = ScoreDataset("data/score.csv")
    
    loader = DataLoader(data, batch_size=16, shuffle= True)
    input_size = data.X.shape[1]
    model = ScorePredictorModel(input_size)

    loss_fn = nn.MSELoss()

    optimizer =  torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, targets in loader:
            predictions =  model(features)

            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
            if epoch % 100 == 0 and epoch > 0:
                ModelManager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    input_size= input_size,
                    loss=total_loss
                )
    
    ModelManager.save_model(model)
    ModelManager.save_preprocessor(data.preprocessor)
    print("Training completed.")

if __name__ == "__main__":
    train()