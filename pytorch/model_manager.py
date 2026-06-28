import os
import joblib
import torch

from model import ScorePredictorModel


class ModelManager:
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "score_model.pth")
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

    @staticmethod
    def ensure_model_dir():
        os.makedirs(ModelManager.MODEL_DIR, exist_ok=True)

    @staticmethod
    def save_model(model: ScorePredictorModel, path: str = None):
        ModelManager.ensure_model_dir()

        save_path = path or ModelManager.MODEL_PATH

        torch.save(model.state_dict(), save_path)

        print(f"Model saved: {save_path}")

    @staticmethod
    def load_model(
        input_size: int,
        path: str = None,
        device: str = "cpu"
    ) -> ScorePredictorModel:

        load_path = path or ModelManager.MODEL_PATH

        model = ScorePredictorModel(input_size)

        model.load_state_dict(
            torch.load(
                load_path,
                map_location=torch.device(device)
            )
        )

        model.to(device)
        model.eval()

        return model

    @staticmethod
    def save_preprocessor(preprocessor, path: str = None):
        ModelManager.ensure_model_dir()

        save_path = path or ModelManager.PREPROCESSOR_PATH

        joblib.dump(preprocessor, save_path)

        print(f"Preprocessor saved: {save_path}")

    @staticmethod
    def load_preprocessor(path: str = None):
        load_path = path or ModelManager.PREPROCESSOR_PATH

        return joblib.load(load_path)

    @staticmethod
    def save_checkpoint(
        model: ScorePredictorModel,
        optimizer,
        epoch: int,
        loss: float,
        input_size: int,
        path: str = None
    ):
        ModelManager.ensure_model_dir()

        save_path = path or ModelManager.CHECKPOINT_PATH

        checkpoint = {
            "epoch": epoch,
            "input_size": input_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(checkpoint, save_path)

        print(f"Checkpoint saved: {save_path}")

    @staticmethod
    def load_checkpoint(
        optimizer,
        path: str = None,
        device: str = "cpu"
    ):
        load_path = path or ModelManager.CHECKPOINT_PATH

        checkpoint = torch.load(
            load_path,
            map_location=torch.device(device)
        )

        input_size = checkpoint["input_size"]

        model = ScorePredictorModel(input_size)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.to(device)

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        print(f"Checkpoint loaded from epoch {epoch}")

        return model, optimizer, epoch, loss