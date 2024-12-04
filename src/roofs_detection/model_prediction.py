from src.roofs_detection.roof_detector import RoofDetector
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from typing import List


def predict(dataloader: DataLoader, model_path: str) -> List[int]:
    """
    Predict new labels
    
    Args:
        dataloader: (DataLoader: The testing DataLoader instance
        model_path (str): Path to the model
    
    Returns:
        List[int]: List of predicted labels
    """
    model = RoofDetector()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    predictions = []

    model.eval()
    with torch.no_grad():
        for image in tqdm(dataloader):
            outputs = model(image)
            predicted = (outputs.squeeze() > 0.5).int()
            predictions.extend(predicted.tolist())

    return predictions

