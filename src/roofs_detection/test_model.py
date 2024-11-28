from src.roofs_detection.roof_detector import RoofDetector
from tqdm import tqdm
import torch
from typing import List


def test_model(test_loader: torch.Tensor, model_path: str) -> List[int]:
    """
    Tests the model
    
    Args:
        test_loader: (torch.Tensor): The testing data DataLoader instance
        model_path (str): Path to the model
    
    Returns:
        List[int]: List of predictions
    """
    model = RoofDetector()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    labels = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            outputs = model(data)
            predicted = (outputs.squeeze() > 0.5).int()
            predictions.extend(predicted.tolist())
            labels.extend(label.tolist())

    return predictions, labels
