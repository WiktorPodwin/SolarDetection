from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from typing import List


def predict(model: Tensor, dataloader: DataLoader, model_path: str) -> List[int]:
    """
    Predict new labels
    
    Args:
        model (Tensor): Model initiation class
        dataloader: (DataLoader): The testing DataLoader instance
        model_path (str): Path to the model
    
    Returns:
        List[int]: List of predicted labels
    """
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    predictions = []

    model.eval()
    with torch.no_grad():
        for image in tqdm(dataloader):
            outputs = model(image)
            outputs = outputs.detach()
            pred = (outputs.squeeze() > 0.5).int()
            predictions.extend(pred.tolist())

    return predictions

