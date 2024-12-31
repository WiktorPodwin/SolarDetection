from src.utils.model import BuildTestDataset
from src.datatypes import Image as Img
from torchvision import transforms
from typing import List
from torch.utils.data import DataLoader

def prepare_for_prediction(potential_roofs: List[Img], img_shape: int = 128, batch_size: int = 8) -> DataLoader:
    """
    Prepares data for the prediction 
    
    Args:
        potential_roofs (List[Img]): List of Image class storing plots id and images
        img_shape (int): Size to reshape images
        batch_size (int): Number of samples in batch

    Returns:
        DataLoader: DataLoader instance storing data to prediction
    """
    extracted_roofs = [img.potential_building_transformed for img in potential_roofs]
    transform = transforms.Compose([
                transforms.Resize(img_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    
    data = BuildTestDataset(extracted_roofs, transform)
    data_loader = data.dataloader(batch_size)
    return data_loader