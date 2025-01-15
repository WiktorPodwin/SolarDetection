from solar_detection.utils.model import BuildTestDataset
from solar_detection.datatypes import Image as Img
from torchvision import transforms
from typing import List, Tuple
from torch.utils.data import DataLoader

def prepare_for_prediction(potential_roofs: List[Img], img_shape: int = 128, batch_size: int = 8) -> Tuple[DataLoader, List[str]]:
    """
    Prepares data for the prediction 
    
    Args:
        potential_roofs (List[Img]): List of Image class storing plots id and images
        img_shape (int): Size to reshape images
        batch_size (int): Number of samples in batch

    Returns:
        Tuple[DataLoader, List[str]]: 
            - DataLoader instance storing data to prediction
            - List storing fields id
    """
    extracted_roofs = []
    fields_id = []
    for roof in potential_roofs:
        extracted_roofs.append(roof.potential_building_transformed)
        fields_id.append(roof.new_name[:-4])

    transform = transforms.Compose([
                transforms.Resize((img_shape, img_shape)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    
    data = BuildTestDataset(extracted_roofs, transform)
    data_loader = data.dataloader(batch_size)
    return data_loader, fields_id