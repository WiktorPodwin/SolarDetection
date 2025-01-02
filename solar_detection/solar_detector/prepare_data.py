from solar_detection.api.operations.data_operations import DirectoryOperations
from solar_detection.processing.image_processing.image_process import ImageProcessing
from solar_detection.utils.model import BuildTestDataset
from torchvision import transforms
from typing import Tuple, List
from torch.utils.data import DataLoader
import os

def prepare_for_prediction(dir_path: str, img_shape: int = 256, batch_size: int = 8) -> Tuple[DataLoader, List[str]]:
    """
    Prepares data for the prediction 
    
    Args:
        dir_path (str): Path to the directory stoing images for prediction
        img_shape (int): Size to reshape images
        batch_size (int): Number of samples in batch
    Returns:
        Tuple[DataLoader, List[str]]: 
            - DataLoader instance storing data to prediction
            - List storing plots id
    """
    files = DirectoryOperations.list_directory(dir_path)
    img_processing = ImageProcessing()
    images = []
    file_names = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        image = img_processing.load_image(file_path)
        images.append(image)
        file_names.append(file[:-4])
    
    transform = transforms.Compose([
                transforms.Resize(img_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    data = BuildTestDataset(images=images, transform=transform)
    data_loader = data.dataloader(batch_size)
    return data_loader, file_names