from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from typing import Tuple, List
from PIL import Image
from sklearn.model_selection import train_test_split
from src.datatypes import Image as Img

class BuildTestDataset(Dataset):
    """
    Class to data prepare data for model predictions
    """
    def __init__(self, potential_roofs: List[Img], transform: transforms = None):
        """
        Args:
            image_dir (str): Directory path with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.potential_roofs = potential_roofs
        if transform:
            self.transform = transforms 
        else:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self) -> int:
        """
        Calculates length of data
        
        Returns:
            int: Data length
        """
        return len(self.potential_roofs)
    
    def __getitem__(self, index) -> torch.Tensor:
        """
        Returns a sample at the given index
        """
        Img = self.potential_roofs[index]
        image = Img.potential_building_transformed

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.transform(image)

        return image 
    
    def dataloader(self, batch_size: int = 32) -> DataLoader:
        """
        Creates a DataLoader from the current dataset instance
        
        Args:
            batch_size (int): Number of samples per batch
        
        Returns:
            DataLoader: A DataLoader instance
        """
        dataloader = DataLoader(self, batch_size=batch_size)
        return dataloader

def prepare_for_prediction(potential_roofs: List[Img]) -> DataLoader:
    """
    Prepares data for the prediction 
    
    Args:
        potential_roofs (List[Img]): List of Image class storing plot ids and images
    
    Returns:
        DataLoader: DataLoader instance
    """
    data = BuildTestDataset(potential_roofs)
    data_loader = data.dataloader()
    return data_loader



class BuildTrainTestDataset(Dataset):
    """
    Class to data prepare data for model training input
    """
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame, image_dir: str, training: bool = True, transform: transforms = None):
        """
        Args:
            features (pd.DataFrame): DataFrame storing features.
            labels (pd.DataFrame): DataFrame storing labels
            image_dir (str): Directory with all the images.
            training (bool): If dataset for training
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.features = features
        self.labels = labels
        self.image_dir = image_dir
        self.training = training    
        if transform:
            self.transform = transforms 
        else:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self) -> int:
        """
        Calculates length of data
        
        Returns:
            int: Data length
        """
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample (image and label) at the given index
        """
        img_path = os.path.join(self.image_dir, self.features.iloc[index])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.training:
            label = self.labels.iloc[index]
            return image, label
        else:
            return image
    
    def dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Creates a DataLoader from the current dataset instance
        
        Args:
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the dataset
        
        Returns:
            DataLoader: A DataLoader instance
        """
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return dataloader


def prepare_from_csv_and_dir(csv_file: str, img_dir: str) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Divides the data for training and testing datasets and prepares them to model input

    Args:
        csv_file (str): Path to the csv file with image names and labels.
        image_dir (str): Directory with all the images.
    
    Returns:
        Tuple[DataLoader, DataLoader, List[int]]: 
         - Training DataLoader instance
         - Test DataLoader instance
         - List of test labels
    """
    data = pd.read_csv(csv_file)
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)

    train_dataset = BuildTrainTestDataset(x_train, y_train, img_dir)
    train_loader = train_dataset.dataloader()
    
    test_dataset = BuildTrainTestDataset(x_test, y_test, img_dir, training=False)
    test_loader = test_dataset.dataloader(shuffle=False)

    return train_loader, test_loader, y_test.tolist()

