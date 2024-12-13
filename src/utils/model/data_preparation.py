from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from typing import Tuple, List
from PIL import Image
from sklearn.model_selection import train_test_split
from src.datatypes import Image as Img
from src.api.operations.data_operations import DirectoryOperations

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
            self.transform = transform
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
        for training data and image for test data
        """
        img_path = os.path.join(self.image_dir, self.features.iloc[index])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.training:
            label = self.labels.iloc[index]
            return image, label
        else:
            return image

def prepare_from_csv_and_dir(csv_file: str, img_dir: str, enhance_val: int, resize_val: int | Tuple[int, int] = None) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Divides the data for training and testing datasets and prepares them to model input

    Args:
        enhance_val (int): Data replication number
        csv_file (str): Path to the csv file with image names and labels.
        image_dir (str): Directory with all the images.
        resize_val (int | Tuple[int, int]): Shape of resized image
    
    Returns:
        Tuple[DataLoader, DataLoader, List[int]]: 
         - Training DataLoader instance
         - Test DataLoader instance
         - List of test labels
    """
    data = pd.read_csv(csv_file)

    list_plots = DirectoryOperations.list_directory(img_dir)
    data.iloc[:, 0] = data.iloc[:, 0].str.replace("-", "_", regex=False)
    data.iloc[:, 0] += ".png"
    data = data[data.iloc[:, 0].isin(list_plots)]

    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
    
    train = []
    test = []

    if resize_val:
        transform = transforms.Compose([
                    transforms.Resize(resize_val),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=180),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=180),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    for _ in range(enhance_val):
        train_dataset = BuildTrainTestDataset(x_train, y_train, img_dir, training=True, transform=transform)
        test_dataset = BuildTrainTestDataset(x_test, y_test, img_dir, training=False, transform=transform)
        
        train.append(train_dataset)
        test.append(test_dataset)

    train_combined, test_combined = ConcatDataset(train), ConcatDataset(test)

    train_loader = DataLoader(train_combined, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_combined, batch_size=32, shuffle=False)
    
    y_combined = pd.concat([y_test] * enhance_val, ignore_index=True)
    return train_loader, test_loader, y_combined.tolist()

