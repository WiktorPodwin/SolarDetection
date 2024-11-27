from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import pandas as pd
import os
from typing import Tuple
from PIL import Image
from sklearn.model_selection import train_test_split


class BuildDataset(Dataset):
    def __init__(self, labels: pd.DataFrame, image_dir: str, transform: transforms = None):
        """
        Args:
            labels (pd.DataFrame): DataFrame storing labels
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = labels
        self.image_dir = image_dir           
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
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample (image and label) at the given index
        """
        img_path = os.path.join(self.image_dir, self.data.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[index, 1])

        image = self.transform(image)

        return image, label
    
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


def prepare_data(csv_file: str, img_dir: str) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the data for training and testing datasets and prepares them to model input

    Args:
        csv_file (str): Path to the csv file with image names and labels.
        image_dir (str): Directory with all the images.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoader instances
    """
    data = pd.read_csv(csv_file)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    train_dataset = BuildDataset(train,
                                 img_dir)
    train_loader = train_dataset.dataloader()
    
    test_dataset = BuildDataset(test,
                                img_dir)
    test_loader = test_dataset.dataloader()

    return train_loader, test_loader
