from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from typing import Tuple, List
from PIL import Image
from sklearn.model_selection import train_test_split
from solar_detection.api.operations.data_operations import DirectoryOperations
from solar_detection.processing.image_processing.image_process import ImageProcessing


class BuildTestDataset(Dataset):
    """
    Class to data prepare data for model predictions
    """
    def __init__(self, images: List[np.ndarray], transform: transforms = None):
        """
        Args:
            images (List[np.ndarray]): List storing images to prediction.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
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
        return len(self.images)
    
    def __getitem__(self, index) -> torch.Tensor:
        """
        Returns a sample at the given index
        """
        image = self.images[index]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.transform(image)

        return image 
    
    def dataloader(self, batch_size: int = 8) -> DataLoader:
        """
        Creates a DataLoader from the current dataset instance
        
        Args:
            batch_size (int): Number of samples per batch
        
        Returns:
            DataLoader: A DataLoader instance
        """
        dataloader = DataLoader(self, batch_size=batch_size)
        return dataloader



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
        img_process = ImageProcessing()
        image = img_process.load_image(img_path)
        image = img_process.crop_rectangle_around_plot(image, False, False, False)
        image = Image.fromarray(image)
        # image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.training:
            label = self.labels.iloc[index]
            return image, label
        else:
            return image

def prepare_from_csv_and_dir(csv_file: str, 
                             img_dir: str, 
                             data_multiplier: int, 
                             resize_val: int | Tuple[int, int] = None,
                             batch_size: int = 32) -> Tuple[DataLoader, DataLoader, List[int], pd.Series]:
    """
    Divides the data for training and testing datasets and prepares them to model input

    Args:
        data_multiplier (int): Data multiplication number
        csv_file (str): Path to the csv file with image names and labels.
        image_dir (str): Directory with all the images.
        resize_val (int | Tuple[int, int]): Shape of resized image
    
    Returns:
        Tuple[DataLoader, DataLoader, List[int], pd.Series]: 
         - Training DataLoader instance
         - Test DataLoader instance
         - List of test labels
         - Data classes distribution
    """
    data = pd.read_csv(csv_file)

    list_plots = DirectoryOperations.list_directory(img_dir)
    data.iloc[:, 0] = data.iloc[:, 0].str.replace("-", "_", regex=False)
    data.iloc[:, 0] += ".png"
    data = data[data.iloc[:, 0].isin(list_plots)]

    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
    
    class_distribution = y_data.value_counts()
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
        
    for _ in range(data_multiplier):
        train_dataset = BuildTrainTestDataset(x_train, y_train, img_dir, training=True, transform=transform)
        test_dataset = BuildTrainTestDataset(x_test, y_test, img_dir, training=False, transform=transform)
        
        train.append(train_dataset)
        test.append(test_dataset)

    train_combined, test_combined = ConcatDataset(train), ConcatDataset(test)

    train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_combined, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    y_combined = pd.concat([y_test] * data_multiplier, ignore_index=True)
    return train_loader, test_loader, y_combined.tolist(), class_distribution

