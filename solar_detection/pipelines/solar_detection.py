import pandas as pd
import torch

from solar_detection.utils.model import prepare_from_csv_and_dir, train_model, predict, EvaluateMetrics
from solar_detection.solar_detector.solar_detector import SolarRoofDetector
from solar_detection.solar_detector.prepare_data import prepare_for_prediction
from solar_detection.processing.image_processing.image_process import ImageProcessing
from solar_detection.api.operations.data_operations import DirectoryOperations
from typing import Tuple
from typing import List

def preprocess_data(potential_roof_dir: str, roofs_dir: str, csv_file_path: str, label: str, id: str = "id"):
    """
    Preprocess the data before solar-roof detection

    Args:
        potential_roof_dir (str): Path to the directory storing potential roofs
        roofs_dir (str): Path to the directory to store preprocessed data
        csv_file_path (str): Path to the csv file storing label
        label (str): Column name storing labels
        id (str): Column name storing fields id
    """
    df = pd.read_csv(csv_file_path)
    df_roof = df[df[label] == 1]
    
    DirectoryOperations.create_directory(roofs_dir)
    img_processing = ImageProcessing()

    for _, field in df_roof.iterrows():
        input_path = potential_roof_dir + "/" + field[id] + ".png"
        output_path = roofs_dir + "/" + field[id] + ".png"
        image = img_processing.load_image(input_path)
        img_processing.save_image(output_path, image)
        


def generate_model(device: torch.device,
                   csv_file_path: str,
                   roof_label: str,
                   solar_label: str,
                   potential_roofs_dir: str, 
                   roofs_dir: str, 
                   num_epochs: int,
                   model_path: str, 
                   metrics_dir: str,
                   data_multiplier: int = 1,
                   resize_val: int | Tuple[int, int] = None,
                   batch_size: int = 32,
                   learning_rate: float = 0.0001,
                   step_size: int = None,
                   accumulation_steps: int = 1
                   ) -> None:
    """
    Prepares the data, trains model and tests his performance
    
    Args:
        device (torch.device): The torch device
        csv_file_path (str): A path to the csv file storing plots ID and roofs labels
        roof_label (str): Column name storing roof labels
        solar_label (str): Column name storing solar labels
        potential_roofs_dir (str): A path to the directory storing potential roofs
        roofs_dir (str): A path to the directory storing roofs
        num_epochs (int): Number of epochs
        model_path (str): A path to storing the model
        metrics_dir (str): A directory path for storing metrics
        data_multiplier (int): Data multiplication number
        resize_val (int | Tuple[int, int]): Shape of resized image
        batch_size (int): Number of samples in batch
        learning_rate (float): Learning rate for model training
        step_size (int): Step size in learning rate scheduler
        accumulation_steps (int): Number of batches to accumulate gradients before performing an optimizer step
    """
    preprocess_data(potential_roofs_dir, roofs_dir, csv_file_path, roof_label)
    train_loader, test_loader, labels, class_distr = prepare_from_csv_and_dir(csv_file_path,
                                                                              solar_label, 
                                                                              roofs_dir, 
                                                                              data_multiplier, 
                                                                              resize_val=resize_val, 
                                                                              batch_size=batch_size)
    model = SolarRoofDetector().to(device)
    history = train_model(device,
                           model, 
                           train_loader, 
                           class_distr, 
                           num_epochs=num_epochs, 
                           lr=learning_rate, 
                           step_size=step_size, 
                           accumulation_steps=accumulation_steps, 
                           save_path=model_path)
    
    predictions = predict(device, model, test_loader, model_path, apply_sigmoid=True)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)
    evaluate_metrics.display_history(history, metrics_dir)


def prediction(device: torch.device, 
               model_path: str, 
               img_dir: str, 
               img_shape: int = 256, 
               batch_size: int = 8
               ) -> Tuple[List[int], List[str]]:
    """
    Prepares the data to roof predictions, selects only images with predicted roof, 
    combines them and saves to specified directory

    Args:
        device (torch.device): Torch device
        model_path (str): Path to the saved model
        img_dir (str): Path to the directory storing roof images
        img_shape (int): Size to reshape images
        batch_size (int): Number of samples in batch

    Returns:
        Tuple[List[int], List[str]]: 
            - List of predicted labels
            - List of plot ids
    """
    data_loader, labels = prepare_for_prediction(img_dir, img_shape, batch_size)
    model = SolarRoofDetector().to(device)
    pred = predict(device, model, data_loader, model_path, apply_sigmoid=True)
    return pred, labels
