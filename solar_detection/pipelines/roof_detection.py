from pathlib import Path
from typing import List
import os
from typing import Tuple
import torch
from solar_detection.datatypes import Image
from solar_detection.utils import upload_csv_file
from solar_detection.utils.model import prepare_from_csv_and_dir, train_model, predict, EvaluateMetrics
from solar_detection.roofs_detector.prepare_data import prepare_for_prediction
from solar_detection.processing.image_processing.image_process import ImageProcessing
from solar_detection.api.operations.data_operations import DirectoryOperations
from solar_detection.roofs_detector.roof_detector import RoofDetector

def extract_potential_roofs(base_dir: str, depth_dir: str, potential_roofs_dir: str) -> List[Image]:
    """
    Uploads images from the depth directory, extracts potential candidates for roofs and save them
    
    Args:
        base_dir (str): A directory storing original images
        depth_dir (str): A directory storing images with a depth
        potentoal_roofs_dir (str): A directory to save potential roofs
    
    Returns:
        List[Image]: List of Image classes, storing potential roof parameters
    """
    dir_oper = DirectoryOperations
    dir_oper.create_directory(potential_roofs_dir)
    files = dir_oper.list_directory(depth_dir)
    potential_roofs = []
    image_processing = ImageProcessing()
    # print("Extracting potential roofs")
    # print(files)

    for file in files:
        file_path = Path(file)
        if file_path.suffix == ".jpg":
            png_original_image_path = os.path.join(base_dir, file_path.with_suffix('.png').name)

            input_file_path = os.path.join(depth_dir, file_path)
            building_file_path = os.path.join(potential_roofs_dir, file_path.with_suffix('.png').name)
            # print(f"{input_file_path=}")
            # print(f"{building_file_path=}")
            segmented_image = image_processing.load_image(input_file_path)
            original_image = image_processing.load_image(png_original_image_path)
            low_boundary = (0, 0, 0)
            high_boundary = (65, 30, 54)
            shapes = image_processing.generate_mask_around_potential_building(segmented_image, low_boundary, high_boundary)
            
            for i, shape in enumerate(shapes):
                plot_id_nosuffix = file_path.with_suffix('')
                plot_id_i = str(plot_id_nosuffix) + f"_{i}.png"
                building_file_path = os.path.join(potential_roofs_dir, plot_id_i)

                building_mask = image_processing.apply_mask(original_image, shape)
                extracted_rectangle = image_processing.crop_rectangle_around_plot(building_mask, return_with_mask=True)
                extracted_buliding = image_processing.crop_plot(extracted_rectangle)
                image_processing.save_image(building_file_path, extracted_buliding)

                potential_roofs.append(Image(name=str(file_path.with_suffix('.png')), 
                                             new_name=plot_id_i, 
                                             potential_building=building_mask, 
                                             potential_building_transformed=extracted_buliding
                                             ))

    return potential_roofs


def generate_model(device: torch.device,
                   csv_file_path: str, 
                   label: str, 
                   potential_roofs_dir: str, 
                   num_epochs: int,
                   model_path: str, 
                   metrics_dir: str,
                   plot_id: str = "id",
                   data_multiplier: int = 1,
                   resize_val: int = None,
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
        label (str): Name of the columns storing labels in csv file
        potential_roofs_dir (str): A path to the directory storing roofs
        num_epochs (int): Number of epochs
        model_path (str): A path to storing the model
        metrics_dir (str): A directory path for storing metrics
        plot_id (str): Name of the columns storing plot ID in csv file
        data_multiplier (int): Data multiplication number
        resize_val (int): Shape of resized image
        batch_size (int): Number of samples in batch
        learning_rate (float): Learning rate for model training
        step_size (int): Step size in learning rate scheduler
        accumulation_steps (int): Number of batches to accumulate gradients before performing an optimizer step
    """
    train_loader, test_loader, labels, class_distribution = prepare_from_csv_and_dir(csv_file_path, 
                                                                                     label, 
                                                                                     potential_roofs_dir, 
                                                                                     data_multiplier, 
                                                                                     plot_id, 
                                                                                     resize_val, 
                                                                                     batch_size)
    model = RoofDetector().to(device)
    history = train_model(device, model, train_loader, class_distribution, num_epochs=num_epochs, lr=learning_rate, step_size=step_size, accumulation_steps=accumulation_steps, save_path=model_path)
    predictions = predict(device, model, test_loader, model_path)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)
    evaluate_metrics.display_history(history, metrics_dir)


def prediction(device: torch.device, 
               potential_roofs: List[Image], 
               model_path: str, 
               img_dir: str, 
               img_shape: int = 128, 
               batch_size: int = 8
               ) -> Tuple[List[int], List[str]]:
    """
    Prepares the data to roof predictions, selects only images with predicted roof, 
    combines them and saves to specified directory

    Args:
        device (torch.device): Torch device
        potential_roofs (List[Image]): List of Image classes, storing potential roof parameters
        model_path (str): Path to the saved model
        img_dir (str): Directory path to store extracted roofs
        img_shape (int): Size to reshape images
        batch_size (int): Number of samples in batch

    Returns:
        Tuple[List[int], List[str]]: 
            - List of predicted labels
            - List of plot ids
    """
    dataloader, labels = prepare_for_prediction(potential_roofs, img_shape, batch_size)

    model = RoofDetector().to(device)
    pred = predict(device, model, dataloader, model_path)
    # roofs = dict()
    # DirectoryOperations.create_directory(img_dir)
    # img_processing = ImageProcessing()

    # for i, Image in enumerate(potential_roofs):
    #     if pred[i] == 1:
    #         masked_roof = Image.potential_building
    #         cropped_roof = img_processing.crop_plot(masked_roof)

    #         if Image.name in roofs:
    #             roofs[Image.name].append(cropped_roof)
    #         else:
    #             roofs[Image.name] = [cropped_roof]

    # for key, values in roofs.items():
    #     roofs_extracted = img_processing.connect_images(values)
    #     save_path = os.path.join(img_dir, key)
    #     img_processing.save_image(save_path, roofs_extracted)

    return pred, labels


